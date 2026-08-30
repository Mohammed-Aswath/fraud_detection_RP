"""Leakage-aware feature engineering for the IEEE-CIS fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.config_loader import load_config


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FeatureArtifacts:
    """Train-fitted values needed to apply the feature transform consistently."""

    high_missing_columns: list[str]
    numeric_medians: dict[str, float]
    target_encoding_maps: dict[str, dict[str, float]]
    target_encoding_stats: dict[str, dict[str, tuple[float, int]]]
    ordinal_encoding_maps: dict[str, dict[str, int]]
    global_fraud_rate: float
    target_column: str
    missing_category: str
    velocity_card_column: str
    velocity_email_column: str
    velocity_windows_seconds: tuple[int, ...]
    velocity_history: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def persistable_copy(self) -> FeatureArtifacts:
        """Return artifacts without run-specific transaction history."""
        return FeatureArtifacts(
            high_missing_columns=self.high_missing_columns,
            numeric_medians=self.numeric_medians,
            target_encoding_maps=self.target_encoding_maps,
            target_encoding_stats=self.target_encoding_stats,
            ordinal_encoding_maps=self.ordinal_encoding_maps,
            global_fraud_rate=self.global_fraud_rate,
            target_column=self.target_column,
            missing_category=self.missing_category,
            velocity_card_column=self.velocity_card_column,
            velocity_email_column=self.velocity_email_column,
            velocity_windows_seconds=self.velocity_windows_seconds,
        )


def _repository_path(path_value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _categorical_columns(data: pd.DataFrame) -> list[str]:
    """Identify object, string, and categorical input columns."""
    return [
        column
        for column in data.columns
        if (
            pd.api.types.is_object_dtype(data[column])
            or pd.api.types.is_string_dtype(data[column])
            or isinstance(data[column].dtype, pd.CategoricalDtype)
        )
    ]


def _normalise_categories(series: pd.Series, missing_category: str) -> pd.Series:
    """Create stable string category keys while preserving missingness explicitly."""
    return series.astype("string").fillna(missing_category).astype(str)


def _fit_artifacts(reference_data: pd.DataFrame, config: dict[str, Any]) -> FeatureArtifacts:
    """Fit imputation and encoding values using the training split only."""
    settings = config["feature_engineering"]
    target_column = settings["target_column"]
    if target_column not in reference_data:
        raise KeyError(f"{target_column} is required to fit target encoders.")

    missing_category = settings["missing_category"]
    feature_data = reference_data.drop(columns=[target_column], errors="ignore")
    missingness = feature_data.isna().mean()
    high_missing_columns = missingness[
        missingness > float(settings["high_missingness_threshold"])
    ].index.tolist()

    numeric_medians: dict[str, float] = {}
    for column in feature_data.select_dtypes(include=np.number).columns:
        if 0 < missingness[column] <= float(settings["high_missingness_threshold"]):
            median = feature_data[column].median()
            if pd.notna(median):
                numeric_medians[column] = float(median)

    categorical_columns = _categorical_columns(feature_data)
    requested_target_columns = settings["target_encoding"]["columns"]
    target_encoded_columns = [
        column for column in requested_target_columns if column in categorical_columns
    ]
    global_fraud_rate = float(reference_data[target_column].mean())
    smoothing = float(settings["target_encoding"]["smoothing"])

    target_encoding_maps: dict[str, dict[str, float]] = {}
    target_encoding_stats: dict[str, dict[str, tuple[float, int]]] = {}
    for column in target_encoded_columns:
        categories = _normalise_categories(reference_data[column], missing_category)
        grouped = (
            pd.DataFrame({"category": categories, "target": reference_data[target_column]})
            .groupby("category", sort=False)["target"]
            .agg(["sum", "count"])
        )
        target_encoding_maps[column] = {
            str(category): float(
                (values["sum"] + smoothing * global_fraud_rate)
                / (values["count"] + smoothing)
            )
            for category, values in grouped.iterrows()
        }
        target_encoding_stats[column] = {
            str(category): (float(values["sum"]), int(values["count"]))
            for category, values in grouped.iterrows()
        }

    ordinal_encoding_maps: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        if column in target_encoded_columns:
            continue
        categories = _normalise_categories(reference_data[column], missing_category)
        ordinal_encoding_maps[column] = {
            category: index for index, category in enumerate(pd.unique(categories))
        }

    velocity_settings = settings["velocity"]
    artifacts = FeatureArtifacts(
        high_missing_columns=high_missing_columns,
        numeric_medians=numeric_medians,
        target_encoding_maps=target_encoding_maps,
        target_encoding_stats=target_encoding_stats,
        ordinal_encoding_maps=ordinal_encoding_maps,
        global_fraud_rate=global_fraud_rate,
        target_column=target_column,
        missing_category=missing_category,
        velocity_card_column=velocity_settings["card_column"],
        velocity_email_column=velocity_settings["email_column"],
        velocity_windows_seconds=tuple(
            int(window) for window in velocity_settings["windows_seconds"]
        ),
    )
    LOGGER.info(
        "Fitted target encoders: %s",
        ", ".join(
            f"{column}={len(mapping):,} categories"
            for column, mapping in artifacts.target_encoding_maps.items()
        ),
    )
    LOGGER.info(
        "Fitted %d numeric median imputers and %d ordinal encoders.",
        len(artifacts.numeric_medians),
        len(artifacts.ordinal_encoding_maps),
    )
    return artifacts


def _apply_missingness_features(
    data: pd.DataFrame, artifacts: FeatureArtifacts
) -> pd.DataFrame:
    """Add high-missing indicators and apply train-derived numeric medians."""
    indicators = {
        f"{column}__is_missing": data[column].isna().astype(np.int8)
        for column in artifacts.high_missing_columns
        if column in data
    }
    if indicators:
        data = pd.concat([data, pd.DataFrame(indicators, index=data.index)], axis=1, copy=False)
    for column, median in artifacts.numeric_medians.items():
        if column in data:
            data[column] = data[column].fillna(median)
    return data


def _apply_categorical_encodings(
    data: pd.DataFrame,
    artifacts: FeatureArtifacts,
    use_leave_one_out: bool,
    smoothing: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Replace raw categoricals with target or ordinal encoded numeric features."""
    dropped_columns: list[str] = []
    encoded_features: dict[str, pd.Series] = {}
    target = data[artifacts.target_column] if artifacts.target_column in data else None

    for column, mapping in artifacts.target_encoding_maps.items():
        if column not in data:
            continue
        categories = _normalise_categories(data[column], artifacts.missing_category)
        if use_leave_one_out:
            if target is None:
                raise KeyError("The training target is needed for leave-one-out encoding.")
            stats = artifacts.target_encoding_stats[column]
            sums = categories.map({key: value[0] for key, value in stats.items()})
            counts = categories.map({key: value[1] for key, value in stats.items()})
            encoded = (
                (sums.astype(float) - target.astype(float) + smoothing * artifacts.global_fraud_rate)
                / (counts.astype(float) - 1 + smoothing)
            )
        else:
            encoded = categories.map(mapping).fillna(artifacts.global_fraud_rate)
        encoded_features[f"{column}__target_rate"] = encoded.astype(np.float32)
        dropped_columns.append(column)

    for column, mapping in artifacts.ordinal_encoding_maps.items():
        if column not in data:
            continue
        categories = _normalise_categories(data[column], artifacts.missing_category)
        encoded_features[f"{column}__ordinal"] = categories.map(mapping).fillna(-1).astype(
            np.int32
        )
        dropped_columns.append(column)

    encoded_data = pd.DataFrame(encoded_features, index=data.index)
    transformed = pd.concat(
        [data.drop(columns=dropped_columns), encoded_data], axis=1, copy=False
    )
    return transformed, dropped_columns


def _velocity_records(data: pd.DataFrame, artifacts: FeatureArtifacts) -> pd.DataFrame:
    """Extract only the fields necessary to compute chronological velocity features."""
    required = {
        "TransactionDT",
        "TransactionAmt",
        artifacts.velocity_card_column,
        artifacts.velocity_email_column,
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Velocity features require columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "_time": pd.to_numeric(data["TransactionDT"], errors="raise"),
            "_card": _normalise_categories(
                data[artifacts.velocity_card_column], artifacts.missing_category
            ),
            "_email": _normalise_categories(
                data[artifacts.velocity_email_column], artifacts.missing_category
            ),
            "_amount": pd.to_numeric(data["TransactionAmt"], errors="coerce").fillna(0.0),
        }
    )


def _add_velocity_features(
    data: pd.DataFrame, artifacts: FeatureArtifacts
) -> pd.DataFrame:
    """Add card and card/email features using strictly earlier TransactionDT values."""
    current = _velocity_records(data, artifacts)
    current["_row_id"] = np.arange(len(current), dtype=np.int64)
    current["_current"] = True

    history = artifacts.velocity_history.copy()
    if history.empty:
        history = pd.DataFrame(columns=["_time", "_card", "_email", "_amount"])
        combined = current.copy()
    else:
        history["_row_id"] = -1
        history["_current"] = False
        combined = pd.concat([history, current], ignore_index=True, copy=False)

    count_features = {
        window: np.zeros(len(data), dtype=np.int32)
        for window in artifacts.velocity_windows_seconds
    }
    amount_features = {
        window: np.zeros(len(data), dtype=np.float32)
        for window in artifacts.velocity_windows_seconds
    }
    amount_zscore = np.full(len(data), np.nan, dtype=np.float32)
    seconds_since_last = np.full(len(data), np.nan, dtype=np.float32)

    for _, group in combined.groupby("_card", sort=False):
        group = group.sort_values("_time", kind="stable")
        times = group["_time"].to_numpy(dtype=np.float64)
        amounts = group["_amount"].to_numpy(dtype=np.float64)
        row_ids = group["_row_id"].to_numpy(dtype=np.int64)
        is_current = group["_current"].to_numpy(dtype=bool)
        cumulative_amount = np.concatenate(([0.0], np.cumsum(amounts)))
        cumulative_squared_amount = np.concatenate(([0.0], np.cumsum(amounts**2)))

        for position in np.flatnonzero(is_current):
            row_id = row_ids[position]
            time = times[position]
            prior_end = np.searchsorted(times, time, side="left")
            for window in artifacts.velocity_windows_seconds:
                prior_start = np.searchsorted(times, time - window, side="left")
                count_features[window][row_id] = prior_end - prior_start
                amount_features[window][row_id] = (
                    cumulative_amount[prior_end] - cumulative_amount[prior_start]
                )
            if prior_end > 1:
                mean = cumulative_amount[prior_end] / prior_end
                variance = cumulative_squared_amount[prior_end] / prior_end - mean**2
                standard_deviation = np.sqrt(max(variance, 0.0))
                if standard_deviation > 0:
                    amount_zscore[row_id] = (amounts[position] - mean) / standard_deviation

    for _, group in combined.groupby(["_card", "_email"], sort=False):
        group = group.sort_values("_time", kind="stable")
        times = group["_time"].to_numpy(dtype=np.float64)
        row_ids = group["_row_id"].to_numpy(dtype=np.int64)
        is_current = group["_current"].to_numpy(dtype=bool)
        for position in np.flatnonzero(is_current):
            prior_end = np.searchsorted(times, times[position], side="left")
            if prior_end:
                seconds_since_last[row_ids[position]] = times[position] - times[prior_end - 1]

    velocity_features: dict[str, np.ndarray] = {}
    for window in artifacts.velocity_windows_seconds:
        window_label = f"{window // 3600}h" if window % 3600 == 0 else f"{window}s"
        velocity_features[f"card1__transaction_count_{window_label}"] = count_features[window]
        velocity_features[f"card1__amount_total_{window_label}"] = amount_features[window]
    velocity_features["card1__amount_zscore_history"] = amount_zscore
    velocity_features["card1_email__seconds_since_last"] = seconds_since_last

    current_history = current[["_time", "_card", "_email", "_amount"]]
    artifacts.velocity_history = (
        current_history.copy()
        if history.empty
        else pd.concat([history[["_time", "_card", "_email", "_amount"]], current_history], ignore_index=True, copy=False)
    )
    return pd.concat(
        [data, pd.DataFrame(velocity_features, index=data.index)], axis=1, copy=False
    )


def engineer_features(
    df: pd.DataFrame,
    fit_split: pd.DataFrame | None = None,
    artifacts: FeatureArtifacts | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, FeatureArtifacts]:
    """Engineer features using train-fitted artifacts and only prior transactions.

    Pass the training DataFrame as ``fit_split`` on the first call. Reuse the
    returned ``artifacts`` for later temporal splits so validation sees training
    history and test sees training-plus-validation history, never future rows.
    New feature blocks are appended in batches to keep memory usage practical
    for the 434-column IEEE-CIS splits.
    """
    config = config or load_config()
    settings = config["feature_engineering"]
    is_training_input = artifacts is None and (fit_split is None or fit_split is df)
    if artifacts is None:
        reference = fit_split if fit_split is not None else df
        artifacts = _fit_artifacts(reference, config)
        if reference is not df:
            artifacts.velocity_history = _velocity_records(reference, artifacts)

    feature_data = df
    before_count = len(feature_data.columns) - int(artifacts.target_column in feature_data)
    feature_data = _apply_missingness_features(feature_data, artifacts)
    feature_data = _add_velocity_features(feature_data, artifacts)
    feature_data, dropped_columns = _apply_categorical_encodings(
        feature_data,
        artifacts,
        use_leave_one_out=is_training_input,
        smoothing=float(settings["target_encoding"]["smoothing"]),
    )
    after_count = len(feature_data.columns) - int(artifacts.target_column in feature_data)
    LOGGER.info(
        "Engineered %d features into %d features (%+d).",
        before_count,
        after_count,
        after_count - before_count,
    )
    LOGGER.info(
        "Replaced %d raw categorical columns: %s",
        len(dropped_columns),
        ", ".join(dropped_columns),
    )
    LOGGER.info(
        "Added %d high-missingness indicators; retained the source columns.",
        len(artifacts.high_missing_columns),
    )
    return feature_data, artifacts


def save_feature_artifacts(
    artifacts: FeatureArtifacts, config: dict[str, Any] | None = None
) -> Path:
    """Persist train-fitted encoders and imputers for later inference reuse."""
    config = config or load_config()
    output_path = _repository_path(config["feature_engineering"]["encoder_artifact_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.persistable_copy(), output_path)
    LOGGER.info("Saved fitted encoders and imputers to %s", output_path)
    return output_path


def _load_split(name: str, config: dict[str, Any]) -> pd.DataFrame:
    """Load one Phase 1 parquet split."""
    split_path = _repository_path(config["data"]["splits_path"]) / f"{name}.parquet"
    LOGGER.info("Loading %s split from %s", name, split_path)
    return pd.read_parquet(split_path)


def _save_feature_split(name: str, data: pd.DataFrame, config: dict[str, Any]) -> Path:
    """Save one feature-engineered parquet split."""
    output_dir = _repository_path(config["data"]["processed_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_features.parquet"
    data.to_parquet(output_path, index=False)
    LOGGER.info("Saved %s feature split to %s", name, output_path)
    return output_path


def run_feature_pipeline() -> dict[str, Path]:
    """Fit on train, transform each chronological split, and save all artifacts."""
    config = load_config()
    outputs: dict[str, Path] = {}
    train = _load_split("train", config)
    train_features, artifacts = engineer_features(train, fit_split=train, config=config)
    outputs["train"] = _save_feature_split("train", train_features, config)
    del train, train_features

    for split_name in ("val", "test"):
        split = _load_split(split_name, config)
        split_features, artifacts = engineer_features(
            split, artifacts=artifacts, config=config
        )
        outputs[split_name] = _save_feature_split(split_name, split_features, config)
        del split, split_features

    outputs["encoders"] = save_feature_artifacts(artifacts, config)
    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Re-import through the package name so joblib records FeatureArtifacts as
    # ``src.features.engineer.FeatureArtifacts`` rather than ``__main__``.
    from src.features.engineer import run_feature_pipeline as package_pipeline

    generated_outputs = package_pipeline()
    for output_name, output_path in generated_outputs.items():
        print(f"Saved {output_name}: {output_path}")

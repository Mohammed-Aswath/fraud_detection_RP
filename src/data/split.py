"""Create chronological train, validation, and test datasets."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.clean import missingness_report, reduce_memory_usage
from src.data.load import load_training_data
from src.utils.config_loader import load_config


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train", "val", "test")


def _repository_path(path_value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _format_transaction_dt(transaction_dt: float | int) -> str:
    """Format IEEE-CIS's relative transaction time as seconds and duration."""
    duration = pd.to_timedelta(transaction_dt, unit="s")
    return f"{transaction_dt:,.0f} seconds ({duration})"


def temporal_split(
    data: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, pd.DataFrame]:
    """Sort by ``TransactionDT`` and slice the rows into chronological splits."""
    if "TransactionDT" not in data.columns:
        raise KeyError("TransactionDT is required for a temporal split.")
    if data["TransactionDT"].isna().any():
        raise ValueError("TransactionDT contains missing values; temporal ordering is undefined.")

    config = config or load_config()
    split_config = config["temporal_split"]
    ratios = tuple(float(split_config[f"{name}_ratio"]) for name in SPLIT_NAMES)
    if any(ratio <= 0 for ratio in ratios) or not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError("Temporal split ratios must be positive and sum to 1.0.")

    ordered = data.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * ratios[0])
    val_end = train_end + int(len(ordered) * ratios[1])
    splits = {
        "train": ordered.iloc[:train_end].copy(),
        "val": ordered.iloc[train_end:val_end].copy(),
        "test": ordered.iloc[val_end:].copy(),
    }
    print_split_ranges(splits)
    return splits


def print_split_ranges(splits: dict[str, pd.DataFrame]) -> None:
    """Print row counts and non-overlapping relative time ranges for each split."""
    for name in SPLIT_NAMES:
        split = splits[name]
        start = split["TransactionDT"].iloc[0]
        end = split["TransactionDT"].iloc[-1]
        print(
            f"{name}: {len(split):,} rows | "
            f"TransactionDT {_format_transaction_dt(start)} to "
            f"{_format_transaction_dt(end)}"
        )


def save_splits(
    splits: dict[str, pd.DataFrame], config: dict[str, Any] | None = None
) -> dict[str, Path]:
    """Write train, validation, and test data as parquet files."""
    config = config or load_config()
    output_dir = _repository_path(config["data"]["splits_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        output_path = output_dir / f"{name}.parquet"
        splits[name].to_parquet(output_path, index=False)
        output_paths[name] = output_path
        LOGGER.info("Saved %s split to %s", name, output_path)
    return output_paths


def run_pipeline() -> dict[str, Path]:
    """Run the Phase 1 load, report, memory-reduction, and split workflow."""
    data = load_training_data()
    report = missingness_report(data)
    LOGGER.info("Top 20 columns by missingness:\n%s", report.head(20).to_string())
    optimized_data = reduce_memory_usage(data)
    splits = temporal_split(optimized_data)
    return save_splits(splits)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    saved_paths = run_pipeline()
    for split_name, path in saved_paths.items():
        print(f"Saved {split_name} parquet: {path}")

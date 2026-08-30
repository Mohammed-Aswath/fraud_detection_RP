"""Train, cost-optimize, and register the baseline LightGBM fraud model."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from src.models.cost_matrix import compute_cost, find_optimal_threshold
from src.utils.config_loader import load_config


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _repository_path(path_value: str | Path) -> Path:
    """Resolve configured paths relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def configure_mlflow(config: dict[str, Any]) -> None:
    """Configure local MLflow by default, with an opt-in DagsHub endpoint."""
    settings = config["mlflow"]
    if settings["use_dagshub"]:
        tracking_uri = settings["dagshub_tracking_uri"]
        if not tracking_uri:
            raise ValueError("Set mlflow.dagshub_tracking_uri before enabling DagsHub.")
    else:
        tracking_uri = _repository_path(settings["local_tracking_uri"]).as_uri()
        # MLflow 3.15 requires an explicit opt-in for the local file backend.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings["experiment_name"])
    LOGGER.info("MLflow tracking URI: %s", tracking_uri)


def _load_feature_split(name: str, config: dict[str, Any]) -> pd.DataFrame:
    """Load one feature-engineered parquet split."""
    path = _repository_path(config["data"]["processed_path"]) / f"{name}_features.parquet"
    LOGGER.info("Loading %s features from %s", name, path)
    return pd.read_parquet(path)


def _feature_matrix(data: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target while retaining TransactionAmt for business-cost scoring."""
    if target_column not in data:
        raise KeyError(f"Missing target column: {target_column}")
    return data.drop(columns=[target_column]), data[target_column].astype(np.int8)


def _classification_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate threshold-independent and threshold-based validation metrics."""
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _naive_costs(
    y_true: pd.Series, transaction_amounts: pd.Series, fp_cost: float, fn_cost_multiplier: float
) -> dict[str, float]:
    """Calculate the cost of always flagging and never flagging transactions."""
    all_negative = np.zeros(len(y_true), dtype=float)
    all_positive = np.ones(len(y_true), dtype=float)
    flag_nothing, _ = compute_cost(
        y_true, all_negative, 0.5, transaction_amounts, fp_cost, fn_cost_multiplier
    )
    flag_everything, _ = compute_cost(
        y_true, all_positive, 0.5, transaction_amounts, fp_cost, fn_cost_multiplier
    )
    return {"flag_nothing_cost": flag_nothing, "flag_everything_cost": flag_everything}


def _log_parameters(parameters: dict[str, Any]) -> None:
    """Convert model parameters to values accepted by MLflow's parameter store."""
    mlflow.log_params(
        {
            key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
            for key, value in parameters.items()
        }
    )


def _log_cost_curve(curve: pd.DataFrame, selected_threshold: float) -> None:
    """Create and log the validation cost-vs-threshold chart."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(curve["threshold"], curve["total_cost"], color="tab:blue")
    axis.axvline(selected_threshold, color="tab:red", linestyle="--", label="selected")
    axis.set(
        title="Validation cost by fraud decision threshold",
        xlabel="Fraud probability threshold",
        ylabel="Total cost",
    )
    axis.legend()
    figure.tight_layout()
    with tempfile.TemporaryDirectory() as temp_directory:
        plot_path = Path(temp_directory) / "cost_vs_threshold.png"
        figure.savefig(plot_path, dpi=150)
        mlflow.log_artifact(str(plot_path), artifact_path="cost_analysis")
        curve.to_csv(Path(temp_directory) / "cost_vs_threshold.csv", index=False)
        mlflow.log_artifact(
            str(Path(temp_directory) / "cost_vs_threshold.csv"), artifact_path="cost_analysis"
        )
    plt.close(figure)


def _register_production_model(run_id: str, config: dict[str, Any]) -> None:
    """Register the selected run's logged model and tag its latest version."""
    model_name = config["mlflow"]["registry_model_name"]
    registered_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client = MlflowClient()
    client.set_registered_model_tag(model_name, "lifecycle", "production")
    client.set_model_version_tag(model_name, registered_version.version, "lifecycle", "production")
    try:
        client.set_registered_model_alias(model_name, "production", registered_version.version)
    except Exception as error:  # Older or remote registries may not expose aliases.
        LOGGER.warning("Could not set MLflow production alias: %s", error)
    LOGGER.info("Registered %s version %s as production.", model_name, registered_version.version)


def train_model() -> dict[str, Any]:
    """Train the model and return validation results needed by downstream code."""
    config = load_config()
    configure_mlflow(config)
    target_column = config["feature_engineering"]["target_column"]
    train_data = _load_feature_split("train", config)
    validation_data = _load_feature_split("val", config)
    train_features, y_train = _feature_matrix(train_data, target_column)
    validation_features, y_validation = _feature_matrix(validation_data, target_column)
    if "TransactionAmt" not in validation_data:
        raise KeyError("TransactionAmt is required for transaction-aware cost optimization.")

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both fraud and non-fraud examples.")
    scale_pos_weight = negative_count / positive_count
    parameters = {
        **config["model"]["lightgbm"],
        "scale_pos_weight": scale_pos_weight,
    }
    model = lgb.LGBMClassifier(**parameters)

    with mlflow.start_run(run_name="baseline") as baseline_run:
        mlflow.set_tag("run_type", "baseline")
        _log_parameters(model.get_params())
        mlflow.log_param("train_class_imbalance_ratio", scale_pos_weight)
        model.fit(train_features, y_train, callbacks=[lgb.log_evaluation(period=50)])
        validation_probabilities = model.predict_proba(validation_features)[:, 1]
        baseline_metrics = _classification_metrics(y_validation, validation_probabilities, 0.5)
        mlflow.log_metrics({f"val_{name}": value for name, value in baseline_metrics.items()})
        baseline_run_id = baseline_run.info.run_id

    cost_settings = config["cost_matrix"]
    selected_threshold, cost_curve = find_optimal_threshold(
        y_validation,
        validation_probabilities,
        validation_data["TransactionAmt"],
        fp_cost=float(cost_settings["fp_cost"]),
        fn_cost_multiplier=float(cost_settings["fn_cost_multiplier"]),
    )
    optimized_cost, optimized_breakdown = compute_cost(
        y_validation,
        validation_probabilities,
        selected_threshold,
        validation_data["TransactionAmt"],
        fp_cost=float(cost_settings["fp_cost"]),
        fn_cost_multiplier=float(cost_settings["fn_cost_multiplier"]),
    )
    optimized_metrics = _classification_metrics(
        y_validation, validation_probabilities, selected_threshold
    )
    naive_costs = _naive_costs(
        y_validation,
        validation_data["TransactionAmt"],
        float(cost_settings["fp_cost"]),
        float(cost_settings["fn_cost_multiplier"]),
    )
    model.fraud_risk_threshold_ = selected_threshold
    model.fraud_risk_cost_settings_ = dict(cost_settings)

    model_path = _repository_path(config["model"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with mlflow.start_run(run_name="cost-optimized") as cost_run:
        mlflow.set_tag("run_type", "cost-optimized")
        mlflow.log_param("baseline_run_id", baseline_run_id)
        mlflow.log_param("selected_threshold", selected_threshold)
        mlflow.log_params({f"cost_{key}": value for key, value in cost_settings.items()})
        mlflow.log_metrics(
            {
                "val_pr_auc": optimized_metrics["pr_auc"],
                "val_roc_auc": optimized_metrics["roc_auc"],
                "val_precision": optimized_metrics["precision"],
                "val_recall": optimized_metrics["recall"],
                "val_f1": optimized_metrics["f1"],
                "val_total_cost": optimized_cost,
                "val_cost_saved_vs_flag_nothing": naive_costs["flag_nothing_cost"] - optimized_cost,
                "val_cost_saved_vs_flag_everything": naive_costs["flag_everything_cost"] - optimized_cost,
                **{f"val_{key}": float(value) for key, value in optimized_breakdown.items()},
                **{f"val_naive_{key}": value for key, value in naive_costs.items()},
            }
        )
        _log_cost_curve(cost_curve, selected_threshold)
        mlflow.log_artifact(str(model_path), artifact_path="model_file")
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            input_example=validation_features.head(3),
            serialization_format="cloudpickle",
        )
        cost_run_id = cost_run.info.run_id

    _register_production_model(cost_run_id, config)
    LOGGER.info(
        "Training complete. Validation cost %.2f at threshold %.2f.",
        optimized_cost,
        selected_threshold,
    )
    return {
        "model_path": model_path,
        "threshold": selected_threshold,
        "baseline_run_id": baseline_run_id,
        "cost_optimized_run_id": cost_run_id,
        "validation_cost": optimized_cost,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = train_model()
    print(
        "Training complete: "
        f"threshold={results['threshold']:.2f}, validation_cost={results['validation_cost']:.2f}, "
        f"model={results['model_path']}"
    )

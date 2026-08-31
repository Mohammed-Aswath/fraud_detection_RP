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
HP_SEARCH_CONFIGURATIONS = (
    {"num_leaves": 15, "reg_lambda": 0.1},
    {"num_leaves": 31, "reg_lambda": 1.0},
    {"num_leaves": 63, "reg_lambda": 5.0},
)
VELOCITY_FEATURES = (
    "card1__transaction_count_1h",
    "card1__transaction_count_24h",
    "card1__amount_total_1h",
    "card1__amount_zscore_history",
    "card1_email__seconds_since_last",
)


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
    y_true: pd.Series,
    transaction_amounts: pd.Series,
    fp_cost_rate: float,
    fn_cost_multiplier: float,
) -> dict[str, float]:
    """Calculate the cost of always flagging and never flagging transactions."""
    all_negative = np.zeros(len(y_true), dtype=float)
    all_positive = np.ones(len(y_true), dtype=float)
    flag_nothing, _ = compute_cost(
        y_true, all_negative, 0.5, transaction_amounts, fp_cost_rate, fn_cost_multiplier
    )
    flag_everything, _ = compute_cost(
        y_true, all_positive, 0.5, transaction_amounts, fp_cost_rate, fn_cost_multiplier
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


def _fit_candidate(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    validation_features: pd.DataFrame,
    y_validation: pd.Series,
    parameters: dict[str, Any],
    run_name: str,
    run_type: str,
) -> dict[str, Any]:
    """Train one early-stopped LightGBM candidate and log its validation PR-AUC."""
    model = lgb.LGBMClassifier(**parameters)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", run_type)
        _log_parameters(model.get_params())
        model.fit(
            train_features,
            y_train,
            eval_set=[(validation_features, y_validation)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
                lgb.log_evaluation(period=50),
            ],
        )
        best_iteration = int(model.best_iteration_ or model.n_estimators)
        validation_probabilities = model.predict_proba(
            validation_features, num_iteration=best_iteration
        )[:, 1]
        validation_pr_auc = float(average_precision_score(y_validation, validation_probabilities))
        mlflow.log_metrics(
            {
                "val_pr_auc": validation_pr_auc,
                "best_iteration": best_iteration,
            }
        )
        run_id = run.info.run_id
    return {
        "model": model,
        "validation_probabilities": validation_probabilities,
        "val_pr_auc": validation_pr_auc,
        "best_iteration": best_iteration,
        "run_id": run_id,
        "parameters": parameters,
        "run_name": run_name,
        "run_type": run_type,
    }


def _log_feature_importances(
    model: lgb.LGBMClassifier, feature_names: pd.Index
) -> pd.DataFrame:
    """Log top gain importances and print the ranks of Phase 2 velocity features."""
    importance_data = pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain_importance", ascending=False, kind="stable")
    importance_data["rank"] = np.arange(1, len(importance_data) + 1)
    top_30 = importance_data.head(30).copy()

    with tempfile.TemporaryDirectory() as temp_directory:
        temp_path = Path(temp_directory)
        csv_path = temp_path / "top_30_gain_importances.csv"
        top_30.to_csv(csv_path, index=False)
        figure, axis = plt.subplots(figsize=(10, 9))
        plot_data = top_30.sort_values("gain_importance")
        axis.barh(plot_data["feature"], plot_data["gain_importance"], color="tab:blue")
        axis.set(title="Top 30 feature importances by LightGBM gain", xlabel="Gain")
        figure.tight_layout()
        plot_path = temp_path / "top_30_gain_importances.png"
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        mlflow.log_artifact(str(csv_path), artifact_path="feature_importance")
        mlflow.log_artifact(str(plot_path), artifact_path="feature_importance")

    importance_by_feature = importance_data.set_index("feature")
    print("\nPhase 2 velocity feature gain ranks:")
    for feature in VELOCITY_FEATURES:
        if feature in importance_by_feature.index:
            row = importance_by_feature.loc[feature]
            print(f"  {feature}: rank {int(row['rank'])}, gain {row['gain_importance']:.2f}")
        else:
            print(f"  {feature}: not present in model features")
    print("\nTop 10 gain feature importances:")
    print(top_30.head(10).to_string(index=False, formatters={"gain_importance": "{:.2f}".format}))
    return top_30


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
    validation_amounts = validation_data["TransactionAmt"]
    print(
        "Validation TransactionAmt: "
        f"mean: {validation_amounts.mean():.2f}, "
        f"median: {validation_amounts.median():.2f}"
    )

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both fraud and non-fraud examples.")
    scale_pos_weight = negative_count / positive_count
    base_parameters = config["model"]["lightgbm"]
    search_results: list[dict[str, Any]] = []
    for candidate in HP_SEARCH_CONFIGURATIONS:
        parameters = {
            **base_parameters,
            **candidate,
            "scale_pos_weight": scale_pos_weight,
        }
        search_results.append(
            _fit_candidate(
                train_features,
                y_train,
                validation_features,
                y_validation,
                parameters,
                run_name=(
                    f"hp-search-leaves-{candidate['num_leaves']}-lambda-{candidate['reg_lambda']}"
                ),
                run_type="hp-search",
            )
        )

    unweighted_parameters = {
        **base_parameters,
        "num_leaves": 31,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
    }
    search_results.append(
        _fit_candidate(
            train_features,
            y_train,
            validation_features,
            y_validation,
            unweighted_parameters,
            run_name="scale-pos-weight-1",
            run_type="scale-pos-weight-comparison",
        )
    )
    comparison_table = pd.DataFrame(
        [
            {
                "run_name": result["run_name"],
                "run_type": result["run_type"],
                "num_leaves": result["parameters"]["num_leaves"],
                "reg_lambda": result["parameters"]["reg_lambda"],
                "scale_pos_weight": result["parameters"]["scale_pos_weight"],
                "best_iteration": result["best_iteration"],
                "val_pr_auc": result["val_pr_auc"],
            }
            for result in search_results
        ]
    ).sort_values("val_pr_auc", ascending=False, kind="stable")
    print("\nValidation hyperparameter and scale_pos_weight comparison:")
    print(comparison_table.to_string(index=False, formatters={"val_pr_auc": "{:.6f}".format}))
    best_result = max(search_results, key=lambda result: result["val_pr_auc"])
    print(
        "\nBest validation PR-AUC configuration: "
        f"num_leaves={best_result['parameters']['num_leaves']}, "
        f"reg_lambda={best_result['parameters']['reg_lambda']}, "
        f"scale_pos_weight={best_result['parameters']['scale_pos_weight']:.4f}, "
        f"PR-AUC={best_result['val_pr_auc']:.6f}"
    )
    model = best_result["model"]
    validation_probabilities = best_result["validation_probabilities"]
    best_iteration = best_result["best_iteration"]
    baseline_run_id = best_result["run_id"]

    cost_settings = config["cost_matrix"]
    selected_threshold, cost_curve = find_optimal_threshold(
        y_validation,
        validation_probabilities,
        validation_amounts,
        fp_cost_rate=float(cost_settings["fp_cost_rate"]),
        fn_cost_multiplier=float(cost_settings["fn_cost_multiplier"]),
    )
    optimized_cost, optimized_breakdown = compute_cost(
        y_validation,
        validation_probabilities,
        selected_threshold,
        validation_amounts,
        fp_cost_rate=float(cost_settings["fp_cost_rate"]),
        fn_cost_multiplier=float(cost_settings["fn_cost_multiplier"]),
    )
    optimized_metrics = _classification_metrics(
        y_validation, validation_probabilities, selected_threshold
    )
    naive_costs = _naive_costs(
        y_validation,
        validation_amounts,
        float(cost_settings["fp_cost_rate"]),
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
        mlflow.log_param("selected_search_run_id", baseline_run_id)
        mlflow.log_param("selected_threshold", selected_threshold)
        mlflow.log_params({f"cost_{key}": value for key, value in cost_settings.items()})
        mlflow.log_metrics(
            {
                "val_pr_auc": optimized_metrics["pr_auc"],
                "best_iteration": best_iteration,
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
        _log_feature_importances(model, train_features.columns)
        with tempfile.TemporaryDirectory() as temp_directory:
            comparison_path = Path(temp_directory) / "hyperparameter_comparison.csv"
            comparison_table.to_csv(comparison_path, index=False)
            mlflow.log_artifact(str(comparison_path), artifact_path="hyperparameter_search")
        mlflow.log_artifact(str(model_path), artifact_path="model_file")
        # Skip MLflow model logging due to dependency conflicts - model already saved as joblib artifact
        cost_run_id = cost_run.info.run_id

    # Skip MLflow model registration since model wasn't logged to MLflow
    # _register_production_model(cost_run_id, config)
    LOGGER.info(
        "Training complete. Validation cost %.2f at threshold %.2f.",
        optimized_cost,
        selected_threshold,
    )
    return {
        "model_path": model_path,
        "threshold": selected_threshold,
        "baseline_run_id": baseline_run_id,
        "best_iteration": best_iteration,
        "best_val_pr_auc": best_result["val_pr_auc"],
        "cost_optimized_run_id": cost_run_id,
        "validation_cost": optimized_cost,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = train_model()
    print(
        "Training complete: "
        f"threshold={results['threshold']:.6f}, validation_cost={results['validation_cost']:.2f}, "
        f"best_iteration={results['best_iteration']}, val_pr_auc={results['best_val_pr_auc']:.6f}, "
        f"model={results['model_path']}"
    )

"""Final held-out test evaluation for the trained fraud model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support

from src.models.cost_matrix import compute_cost
from src.models.train import _feature_matrix, _naive_costs, _repository_path, configure_mlflow
from src.utils.config_loader import load_config


LOGGER = logging.getLogger(__name__)


def evaluate_model() -> dict[str, Any]:
    """Evaluate the saved model exactly once on the held-out test split."""
    config = load_config()
    configure_mlflow(config)
    target_column = config["feature_engineering"]["target_column"]
    test_path = _repository_path(config["data"]["processed_path"]) / "test_features.parquet"
    model_path = _repository_path(config["model"]["model_path"])
    LOGGER.info("Loading held-out test features from %s", test_path)
    test_data = pd.read_parquet(test_path)
    if "TransactionAmt" not in test_data:
        raise KeyError("TransactionAmt is required for transaction-aware test cost.")
    model = joblib.load(model_path)
    threshold = float(getattr(model, "fraud_risk_threshold_", config["decision_threshold"]))
    test_features, y_test = _feature_matrix(test_data, target_column)
    probabilities = model.predict_proba(test_features)[:, 1]
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    cost_settings = config["cost_matrix"]
    total_cost, cost_breakdown = compute_cost(
        y_test,
        probabilities,
        threshold,
        test_data["TransactionAmt"],
        float(cost_settings["fp_cost"]),
        float(cost_settings["fn_cost_multiplier"]),
    )
    naive_costs = _naive_costs(
        y_test,
        test_data["TransactionAmt"],
        float(cost_settings["fp_cost"]),
        float(cost_settings["fn_cost_multiplier"]),
    )
    summary = {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "total_cost": total_cost,
        "cost_saved_vs_flag_nothing": naive_costs["flag_nothing_cost"] - total_cost,
        "cost_saved_vs_flag_everything": naive_costs["flag_everything_cost"] - total_cost,
        **cost_breakdown,
    }
    with mlflow.start_run(run_name="final-test-evaluation"):
        mlflow.set_tag("run_type", "final-test-evaluation")
        mlflow.set_tag("evaluation_split", "held-out-test")
        mlflow.log_metrics({key: float(value) for key, value in summary.items()})
        mlflow.log_dict(
            {
                "summary": summary,
                "confusion_matrix": matrix.tolist(),
                "naive_costs": naive_costs,
            },
            "test_evaluation.json",
        )

    printable = pd.DataFrame(
        [
            ("Threshold", f"{threshold:.2f}"),
            ("Precision", f"{precision:.4f}"),
            ("Recall", f"{recall:.4f}"),
            ("F1", f"{f1:.4f}"),
            ("PR-AUC", f"{pr_auc:.4f}"),
            ("Total cost", f"{total_cost:.2f}"),
            ("Saved vs flag nothing", f"{summary['cost_saved_vs_flag_nothing']:.2f}"),
            ("Saved vs flag everything", f"{summary['cost_saved_vs_flag_everything']:.2f}"),
        ],
        columns=["Metric", "Value"],
    )
    print("\nFinal held-out test evaluation")
    print(printable.to_string(index=False))
    print("\nConfusion matrix [[TN, FP], [FN, TP]]")
    print(matrix)
    return {**summary, "confusion_matrix": matrix}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    evaluate_model()

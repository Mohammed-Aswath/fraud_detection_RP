"""LightGBM SHAP explainability for the production fraud model."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import matplotlib

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPOSITORY_ROOT / "models" / "model.joblib"
TEST_PATH = REPOSITORY_ROOT / "data" / "processed" / "test_features.parquet"
ARTIFACT_DIR = REPOSITORY_ROOT / "artifacts" / "shap"
MODEL_NAME = "fraud-risk-manager-model"
MODEL_VERSION = "7"


def _load_model() -> object:
    """Load the saved production LightGBM model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Expected a LightGBM classifier-like model at {MODEL_PATH}")
    return model


def _resolve_threshold(model: object) -> float:
    """Read the threshold from the model or, if needed, from the MLflow run params."""
    threshold = getattr(model, "fraud_risk_threshold_", None)
    if threshold is not None:
        return float(threshold)

    tracking_uri = (REPOSITORY_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_version = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
    run = client.get_run(run_id=model_version.run_id)
    value = run.data.params.get("selected_threshold")
    if value is None:
        raise ValueError("Could not resolve the model decision threshold from the model or MLflow.")
    return float(value)


def _production_run_id() -> str:
    """Return the MLflow run ID attached to the production model version."""
    mlflow.set_tracking_uri((REPOSITORY_ROOT / "mlruns").as_uri())
    client = MlflowClient()
    model_version = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
    return model_version.run_id


def _sample_test_features(model: object) -> pd.DataFrame:
    """Load the test split and sample a fixed-size subset in the model’s training order."""
    frame = pd.read_parquet(TEST_PATH)
    sampled = frame.sample(n=min(2000, len(frame)), random_state=42, replace=False).reset_index(drop=True)
    feature_names = [
        column for column in getattr(model, "feature_name_", sampled.columns) if column in sampled.columns
    ]
    return sampled[feature_names].copy()


def _compute_explanation(model: object, features: pd.DataFrame) -> shap.Explanation:
    """Compute TreeSHAP values on the selected sample."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(features)
    if isinstance(explanation, list):
        explanation = explanation[0]
    return explanation


def _save_figure(figure_path: Path) -> None:
    """Ensure the parent directory exists and save the current matplotlib figure as PNG."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()


def _log_plot_to_mlflow(file_path: Path, artifact_path: str = "shap") -> None:
    """Log a plot into the production MLflow run."""
    mlflow.set_tracking_uri((REPOSITORY_ROOT / "mlruns").as_uri())
    with mlflow.start_run(run_id=_production_run_id()):
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)


def _global_summary_plot(explanation: shap.Explanation, output_path: Path) -> None:
    """Create and save the global SHAP beeswarm summary."""
    shap.plots.beeswarm(explanation, max_display=20)
    _save_figure(output_path)
    _log_plot_to_mlflow(output_path, artifact_path="shap")


def _select_example_transactions(
    sample_df: pd.DataFrame, scores: np.ndarray, threshold: float
) -> dict[str, dict[str, float | int | bool]]:
    """Choose one TP, one FP, and one TN-near-threshold example."""
    sample_df = sample_df.copy()
    sample_df["score"] = scores
    sample_df["label"] = sample_df["isFraud"].astype(int)

    tp_candidates = sample_df[(sample_df["label"] == 1) & (sample_df["score"] >= threshold)]
    fp_candidates = sample_df[(sample_df["label"] == 0) & (sample_df["score"] >= threshold)]
    tn_candidates = sample_df[(sample_df["label"] == 0) & (sample_df["score"] < threshold)]

    selected: dict[str, dict[str, float | int | bool]] = {}

    if not tp_candidates.empty:
        tp_row = tp_candidates.sort_values("score", ascending=False).iloc[0]
        selected["tp"] = {
            "transaction_id": int(tp_row["TransactionID"]),
            "score": float(tp_row["score"]),
            "label": int(tp_row["label"]),
            "predicted_flag": bool(tp_row["score"] >= threshold),
            "is_correct": True,
        }
    else:
        raise ValueError("No true positive example found above the threshold in the sample.")

    if not fp_candidates.empty:
        fp_row = fp_candidates.sort_values("score", ascending=False).iloc[0]
        selected["fp"] = {
            "transaction_id": int(fp_row["TransactionID"]),
            "score": float(fp_row["score"]),
            "label": int(fp_row["label"]),
            "predicted_flag": bool(fp_row["score"] >= threshold),
            "is_correct": False,
        }
    else:
        raise ValueError("No false positive example found in the sample.")

    if not tn_candidates.empty:
        tn_row = tn_candidates.sort_values("score", ascending=False).iloc[0]
        selected["tn"] = {
            "transaction_id": int(tn_row["TransactionID"]),
            "score": float(tn_row["score"]),
            "label": int(tn_row["label"]),
            "predicted_flag": bool(tn_row["score"] >= threshold),
            "is_correct": True,
        }
    else:
        raise ValueError("No true negative example found below the threshold in the sample.")

    return selected


def _waterfall_plot(explanation: shap.Explanation, row_index: int, output_path: Path) -> None:
    """Capture a single-row SHAP waterfall plot for the given row index."""
    instance = explanation[row_index]
    shap.plots.waterfall(instance, max_display=10)
    _save_figure(output_path)
    _log_plot_to_mlflow(output_path, artifact_path="shap")


def run_explainability() -> dict[str, dict[str, float | int | bool]]:
    """Run the complete SHAP explainability pipeline."""
    model = _load_model()
    threshold = _resolve_threshold(model)
    full_df = pd.read_parquet(TEST_PATH)
    sample_df = full_df.sample(n=min(2000, len(full_df)), random_state=42, replace=False).reset_index(drop=True)
    feature_names = [
        column for column in getattr(model, "feature_name_", sample_df.columns) if column in sample_df.columns
    ]
    feature_matrix = sample_df[feature_names].copy()
    probabilities = model.predict_proba(feature_matrix)[:, 1]
    sample_df = sample_df.copy()
    sample_df["score"] = probabilities

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    explanation = _compute_explanation(model, feature_matrix)
    summary_path = ARTIFACT_DIR / "global_shap_summary.png"
    _global_summary_plot(explanation, summary_path)

    selected = _select_example_transactions(sample_df, probabilities, threshold)

    for example_name, example in selected.items():
        row_index = sample_df.index[sample_df["TransactionID"] == example["transaction_id"]][0]
        output_path = ARTIFACT_DIR / f"{example_name}_waterfall.png"
        _waterfall_plot(explanation, row_index, output_path)

    LOGGER.info(
        "SHAP summary saved to %s and 3 example waterfalls saved to %s",
        summary_path,
        ARTIFACT_DIR,
    )
    return selected


def main() -> None:
    """Entry point for the explainability workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    model = _load_model()
    threshold = _resolve_threshold(model)
    full_df = pd.read_parquet(TEST_PATH)
    sample_df = full_df.sample(n=min(2000, len(full_df)), random_state=42, replace=False).reset_index(drop=True)
    feature_names = [
        column for column in getattr(model, "feature_name_", sample_df.columns) if column in sample_df.columns
    ]
    probability_scores = model.predict_proba(sample_df[feature_names])[:, 1]
    selected = _select_example_transactions(sample_df, probability_scores, threshold)

    print(f"Threshold: {threshold:.6f}")
    for label_key, record in selected.items():
        print(
            f"{label_key.upper()}: TransactionID={record['transaction_id']} "
            f"score={record['score']:.6f} true_label={record['label']} "
            f"predicted_flag={record['predicted_flag']} correct={record['is_correct']}"
        )

    run_explainability()


if __name__ == "__main__":
    main()

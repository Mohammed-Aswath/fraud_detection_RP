"""Transaction-aware decision costs and threshold selection."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _validated_arrays(
    y_true: Iterable[int],
    y_pred_proba: Iterable[float],
    transaction_amounts: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cost inputs to aligned one-dimensional numeric arrays."""
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(y_pred_proba, dtype=float).reshape(-1)
    amounts = np.asarray(transaction_amounts, dtype=float).reshape(-1)
    if not (len(labels) == len(probabilities) == len(amounts)):
        raise ValueError("y_true, y_pred_proba, and transaction_amounts must align.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary 0/1 labels.")
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Transaction amounts must be finite, non-negative values.")
    return labels, probabilities, amounts


def compute_cost(
    y_true: Iterable[int],
    y_pred_proba: Iterable[float],
    threshold: float,
    transaction_amounts: Iterable[float],
    fp_cost_rate: float,
    fn_cost_multiplier: float,
) -> tuple[float, dict[str, float | int]]:
    """Return cost from a fraud threshold and a transparent error breakdown.

    ``fp_cost_rate`` is the fraction of a legitimate transaction's amount lost
    through false-positive friction or a declined sale.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if not 0 <= fp_cost_rate <= 1:
        raise ValueError("fp_cost_rate must be between 0 and 1.")
    if fn_cost_multiplier < 0:
        raise ValueError("fn_cost_multiplier must be non-negative.")
    labels, probabilities, amounts = _validated_arrays(
        y_true, y_pred_proba, transaction_amounts
    )
    predictions = probabilities >= threshold
    false_positives = (labels == 0) & predictions
    false_negatives = (labels == 1) & ~predictions
    total_fp_cost = float(amounts[false_positives].sum() * fp_cost_rate)
    total_fn_cost = float(amounts[false_negatives].sum() * fn_cost_multiplier)
    total_cost = total_fp_cost + total_fn_cost
    return total_cost, {
        "total_cost": total_cost,
        "total_fp_cost": total_fp_cost,
        "total_fn_cost": total_fn_cost,
        "false_positive_count": int(false_positives.sum()),
        "false_negative_count": int(false_negatives.sum()),
    }


def find_optimal_threshold(
    y_true: Iterable[int],
    y_pred_proba: Iterable[float],
    transaction_amounts: Iterable[float],
    fp_cost_rate: float,
    fn_cost_multiplier: float,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Sweep thresholds and return the threshold with the lowest business cost."""
    labels, probabilities, amounts = _validated_arrays(
        y_true, y_pred_proba, transaction_amounts
    )
    candidate_thresholds = (
        np.linspace(probabilities.min(), probabilities.max(), 200)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=float)
    )
    if candidate_thresholds.size == 0:
        raise ValueError("At least one candidate threshold is required.")

    rows: list[dict[str, float | int]] = []
    for threshold in candidate_thresholds:
        total_cost, breakdown = compute_cost(
            labels,
            probabilities,
            float(threshold),
            amounts,
            fp_cost_rate,
            fn_cost_multiplier,
        )
        rows.append({"threshold": float(threshold), "total_cost": total_cost, **breakdown})

    curve = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    optimal_index = curve["total_cost"].idxmin()
    return float(curve.loc[optimal_index, "threshold"]), curve

"""Tests for transaction-aware threshold cost selection."""

import numpy as np

from src.models.cost_matrix import compute_cost, find_optimal_threshold


def test_compute_cost_breaks_down_false_positive_and_negative_costs() -> None:
    total_cost, breakdown = compute_cost(
        y_true=[1, 0, 1],
        y_pred_proba=[0.9, 0.8, 0.1],
        threshold=0.5,
        transaction_amounts=[100.0, 10.0, 50.0],
        fp_cost_rate=0.3,
        fn_cost_multiplier=1.0,
    )

    assert total_cost == 53.0
    assert breakdown == {
        "total_cost": 53.0,
        "total_fp_cost": 3.0,
        "total_fn_cost": 50.0,
        "false_positive_count": 1,
        "false_negative_count": 1,
    }


def test_find_optimal_threshold_selects_known_lower_cost_choice() -> None:
    threshold, curve = find_optimal_threshold(
        y_true=[1, 0, 1, 0],
        y_pred_proba=[0.9, 0.7, 0.4, 0.1],
        transaction_amounts=[100.0, 1.0, 100.0, 1.0],
        fp_cost_rate=0.3,
        fn_cost_multiplier=1.0,
        thresholds=[0.5, 0.8],
    )

    assert threshold == 0.8
    assert np.isclose(curve.loc[curve["threshold"] == 0.5, "total_cost"].item(), 100.3)
    assert np.isclose(curve.loc[curve["threshold"] == 0.8, "total_cost"].item(), 100.0)

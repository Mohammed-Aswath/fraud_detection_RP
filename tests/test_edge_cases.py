"""Fast unit tests for Phase 1 data-foundation utilities."""

import pandas as pd

from src.data.clean import missingness_report, reduce_memory_usage
from src.data.split import temporal_split


def test_missingness_report_keeps_all_columns() -> None:
    data = pd.DataFrame({"complete": [1, 2], "partial": [1, None]})

    report = missingness_report(data)

    assert list(report.index) == ["partial", "complete"]
    assert report.loc["partial", "missing_pct"] == 50.0
    assert report.loc["complete", "missing_pct"] == 0.0


def test_reduce_memory_usage_downcasts_safe_numeric_columns() -> None:
    data = pd.DataFrame(
        {
            "small_integer": pd.Series([1, 2], dtype="int64"),
            "decimal": pd.Series([1.5, 2.5], dtype="float64"),
        }
    )

    optimized = reduce_memory_usage(data)

    assert optimized["small_integer"].dtype == "int32"
    assert optimized["decimal"].dtype == "float32"


def test_temporal_split_orders_earliest_to_latest() -> None:
    data = pd.DataFrame({"TransactionDT": [30, 10, 40, 20], "value": list("cadb")})
    config = {
        "temporal_split": {
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
        }
    }

    splits = temporal_split(data, config)

    assert splits["train"]["TransactionDT"].tolist() == [10, 20]
    assert splits["val"]["TransactionDT"].tolist() == [30]
    assert splits["test"]["TransactionDT"].tolist() == [40]

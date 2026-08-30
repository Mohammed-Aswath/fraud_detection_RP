"""Non-destructive data-quality reporting and memory optimization."""

import logging

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def missingness_report(data: pd.DataFrame) -> pd.DataFrame:
    """Return null counts and percentages for every column, highest first."""
    report = pd.DataFrame(
        {
            "missing_count": data.isna().sum(),
            "missing_pct": data.isna().mean().mul(100),
        }
    ).sort_values("missing_pct", ascending=False)
    report.index.name = "column"
    return report


def reduce_memory_usage(data: pd.DataFrame) -> pd.DataFrame:
    """Safely downcast int64 and float64 columns without imputing values.

    The DataFrame is optimized in place and returned for convenient chaining.
    """
    optimized = data
    before_mb = optimized.memory_usage(deep=True).sum() / 1024**2

    for column in optimized.columns:
        series = optimized[column]
        if series.dtype == np.dtype("int64"):
            limits = np.iinfo(np.int32)
            if series.min() >= limits.min and series.max() <= limits.max:
                optimized[column] = series.astype(np.int32)
        elif series.dtype == np.dtype("float64"):
            finite_values = series.dropna()
            if finite_values.empty or finite_values.abs().max() <= np.finfo(np.float32).max:
                optimized[column] = series.astype(np.float32)

    after_mb = optimized.memory_usage(deep=True).sum() / 1024**2
    LOGGER.info(
        "Memory usage reduced from %.2f MiB to %.2f MiB (%.1f%% reduction)",
        before_mb,
        after_mb,
        (1 - after_mb / before_mb) * 100 if before_mb else 0,
    )
    return optimized

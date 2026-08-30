"""Load and combine the IEEE-CIS fraud training datasets."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import load_config


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _repository_path(path_value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_training_data(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load transaction and identity data, then left-join on ``TransactionID``.

    Identity rows are optional: transactions without an identity record are
    retained because that missingness can carry fraud-risk signal.
    """
    config = config or load_config()
    data_config = config["data"]
    transaction_path = _repository_path(data_config["train_transaction_path"])
    identity_path = _repository_path(data_config["train_identity_path"])

    LOGGER.info("Loading transaction data from %s", transaction_path)
    transactions = pd.read_csv(transaction_path)
    LOGGER.info("Loading identity data from %s", identity_path)
    identities = pd.read_csv(identity_path)

    merged = transactions.merge(identities, on="TransactionID", how="left")
    memory_mb = merged.memory_usage(deep=True).sum() / 1024**2
    LOGGER.info(
        "Merged data loaded: %d rows, %d columns, %.2f MiB",
        len(merged),
        len(merged.columns),
        memory_mb,
    )
    return merged

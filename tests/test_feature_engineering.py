"""Tests for leakage-aware Phase 2 feature engineering."""

import numpy as np
import pandas as pd

from src.features.engineer import engineer_features


def _feature_config() -> dict:
    return {
        "feature_engineering": {
            "high_missingness_threshold": 0.9,
            "target_column": "isFraud",
            "missing_category": "__MISSING__",
            "target_encoding": {
                "columns": ["card4", "P_emaildomain"],
                "smoothing": 1.0,
            },
            "velocity": {
                "card_column": "card1",
                "email_column": "P_emaildomain",
                "windows_seconds": [3600, 86400],
            },
            "encoder_artifact_path": "artifacts/encoders.joblib",
        }
    }


def test_target_encoding_uses_train_mapping_for_new_categories() -> None:
    train = pd.DataFrame(
        {
            "TransactionDT": [0, 3600, 7200],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1": [1, 1, 2],
            "card4": ["visa", "visa", "mastercard"],
            "P_emaildomain": ["a.com", "a.com", "b.com"],
            "isFraud": [0, 1, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "TransactionDT": [7200, 10800],
            "TransactionAmt": [40.0, 50.0],
            "card1": [1, 3],
            "card4": ["visa", "new-card"],
            "P_emaildomain": ["a.com", "new.com"],
            "isFraud": [0, 0],
        }
    )

    train_features, artifacts = engineer_features(train, fit_split=train, config=_feature_config())
    validation_features, _ = engineer_features(
        validation, artifacts=artifacts, config=_feature_config()
    )

    assert "card4" not in train_features
    assert "card4__target_rate" in train_features
    assert validation_features.loc[1, "card4__target_rate"] == np.float32(2 / 3)
    assert validation_features.loc[0, "card1__transaction_count_1h"] == 1


def test_velocity_features_exclude_same_and_future_timestamps() -> None:
    train = pd.DataFrame(
        {
            "TransactionDT": [0, 3600, 7200],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1": [1, 1, 1],
            "P_emaildomain": ["a.com", "a.com", "a.com"],
            "isFraud": [0, 0, 1],
        }
    )

    features, _ = engineer_features(train, fit_split=train, config=_feature_config())

    assert features["card1__transaction_count_1h"].tolist() == [0, 1, 1]
    assert features["card1__amount_total_1h"].tolist() == [0.0, 10.0, 20.0]
    assert np.isnan(features.loc[0, "card1_email__seconds_since_last"])
    assert features.loc[1, "card1_email__seconds_since_last"] == 3600.0

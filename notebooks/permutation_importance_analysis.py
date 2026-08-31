"""
Manual permutation importance analysis on validation set for velocity features.
Keeps all 451 features intact, only shuffles target features.
"""

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import average_precision_score
from src.utils.config_loader import load_config

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[0]

# Target features to analyze
TARGET_FEATURES = [
    "card1__transaction_count_1h",
    "card1__transaction_count_24h",
    "card1__amount_total_1h",
    "card1__amount_zscore_history",
    "card1_email__seconds_since_last",
    "card6__target_rate",
    "card4__target_rate",
]


def _repository_path(path_value: str | Path) -> Path:
    """Resolve configured paths relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def run_manual_permutation_importance():
    """Run manual permutation importance analysis on validation set."""
    config = load_config()
    
    # Load validation features and target
    val_features_path = _repository_path(config["data"]["processed_path"]) / "val_features.parquet"
    print(f"Loading validation features from {val_features_path}")
    val_data = pd.read_parquet(val_features_path)
    
    target_column = config["feature_engineering"]["target_column"]
    val_features = val_data.drop(columns=[target_column])
    y_val = val_data[target_column].astype(np.int8)
    
    print(f"Validation set shape: {val_features.shape}")
    print(f"Validation target distribution: {y_val.value_counts().to_dict()}")
    
    # Load trained model
    model_path = _repository_path(config["model"]["model_path"])
    print(f"\nLoading model from {model_path}")
    model = joblib.load(model_path)
    
    # Filter to only target features that exist in the model
    available_features = [f for f in TARGET_FEATURES if f in val_features.columns]
    missing_features = [f for f in TARGET_FEATURES if f not in val_features.columns]
    
    if missing_features:
        print(f"WARNING: Missing features in validation data: {missing_features}")
    
    print(f"Analyzing permutation importance for {len(available_features)} features:")
    for feat in available_features:
        feat_type = "VELOCITY" if "card1" in feat else "TARGET-ENCODED"
        print(f"  - {feat} ({feat_type})")
    
    # Calculate baseline score
    baseline_proba = model.predict_proba(val_features)[:, 1]
    baseline_pr_auc = average_precision_score(y_val, baseline_proba)
    print(f"\nBaseline validation PR-AUC: {baseline_pr_auc:.6f}")
    
    # Run manual permutation importance
    print(f"\nRunning manual permutation importance (n_repeats=10)...")
    print("This may take a minute...\n")
    
    results = []
    for feature_name in available_features:
        importances = []
        
        for repeat in range(10):
            # Create a copy and shuffle the feature
            X_permuted = val_features.copy()
            X_permuted[feature_name] = np.random.permutation(X_permuted[feature_name].values)
            
            # Score with shuffled feature
            permuted_proba = model.predict_proba(X_permuted)[:, 1]
            permuted_pr_auc = average_precision_score(y_val, permuted_proba)
            
            # Calculate importance as drop in PR-AUC
            importance = baseline_pr_auc - permuted_pr_auc
            importances.append(importance)
        
        importance_mean = np.mean(importances)
        importance_std = np.std(importances)
        
        feat_type = "VELOCITY" if "card1" in feature_name else "TARGET-ENCODED"
        pct_drop = 100.0 * importance_mean / baseline_pr_auc if baseline_pr_auc > 0 else 0.0
        
        results.append({
            "feature": feature_name,
            "feature_type": feat_type,
            "importance_mean": importance_mean,
            "importance_std": importance_std,
            "importances": importances,
            "pct_drop": pct_drop,
        })
    
    # Create results dataframe
    importance_df = pd.DataFrame(results).sort_values("importance_mean", ascending=False, kind="stable")
    
    print("\n" + "="*110)
    print("PERMUTATION IMPORTANCE ANALYSIS RESULTS")
    print("="*110)
    print(f"Baseline validation PR-AUC: {baseline_pr_auc:.6f}\n")
    
    # Display results sorted by importance
    display_df = importance_df[[
        "feature", "feature_type", "importance_mean", "importance_std", "pct_drop"
    ]].copy()
    display_df.columns = ["Feature", "Type", "Mean Drop (PR-AUC)", "Std Dev", "% Drop"]
    
    print(display_df.to_string(index=False, 
          formatters={
              "Mean Drop (PR-AUC)": "{:.6f}".format,
              "Std Dev": "{:.6f}".format,
              "% Drop": "{:.2f}%".format,
          }))
    
    print("\n" + "="*110)
    print("SUMMARY BY FEATURE TYPE")
    print("="*110)
    
    velocity_feats = importance_df[importance_df["feature_type"] == "VELOCITY"]
    target_encoded_feats = importance_df[importance_df["feature_type"] == "TARGET-ENCODED"]
    
    print("\nVELOCITY FEATURES (5 total):")
    print("-" * 110)
    print(velocity_feats[[
        "feature", "importance_mean", "importance_std", "pct_drop"
    ]].to_string(index=False,
          formatters={
              "importance_mean": "{:.6f}".format,
              "importance_std": "{:.6f}".format,
              "pct_drop": "{:.2f}%".format,
          }))
    print(f"  Mean importance across velocity features: {velocity_feats['importance_mean'].mean():.6f}")
    print(f"  Max importance (velocity): {velocity_feats['importance_mean'].max():.6f}")
    print(f"  Min importance (velocity): {velocity_feats['importance_mean'].min():.6f}")
    print(f"  Total importance (sum): {velocity_feats['importance_mean'].sum():.6f}")
    
    print("\nTARGET-ENCODED FEATURES (2 total - for comparison):")
    print("-" * 110)
    print(target_encoded_feats[[
        "feature", "importance_mean", "importance_std", "pct_drop"
    ]].to_string(index=False,
          formatters={
              "importance_mean": "{:.6f}".format,
              "importance_std": "{:.6f}".format,
              "pct_drop": "{:.2f}%".format,
          }))
    print(f"  Mean importance across target-encoded features: {target_encoded_feats['importance_mean'].mean():.6f}")
    print(f"  Total importance (sum): {target_encoded_feats['importance_mean'].sum():.6f}")
    
    print("\n" + "="*110)
    print("INTERPRETATION & DECISION FRAMEWORK")
    print("="*110)
    
    velocity_mean = velocity_feats['importance_mean'].mean()
    target_encoded_mean = target_encoded_feats['importance_mean'].mean()
    velocity_total = velocity_feats['importance_mean'].sum()
    target_encoded_total = target_encoded_feats['importance_mean'].sum()
    
    print(f"""
PERMUTATION IMPORTANCE measures the drop in model performance when a feature's values 
are randomly shuffled. A feature with high importance means the model relies on it;
low/zero importance means the feature contributes little even if it exists.

BASELINE VALIDATION PR-AUC: {baseline_pr_auc:.6f}

KEY FINDINGS:
1. Velocity Features Mean Drop: {velocity_mean:.6f} PR-AUC 
   ({velocity_feats['importance_mean'].mean()/baseline_pr_auc*100:.3f}% of baseline)
   Total Drop: {velocity_total:.6f} PR-AUC
   
2. Target-Encoded Features Mean Drop: {target_encoded_mean:.6f} PR-AUC
   ({target_encoded_feats['importance_mean'].mean()/baseline_pr_auc*100:.3f}% of baseline)
   Total Drop: {target_encoded_total:.6f} PR-AUC

3. Ratio (Velocity/Target-Encoded):
   - By Mean: {velocity_mean / target_encoded_mean:.3f}x
   - By Total: {velocity_total / target_encoded_total:.3f}x

DECISION LOGIC:
If velocity features show CLOSE-TO-ZERO permutation importance (<0.0001 PR-AUC drop):
  ✗ Velocity features are NOT contributing to model predictions
  → RECOMMEND: DROP velocity features as redundant
  → Justification: Gain-based importance + permutation importance both show zero contribution
  
If velocity features show MEANINGFUL permutation importance (>0.0005 PR-AUC drop):
  ✓ Velocity features ARE providing marginal value through permutation 
  → Recommend: KEEP velocity features for marginal predictive value
  → Justification: Permutation importance captures downstream effects that gain misses

VERDICT:
""")
    
    if velocity_mean < 0.0001:
        print("""✗ DROP VELOCITY FEATURES
  Both gain-based importance (0.00) and permutation importance (<0.0001) confirm 
  velocity features are redundant. The target-encoded card features completely capture 
  fraud signal, making velocity computations unnecessary.""")
    elif velocity_mean < 0.0005:
        print("""⚠ MARGINAL VALUE - DECISION DEPENDS ON ENGINEERING COST
  Velocity features have minimal but non-zero permutation importance. 
  Consider: Compute cost vs. marginal performance gain. If feature engineering
  is cheap, keep them; if expensive, drop them.""")
    else:
        print("""✓ KEEP VELOCITY FEATURES
  Permutation importance shows meaningful contribution not captured by gain.
  Velocity features provide marginal value through interactions or downstream
  effects that deserve to stay in the model.""")

    print("\n" + "="*110)
    
    # Save detailed results to CSV
    results_path = _repository_path("artifacts") / "permutation_importance_validation.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df[[
        "feature", "feature_type", "importance_mean", "importance_std", "pct_drop"
    ]].to_csv(results_path, index=False)
    print(f"Detailed results saved to {results_path}\n")
    
    return importance_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    run_manual_permutation_importance()

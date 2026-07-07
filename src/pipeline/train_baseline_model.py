from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.pipeline.define_model_features import get_model_feature_columns, get_target_column

PROJECT_ROOT = Path(__file__).parents[2]
TRAIN_PATH   = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH     = PROJECT_ROOT / "data" / "processed" / "validation.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_PATH   = MODELS_DIR / "baseline_xgb.json"

# Columns in the whitelist that are identifiers, not model inputs.
IDENTIFIER_COLS = {"GAME_ID", "GAME_DATE"}


def train_baseline_model() -> xgb.XGBClassifier:
    """Train a baseline XGBoost classifier for NBA spread-cover prediction.

    Uses the whitelisted feature columns from define_model_features.py as
    inputs and home_covers_spread as the binary target.  Null values are
    left in place — XGBoost handles missing values natively by learning the
    optimal default direction for each split, so no rows or columns need to
    be dropped.  This means all 3,538 train rows and all 1,230 val rows are
    used, including early-season games lacking rolling history and 2024/2025
    games lacking injury coverage.

    Metrics
    -------
    Accuracy   : fraction of correct binary predictions.  The simplest
                 measure, but can be misleading on imbalanced classes.
    ROC-AUC    : area under the receiver-operating-characteristic curve.
                 Measures rank-ordering quality independent of threshold;
                 0.5 = random, 1.0 = perfect.  The primary metric here
                 since we care more about ranking games than raw accuracy.
    Log loss   : average negative log-likelihood of the predicted
                 probabilities.  Penalises confident wrong predictions
                 heavily; lower is better.  Useful for calibration checks.

    Returns
    -------
    xgb.XGBClassifier
        The fitted model.
    """
    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #
    print(f"Loading training data  : {TRAIN_PATH.name}")
    train = pd.read_csv(TRAIN_PATH)
    print(f"Loading validation data: {VAL_PATH.name}")
    val   = pd.read_csv(VAL_PATH)

    # ------------------------------------------------------------------ #
    # Build feature matrices                                               #
    # ------------------------------------------------------------------ #
    all_feature_cols = get_model_feature_columns()
    target_col       = get_target_column()

    # Remove identifiers only — nulls are left in place for XGBoost to handle.
    model_cols = [c for c in all_feature_cols if c not in IDENTIFIER_COLS]
    print(f"\n  Model features: {len(model_cols)} columns  (29 = 31 whitelisted − 2 identifiers)")

    X_train = train[model_cols]
    y_train = train[target_col]
    X_val   = val[model_cols]
    y_val   = val[target_col]

    print(f"  Train rows: {len(X_train):,}  |  Val rows: {len(X_val):,}")

    # ------------------------------------------------------------------ #
    # Majority-class baseline (determined from full training set)          #
    # ------------------------------------------------------------------ #
    majority_class   = int(y_train.mode()[0])
    majority_acc_val = (y_val == majority_class).mean()
    print(f"\n  Majority-class baseline accuracy on validation: {100 * majority_acc_val:.1f}%"
          f"  (always predict class {majority_class})")

    # ------------------------------------------------------------------ #
    # Train XGBoost                                                        #
    # ------------------------------------------------------------------ #
    print("\n  Training XGBClassifier ...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Evaluate on validation                                               #
    # ------------------------------------------------------------------ #
    y_pred      = model.predict(X_val)
    y_prob      = model.predict_proba(X_val)[:, 1]

    acc         = accuracy_score(y_val, y_pred)
    auc         = roc_auc_score(y_val, y_prob)
    ll          = log_loss(y_val, y_prob)

    print(f"\n{'─' * 48}")
    print(f"  Validation results  ({len(X_val):,} rows)")
    print(f"{'─' * 48}")
    print(f"  Majority-class baseline accuracy : {100 * majority_acc_val:.2f}%")
    print(f"  Model accuracy                   : {100 * acc:.2f}%")
    print(f"  ROC-AUC                          : {auc:.4f}  (0.5=random, 1.0=perfect)")
    print(f"  Log loss                         : {ll:.4f}  (lower is better)")

    # ------------------------------------------------------------------ #
    # Feature importance                                                   #
    # ------------------------------------------------------------------ #
    importances = pd.Series(model.feature_importances_, index=model_cols)
    top15 = importances.sort_values(ascending=False).head(15)

    print(f"\n  Top 15 features by importance:")
    for feat, score in top15.items():
        print(f"    {feat:<40} {score:.4f}")

    # ------------------------------------------------------------------ #
    # Save model                                                           #
    # ------------------------------------------------------------------ #
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"\n  Model saved to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_baseline_model()

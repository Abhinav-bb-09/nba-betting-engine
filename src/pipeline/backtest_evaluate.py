from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.pipeline.define_model_features import get_model_feature_columns, get_target_column

PROJECT_ROOT  = Path(__file__).parents[2]
TEST_PATH     = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH    = PROJECT_ROOT / "models" / "baseline_xgb_tuned.json"
OUT_PATH      = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"

IDENTIFIER_COLS = {"GAME_ID", "GAME_DATE"}


def evaluate_on_test() -> pd.DataFrame:
    """Evaluate the tuned baseline model on the held-out 2024-25 test season.

    This function is called exactly once at the end of Phase 3.  The test set
    (season 2025) has been locked away since the beginning of this project —
    it was never used to select features, tune hyperparameters, or make any
    modelling decisions.  The numbers produced here are the first unbiased
    estimate of real-world performance.

    Column filtering mirrors tune_model_cv.py: the whitelisted feature columns
    are intersected with those actually present in the loaded CSV, so this
    function remains safe if pointed at any feature-matrix version.

    Outputs
    -------
    data/processed/test_predictions.csv
        One row per game with identifiers, spread, actual outcome, predicted
        probability, and predicted class — the input file for bet simulation.

    Returns
    -------
    pd.DataFrame
        The test_predictions DataFrame.
    """
    # ------------------------------------------------------------------ #
    # Load test data and model                                             #
    # ------------------------------------------------------------------ #
    print(f"Loading test set: {TEST_PATH.name} ...")
    test = pd.read_csv(TEST_PATH)
    print(f"  {len(test):,} rows  (season 2025 = 2024-25)")

    print(f"\nLoading model: {MODEL_PATH.name} ...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    # ------------------------------------------------------------------ #
    # Build feature matrix (same filtering pattern as tune_model_cv.py)   #
    # ------------------------------------------------------------------ #
    available      = set(test.columns)
    all_whitelisted = [c for c in get_model_feature_columns() if c not in IDENTIFIER_COLS]
    model_cols     = [c for c in all_whitelisted if c in available]
    filtered_out   = [c for c in all_whitelisted if c not in available]
    target_col     = get_target_column()

    if filtered_out:
        print(f"\n  WARNING — {len(filtered_out)} whitelisted column(s) absent from test data:")
        for col in filtered_out:
            print(f"    FILTERED: {col}")
    print(f"  Using {len(model_cols)} / {len(all_whitelisted)} whitelisted feature columns")

    X_test = test[model_cols]
    y_test = test[target_col]

    # ------------------------------------------------------------------ #
    # Generate predictions                                                 #
    # ------------------------------------------------------------------ #
    predicted_prob  = model.predict_proba(X_test)[:, 1]
    predicted_class = model.predict(X_test)

    # ------------------------------------------------------------------ #
    # Evaluate — first look at 2024-25 performance                        #
    # ------------------------------------------------------------------ #
    acc = accuracy_score(y_test, predicted_class)
    auc = roc_auc_score(y_test, predicted_prob)
    ll  = log_loss(y_test, predicted_prob)

    majority_class   = int(y_test.mode()[0])
    majority_acc     = (y_test == majority_class).mean()

    print(f"\n{'═' * 56}")
    print(f"  FIRST LOOK: 2024-25 TEST SET RESULTS  ({len(test):,} games)")
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  *** These numbers have never been seen before. ***")
    print(f"{'═' * 56}")
    print(f"  Majority-class baseline accuracy : {100 * majority_acc:.2f}%"
          f"  (always predict class {majority_class})")
    print(f"{'─' * 56}")
    print(f"  Model accuracy                   : {100 * acc:.2f}%")
    print(f"  ROC-AUC                          : {auc:.4f}  (0.5=random, 1.0=perfect)")
    print(f"  Log loss                         : {ll:.4f}  (lower is better)")
    print(f"{'═' * 56}")

    # ------------------------------------------------------------------ #
    # Save test_predictions.csv                                            #
    # ------------------------------------------------------------------ #
    preds = test[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_home",
                  "TEAM_ABBREVIATION_away", "spread", target_col]].copy()
    preds["predicted_prob"]  = predicted_prob
    preds["predicted_class"] = predicted_class

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(preds):,} rows to {OUT_PATH.name}")
    print(f"  Columns: {list(preds.columns)}")

    return preds


if __name__ == "__main__":
    evaluate_on_test()

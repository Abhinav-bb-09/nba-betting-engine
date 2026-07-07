import itertools
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

from src.pipeline.define_model_features import get_model_feature_columns, get_target_column

PROJECT_ROOT = Path(__file__).parents[2]
TRAIN_PATH   = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH     = PROJECT_ROOT / "data" / "processed" / "validation.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_PATH   = MODELS_DIR / "baseline_xgb_tuned.json"

IDENTIFIER_COLS = {"GAME_ID", "GAME_DATE"}

# Chronological folds — always train on earlier seasons, validate on later.
# Each fold adds one more season of history so the final fold mirrors the
# real deployment setting (3 years of history → predict the 4th year).
FOLDS = [
    {"train": [2021],             "val": [2022]},
    {"train": [2021, 2022],       "val": [2023]},
    {"train": [2021, 2022, 2023], "val": [2024]},
]

# Hyperparameter grid: 4 × 3 × 3 = 36 combinations × 3 folds = 108 fits.
PARAM_GRID = {
    "max_depth":     [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators":  [100, 200, 300],
}

# subsample and colsample_bytree are fixed rather than tuned to keep the
# grid at a tractable 36 combinations.  Adding even one more binary choice
# doubles the search space to 72+ combinations × 3 folds.  These values
# (0.8) are safe, widely-used defaults that add mild regularisation without
# needing individual optimisation at this dataset size.
FIXED_PARAMS = {
    "subsample":        0.8,
    "colsample_bytree": 0.8,
}


def tune_model_cv() -> xgb.XGBClassifier:
    """Grid-search XGBoost hyperparameters using chronological cross-validation.

    Using random or stratified k-fold on time-series data would let future
    games leak into training.  Instead we use three expanding-window folds
    that always train on earlier seasons and validate on the immediately
    following season — mimicking how the model will actually be used.

    After selecting the best hyperparameters by mean ROC-AUC across folds,
    the final model is retrained on the full 4-season dataset (2021-2024)
    so it benefits from all available history before being evaluated on the
    held-out 2025 test season.

    Returns
    -------
    xgb.XGBClassifier
        Final model retrained on all seasons 2021-2024 using the best
        hyperparameters found during cross-validation.
    """
    # ------------------------------------------------------------------ #
    # Load and concatenate train + validation                              #
    # ------------------------------------------------------------------ #
    print("Loading data ...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    data     = pd.concat([train_df, val_df], ignore_index=True)
    print(f"  Combined: {len(data):,} rows across seasons {sorted(data['season'].unique())}")

    # Intersect whitelist with columns actually present in the loaded data so
    # this script works correctly regardless of which feature-matrix version
    # the splits were built from (v1 doesn't have SOS/streak/rest columns).
    available      = set(data.columns)
    all_whitelisted = [c for c in get_model_feature_columns() if c not in IDENTIFIER_COLS]
    model_cols     = [c for c in all_whitelisted if c in available]
    filtered_out   = [c for c in all_whitelisted if c not in available]
    target_col     = get_target_column()

    if filtered_out:
        print(f"  WARNING — {len(filtered_out)} whitelisted column(s) not found in data and excluded:")
        for col in filtered_out:
            print(f"    FILTERED: {col}")
    print(f"  Using {len(model_cols)} / {len(all_whitelisted)} whitelisted feature columns")

    # ------------------------------------------------------------------ #
    # Cross-validation grid search                                         #
    # ------------------------------------------------------------------ #
    keys         = list(PARAM_GRID.keys())
    combinations = [
        dict(zip(keys, combo))
        for combo in itertools.product(*PARAM_GRID.values())
    ]
    n_total = len(combinations) * len(FOLDS)
    print(f"\n  Grid: {len(combinations)} combinations × {len(FOLDS)} folds = {n_total} fits\n")

    results = []

    for i, params in enumerate(combinations, 1):
        fold_aucs = []
        fold_accs = []

        for fold in FOLDS:
            train_mask = data["season"].isin(fold["train"])
            val_mask   = data["season"].isin(fold["val"])

            X_tr = data.loc[train_mask, model_cols]
            y_tr = data.loc[train_mask, target_col]
            X_v  = data.loc[val_mask,   model_cols]
            y_v  = data.loc[val_mask,   target_col]

            model = xgb.XGBClassifier(
                **params,
                **FIXED_PARAMS,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            model.fit(X_tr, y_tr)

            y_prob = model.predict_proba(X_v)[:, 1]
            y_pred = model.predict(X_v)

            fold_aucs.append(roc_auc_score(y_v, y_prob))
            fold_accs.append(accuracy_score(y_v, y_pred))

        results.append({
            **params,
            "fold_1_auc": fold_aucs[0],
            "fold_2_auc": fold_aucs[1],
            "fold_3_auc": fold_aucs[2],
            "mean_auc":   sum(fold_aucs) / len(fold_aucs),
            "std_auc":    pd.Series(fold_aucs).std(),
            "mean_acc":   sum(fold_accs) / len(fold_accs),
        })

        if i % 9 == 0 or i == len(combinations):
            print(f"  [{i:>2}/{len(combinations)}] last: depth={params['max_depth']} "
                  f"lr={params['learning_rate']} n={params['n_estimators']}  "
                  f"mean_auc={results[-1]['mean_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # Results table sorted by mean ROC-AUC                                 #
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame(results).sort_values("mean_auc", ascending=False).reset_index(drop=True)

    print(f"\n{'─' * 88}")
    print(f"  {'depth':>5}  {'lr':>5}  {'n_est':>5}  "
          f"{'fold1':>7}  {'fold2':>7}  {'fold3':>7}  {'mean_auc':>9}  {'mean_acc':>9}")
    print(f"{'─' * 88}")
    for _, row in results_df.iterrows():
        print(f"  {int(row['max_depth']):>5}  {row['learning_rate']:>5.2f}  {int(row['n_estimators']):>5}  "
              f"{row['fold_1_auc']:>7.4f}  {row['fold_2_auc']:>7.4f}  {row['fold_3_auc']:>7.4f}  "
              f"{row['mean_auc']:>9.4f}  {row['mean_acc']:>9.4f}")
    print(f"{'─' * 88}")

    # ------------------------------------------------------------------ #
    # Top 5 by mean AUC with standard deviation                           #
    # ------------------------------------------------------------------ #
    # A high mean AUC achieved by one strong fold and two weak ones is less
    # trustworthy than a slightly lower mean that is consistent across all
    # three folds.  Std dev across folds surfaces this instability so we can
    # prefer a stable winner over a lucky one.
    print(f"\n  Top 5 by mean ROC-AUC (with fold stability):")
    print(f"  {'depth':>5}  {'lr':>5}  {'n_est':>5}  {'mean_auc':>9}  {'std_auc':>8}")
    for _, row in results_df.head(5).iterrows():
        print(f"  {int(row['max_depth']):>5}  {row['learning_rate']:>5.2f}  "
              f"{int(row['n_estimators']):>5}  {row['mean_auc']:>9.4f}  {row['std_auc']:>8.4f}")

    # ------------------------------------------------------------------ #
    # Best hyperparameters                                                  #
    # ------------------------------------------------------------------ #
    best = results_df.iloc[0]
    best_params = {
        "max_depth":     int(best["max_depth"]),
        "learning_rate": best["learning_rate"],
        "n_estimators":  int(best["n_estimators"]),
    }
    print(f"\n  Best hyperparameters: {best_params}")
    print(f"  Mean ROC-AUC: {best['mean_auc']:.4f}  |  Std: {best['std_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # Retrain final model on full 2021-2024 data                           #
    # ------------------------------------------------------------------ #
    print(f"\n  Retraining on full 2021-2024 data ({len(data):,} rows) ...")
    X_full = data[model_cols]
    y_full = data[target_col]

    final_model = xgb.XGBClassifier(
        **best_params,
        **FIXED_PARAMS,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X_full, y_full)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(MODEL_PATH))
    print(f"  Saved to {MODEL_PATH}")

    return final_model


if __name__ == "__main__":
    tune_model_cv()

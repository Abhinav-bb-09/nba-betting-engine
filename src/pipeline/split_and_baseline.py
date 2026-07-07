from pathlib import Path

import pandas as pd

from src.pipeline.define_model_features import get_model_feature_columns, get_target_column

PROJECT_ROOT = Path(__file__).parents[2]
MATRIX_PATH  = PROJECT_ROOT / "data" / "processed" / "feature_matrix.csv"
TRAIN_PATH   = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH     = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_PATH    = PROJECT_ROOT / "data" / "processed" / "test.csv"

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASONS   = [2024]
TEST_SEASONS  = [2025]


def split_and_baseline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split feature_matrix into train/validation/test by season and compute baselines.

    We use a strict time-based split — no shuffling — so that the model is
    always evaluated on games it could not have trained on.  Seasons are
    identified by their ending year (2021 = 2020-21 season).

      Train      : 2021, 2022, 2023  (3 seasons of history)
      Validation : 2024              (used for hyperparameter tuning)
      Test       : 2025              (held out until final evaluation)

    The majority-class baseline tells us the accuracy achievable by a trivial
    classifier that ignores all features and always predicts the most common
    class from training.  Any real model must beat this to be considered
    useful.

    Steps
    -----
    1. Load feature_matrix.csv.
    2. Split by season column.
    3. Print per-split row counts, target distribution, and null counts across
       approved feature columns.
    4. Compute majority-class baseline accuracy on validation and test.
    5. Save splits to data/processed/.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, validation, test) DataFrames, each containing all columns
        from feature_matrix.csv.
    """
    # ------------------------------------------------------------------ #
    # 1. Load                                                              #
    # ------------------------------------------------------------------ #
    print(f"Loading {MATRIX_PATH} ...")
    df = pd.read_csv(MATRIX_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"  {len(df):,} total rows, {df.shape[1]} columns\n")

    feature_cols = get_model_feature_columns()
    target_col   = get_target_column()

    # ------------------------------------------------------------------ #
    # 2. Split by season                                                   #
    # ------------------------------------------------------------------ #
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val   = df[df["season"].isin(VAL_SEASONS)].copy()
    test  = df[df["season"].isin(TEST_SEASONS)].copy()

    # ------------------------------------------------------------------ #
    # 3. Per-split diagnostics                                             #
    # ------------------------------------------------------------------ #
    splits = [
        ("Train",      train, TRAIN_SEASONS),
        ("Validation", val,   VAL_SEASONS),
        ("Test",       test,  TEST_SEASONS),
    ]

    for name, split_df, seasons in splits:
        target_vc  = split_df[target_col].value_counts()
        n          = len(split_df)
        covers     = target_vc.get(1, 0)
        no_covers  = target_vc.get(0, 0)

        null_rows  = split_df[feature_cols].isnull().any(axis=1).sum()

        # Break down which columns drive the nulls — helpful because
        # recent_injuries_count_home/away are NaN by design for 2024/2025
        # (outside injury data coverage) and should not be confused with
        # genuine data quality problems.
        null_by_col = split_df[feature_cols].isnull().sum()
        null_by_col = null_by_col[null_by_col > 0]

        print(f"{'─' * 52}")
        print(f"  {name}  (seasons {seasons})")
        print(f"{'─' * 52}")
        print(f"  Rows          : {n:,}")
        print(f"  Covers    (1) : {covers:,}  ({100 * covers / n:.1f}%)")
        print(f"  No cover  (0) : {no_covers:,}  ({100 * no_covers / n:.1f}%)")
        print(f"  Null feature rows: {null_rows:,}  ({100 * null_rows / n:.1f}%)")
        if not null_by_col.empty:
            for col, cnt in null_by_col.items():
                print(f"    {col}: {cnt:,}")
        print()

    # ------------------------------------------------------------------ #
    # 4. Majority-class baseline accuracy                                  #
    # ------------------------------------------------------------------ #
    # The majority class is determined by the training set — in a real
    # deployment we would not peek at the val/test target distribution.
    train_majority_class = train[target_col].mode()[0]
    train_majority_pct   = (train[target_col] == train_majority_class).mean()

    print(f"{'─' * 52}")
    print(f"  Majority-class baseline  (class={train_majority_class}, "
          f"train prevalence={100 * train_majority_pct:.1f}%)")
    print(f"{'─' * 52}")

    for name, split_df in [("Validation", val), ("Test", test)]:
        baseline_acc = (split_df[target_col] == train_majority_class).mean()
        print(f"  {name:<12} majority-class accuracy: {100 * baseline_acc:.1f}%")

    print()

    # ------------------------------------------------------------------ #
    # 5. Save                                                              #
    # ------------------------------------------------------------------ #
    for path, split_df, name in [
        (TRAIN_PATH, train, "train"),
        (VAL_PATH,   val,   "validation"),
        (TEST_PATH,  test,  "test"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(path, index=False)
        print(f"  Saved {name:<12}: {len(split_df):,} rows → {path.name}")

    return train, val, test


if __name__ == "__main__":
    split_and_baseline()

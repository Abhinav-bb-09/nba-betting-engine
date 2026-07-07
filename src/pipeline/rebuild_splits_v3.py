from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
MATRIX_PATH  = PROJECT_ROOT / "data" / "processed" / "feature_matrix_v3.csv"
TRAIN_PATH   = PROJECT_ROOT / "data" / "processed" / "train_v3.csv"
VAL_PATH     = PROJECT_ROOT / "data" / "processed" / "validation_v3.csv"
TEST_PATH    = PROJECT_ROOT / "data" / "processed" / "test_v3.csv"

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASONS   = [2024]
TEST_SEASONS  = [2025]


def rebuild_splits_v3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split feature_matrix_v3.csv into train/validation/test by season.

    Adds rest_advantage and current_streak_home/away over v2 splits.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, validation, test) DataFrames.
    """
    print(f"Loading {MATRIX_PATH.name} ...")
    df = pd.read_csv(MATRIX_PATH)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val   = df[df["season"].isin(VAL_SEASONS)].copy()
    test  = df[df["season"].isin(TEST_SEASONS)].copy()

    for path, split, name in [
        (TRAIN_PATH, train, "train_v3"),
        (VAL_PATH,   val,   "validation_v3"),
        (TEST_PATH,  test,  "test_v3"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        print(f"  Saved {name}: {len(split):,} rows → {path.name}")

    return train, val, test


if __name__ == "__main__":
    rebuild_splits_v3()

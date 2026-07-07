from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
MATRIX_PATH  = PROJECT_ROOT / "data" / "processed" / "feature_matrix_v2.csv"
TRAIN_PATH   = PROJECT_ROOT / "data" / "processed" / "train_v2.csv"
VAL_PATH     = PROJECT_ROOT / "data" / "processed" / "validation_v2.csv"
TEST_PATH    = PROJECT_ROOT / "data" / "processed" / "test_v2.csv"

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASONS   = [2024]
TEST_SEASONS  = [2025]


def rebuild_splits_v2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-split feature_matrix_v2.csv into train/validation/test by season.

    Identical split logic to split_and_baseline.py but reads from
    feature_matrix_v2.csv (which adds sos_rolling_10_home/away) and writes
    to _v2 output files so the v1 splits are not overwritten.

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
        (TRAIN_PATH, train, "train_v2"),
        (VAL_PATH,   val,   "validation_v2"),
        (TEST_PATH,  test,  "test_v2"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        print(f"  Saved {name}: {len(split):,} rows → {path.name}")

    return train, val, test


if __name__ == "__main__":
    rebuild_splits_v2()

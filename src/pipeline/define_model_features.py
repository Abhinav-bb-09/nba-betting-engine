"""
Feature whitelist for Phase 3 modeling.

feature_matrix.csv retains raw box-score columns (PTS_home, FGA_home,
WL_home, etc.) for traceability and debugging — they let you audit any
row and verify that the pipeline processed it correctly.  However, those
columns are OUTCOMES of the game being predicted and must NEVER be used
as model inputs.  Using them would be catastrophic data leakage: the model
would learn to predict the spread cover from the final score, which is
information that does not exist at tip-off.

get_model_feature_columns() is the single source of truth for what Phase 3
is allowed to train on.  Any change to the feature set must go through this
function so that training, evaluation, and inference all stay in sync.
"""

# ---------------------------------------------------------------------------
# Rolling team-form features (computed in build_rolling_features.py)
# All use .shift(1) before .rolling() — no leakage from the current game.
# ---------------------------------------------------------------------------
_ROLLING_FORM = [
    "rolling_win_pct_5",
    "rolling_win_pct_10",
    "rolling_pt_diff_5",
    "rolling_pt_diff_10",
    "days_rest",
    "is_back_to_back",
]

# ---------------------------------------------------------------------------
# Rolling efficiency features (computed in build_efficiency_features.py)
# Dean Oliver ORtg/DRtg/TS% rolled over last 5 and 10 games.
# ---------------------------------------------------------------------------
_ROLLING_EFFICIENCY = [
    "rolling_ortg_5",
    "rolling_ortg_10",
    "rolling_drtg_5",
    "rolling_drtg_10",
    "rolling_ts_pct_5",
    "rolling_ts_pct_10",
]

# All _home columns first, then all _away columns — matching the spec ordering.
_BASE_ROLLING = _ROLLING_FORM + _ROLLING_EFFICIENCY
_ALL_ROLLING = (
    [f"{col}_home" for col in _BASE_ROLLING] +
    [f"{col}_away" for col in _BASE_ROLLING]
)

# ---------------------------------------------------------------------------
# Injury features (computed in build_injury_features.py)
# NaN for 2023-24 / 2024-25 — always check has_injury_data before using.
# ---------------------------------------------------------------------------
_INJURY = [
    "recent_injuries_count_home",
    "recent_injuries_count_away",
    "has_injury_data",
]

# ---------------------------------------------------------------------------
# Betting-line inputs known before tip-off (from the sportsbook)
# ---------------------------------------------------------------------------
_BETTING_INPUTS = [
    "spread",
    "total",
]

# ---------------------------------------------------------------------------
# Identifying columns — needed for joins, time-series splits, and auditing,
# but must be excluded from the feature matrix passed to sklearn/XGBoost.
# ---------------------------------------------------------------------------
_IDENTIFIERS = [
    "GAME_ID",
    "GAME_DATE",
]


def get_model_feature_columns() -> list[str]:
    """Return the whitelist of column names safe to use as model inputs.

    Includes all rolling team-form and efficiency features (both sides),
    injury context features, and pre-game betting line inputs.  Identifying
    columns (GAME_ID, GAME_DATE) are also returned so callers can separate
    them out for joins and time-series train/test splitting.

    Explicitly excluded (present in feature_matrix.csv but off-limits):
      - Raw box-score columns (PTS_*, FGA_*, WL_*, etc.) — game outcomes.
      - The target column (home_covers_spread).
      - All intermediate pipeline columns.

    Returns
    -------
    list[str]
        Ordered list of approved feature + identifier column names.
    """
    return _ALL_ROLLING + _INJURY + _BETTING_INPUTS + _IDENTIFIERS


def get_target_column() -> str:
    """Return the name of the binary target column.

    Returns
    -------
    str
        "home_covers_spread" — 1 if home team beat the spread, else 0.
    """
    return "home_covers_spread"


# ---------------------------------------------------------------------------
# Verification: run this file directly to confirm every approved column
# actually exists in feature_matrix.csv.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    matrix_path = Path(__file__).parents[2] / "data" / "processed" / "feature_matrix.csv"
    print(f"Loading {matrix_path} ...")
    df = pd.read_csv(matrix_path, nrows=0)  # headers only — no need to read all rows
    matrix_cols = set(df.columns)

    feature_cols = get_model_feature_columns()
    target_col   = get_target_column()

    missing = [c for c in feature_cols + [target_col] if c not in matrix_cols]

    print(f"\nApproved feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  {col}")

    print(f"\nTarget column: {target_col}")

    if missing:
        print(f"\n  ERROR — {len(missing)} column(s) not found in feature_matrix.csv:")
        for col in missing:
            print(f"    MISSING: {col}")
        raise SystemExit(1)
    else:
        print(
            f"\n  All {len(feature_cols)} feature columns and the target column "
            f"verified present in feature_matrix.csv."
        )

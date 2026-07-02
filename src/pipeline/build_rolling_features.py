from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
BASE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_base.csv"
OUT_PATH  = PROJECT_ROOT / "data" / "processed" / "feature_base_v2.csv"

ROLLING_COLS = [
    "rolling_win_pct_5",
    "rolling_win_pct_10",
    "rolling_pt_diff_5",
    "rolling_pt_diff_10",
    "days_rest",
    "is_back_to_back",
]


def build_rolling_features() -> pd.DataFrame:
    """Add per-team rolling history features to the feature base table.

    The NBA API game-log gives us one row per game, but rolling stats
    (recent form, fatigue) require looking across a team's game history.
    We reshape to a long team-game format, compute rolling features there,
    then pivot back to one row per game with _home / _away suffixes.

    Steps
    -----
    1. Load feature_base.csv.
    2. Reshape to long format: two rows per game — one for the home team
       and one for the away team — with pts_for / pts_against / win from
       that team's perspective.
    3. Sort by team + date and compute, within each team's group:
         - rolling_win_pct_5/10  : rolling mean of win over last 5/10 games
         - rolling_pt_diff_5/10  : rolling mean of point differential
         - days_rest             : calendar days since the team's last game
                                   (null for a team's very first game)
         - is_back_to_back       : 1 if days_rest == 1 (consecutive nights)
       All rolling calculations use .shift(1) so that the current game's
       outcome does not feed into its own feature — preventing leakage.
    4. Split long table back into home and away subsets, rename rolling
       columns with _home / _away suffixes, merge onto original table by
       GAME_ID.
    5. Report null counts in the new rolling columns (expected for each
       team's first ~10 games in the dataset) and assess plausibility.
    6. Save to data/processed/feature_base_v2.csv.

    Returns
    -------
    pd.DataFrame
        feature_base extended with 12 new rolling feature columns
        (6 per side × 2 sides).
    """
    # ------------------------------------------------------------------ #
    # 1. Load                                                              #
    # ------------------------------------------------------------------ #
    print(f"Loading {BASE_PATH} ...")
    base = pd.read_csv(BASE_PATH)
    base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
    print(f"  {len(base):,} rows, {base.shape[1]} columns")

    # ------------------------------------------------------------------ #
    # 2. Reshape to long team-game format                                  #
    # ------------------------------------------------------------------ #
    home_view = base[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_home", "TEAM_ABBREVIATION_away", "PTS_home", "PTS_away"]].copy()
    home_view.columns = ["GAME_ID", "GAME_DATE", "team", "opponent", "pts_for", "pts_against"]
    home_view["is_home"] = 1
    home_view["win"] = (home_view["pts_for"] > home_view["pts_against"]).astype(int)

    away_view = base[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_away", "TEAM_ABBREVIATION_home", "PTS_away", "PTS_home"]].copy()
    away_view.columns = ["GAME_ID", "GAME_DATE", "team", "opponent", "pts_for", "pts_against"]
    away_view["is_home"] = 0
    away_view["win"] = (away_view["pts_for"] > away_view["pts_against"]).astype(int)

    long_df = pd.concat([home_view, away_view], ignore_index=True)

    # ------------------------------------------------------------------ #
    # 3. Sort and compute rolling features per team                        #
    # ------------------------------------------------------------------ #
    long_df = long_df.sort_values(["team", "GAME_DATE"]).reset_index(drop=True)

    long_df["pt_diff"] = long_df["pts_for"] - long_df["pts_against"]

    g = long_df.groupby("team", sort=False)

    # .shift(1) moves each value one row forward within the group so that
    # game N's rolling window contains only games 1 through N-1.  Without
    # the shift, game N's own outcome would be included in its features,
    # which is data leakage — the model would see the answer at train time
    # but not at prediction time.
    long_df["rolling_win_pct_5"]  = g["win"].transform(lambda x: x.shift(1).rolling(5).mean())
    long_df["rolling_win_pct_10"] = g["win"].transform(lambda x: x.shift(1).rolling(10).mean())
    long_df["rolling_pt_diff_5"]  = g["pt_diff"].transform(lambda x: x.shift(1).rolling(5).mean())
    long_df["rolling_pt_diff_10"] = g["pt_diff"].transform(lambda x: x.shift(1).rolling(10).mean())

    # diff() gives NaT for the first game of each team — intentional.
    # We want null, not 0, so no fillna here.
    long_df["days_rest"] = g["GAME_DATE"].transform(lambda x: x.diff().dt.days)

    # NaN == 1 evaluates to False in pandas, so the first game correctly
    # gets is_back_to_back = 0 rather than NaN.
    long_df["is_back_to_back"] = (long_df["days_rest"] == 1).astype(int)

    # ------------------------------------------------------------------ #
    # 4. Pivot back to one row per game with _home / _away suffixes        #
    # ------------------------------------------------------------------ #
    def extract_side(is_home_val, suffix):
        side = long_df[long_df["is_home"] == is_home_val][["GAME_ID"] + ROLLING_COLS].copy()
        return side.rename(columns={c: f"{c}{suffix}" for c in ROLLING_COLS})

    home_features = extract_side(1, "_home")
    away_features = extract_side(0, "_away")

    result = base.merge(home_features, on="GAME_ID", how="left") \
                 .merge(away_features, on="GAME_ID", how="left")

    # ------------------------------------------------------------------ #
    # 5. Null report                                                       #
    # ------------------------------------------------------------------ #
    suffixed_cols = [f"{c}_home" for c in ROLLING_COLS] + [f"{c}_away" for c in ROLLING_COLS]

    # days_rest and is_back_to_back don't require a rolling window so they
    # only produce nulls for each team's very first game (30 nulls max).
    # The win-rate and pt-diff columns need N prior games, so each team's
    # first N games are null.  With 30 teams and windows of 5 and 10, the
    # theoretical maximum nulls per rolling column is 30 × 10 = 300.
    print("\n  Null counts in new rolling feature columns:")
    any_rolling_null = result[[f"{c}_home" for c in ROLLING_COLS]].isnull().any(axis=1)
    print(f"  Rows with at least one null rolling feature: {any_rolling_null.sum():,}  "
          f"(expected ≤ ~300 given 30 teams × 10-game max window)")
    print()
    for col in suffixed_cols:
        n = result[col].isnull().sum()
        if n:
            print(f"    {col:<35} {n:>4} nulls")

    # ------------------------------------------------------------------ #
    # 6. Save                                                              #
    # ------------------------------------------------------------------ #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(result):,} rows × {result.shape[1]} columns to {OUT_PATH}")

    return result


if __name__ == "__main__":
    build_rolling_features()

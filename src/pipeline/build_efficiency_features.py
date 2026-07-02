from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
BASE_PATH    = PROJECT_ROOT / "data" / "processed" / "feature_base_v3.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "processed" / "feature_matrix.csv"

# Per-game intermediate columns — computed here, dropped before saving.
# Only the rolling versions are retained as model features.
GAME_COLS = [
    "possessions_home", "possessions_away",
    "ortg_game_home",   "ortg_game_away",
    "drtg_game_home",   "drtg_game_away",
    "ts_pct_game_home", "ts_pct_game_away",
]

EFFICIENCY_COLS = ["ortg_game", "drtg_game", "ts_pct_game"]

ROLLING_EFFICIENCY_COLS = [
    "rolling_ortg_5",  "rolling_ortg_10",
    "rolling_drtg_5",  "rolling_drtg_10",
    "rolling_ts_pct_5", "rolling_ts_pct_10",
]


def build_efficiency_features() -> pd.DataFrame:
    """Add rolling team-efficiency features (ORtg, DRtg, TS%) to the feature matrix.

    Box-score totals (FGA, FTA, OREB, TOV, PTS) are noisier game-to-game
    than efficiency rates.  Rolling efficiency ratings capture how well a
    team has been shooting and defending over recent games, which is a
    stronger signal for spread prediction than raw counting stats.

    Steps
    -----
    1. Load feature_base_v3.csv.
    2. Compute per-game efficiency metrics in the wide table for both sides:
         possessions  — Dean Oliver estimate: FGA - OREB + TOV + 0.44*FTA
         ORtg         — offensive rating: 100 * PTS / possessions
         DRtg         — defensive rating: 100 * opponent_PTS / opponent_possessions
         TS%          — true shooting: PTS / (2 * (FGA + 0.44*FTA))
    3. Reshape to long team-game format (one row per team per game) carrying
       the three per-game efficiency metrics.
    4. Sort by team + date; compute rolling means at windows 5 and 10 using
       .shift(1) before .rolling() to prevent leakage (same pattern as
       build_rolling_features.py — current game must not appear in its own
       feature window).
    5. Pivot back to wide format with _home / _away suffixes; merge onto
       the main table by GAME_ID.
    6. Drop the intermediate single-game columns — they reflect the outcome
       of the current game and must not be model features.
    7. Print null counts and a plausibility check on rolling_ortg_10_home
       (real NBA team ORtg values cluster roughly 95–125).
    8. Save final feature matrix to data/processed/feature_matrix.csv.

    Returns
    -------
    pd.DataFrame
        feature_base_v3 extended with 12 rolling efficiency columns
        (6 metrics × 2 sides), with intermediate game-level columns removed.
    """
    # ------------------------------------------------------------------ #
    # 1. Load                                                              #
    # ------------------------------------------------------------------ #
    print(f"Loading {BASE_PATH} ...")
    df = pd.read_csv(BASE_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    # ------------------------------------------------------------------ #
    # 2. Per-game efficiency metrics (wide, both sides)                    #
    # ------------------------------------------------------------------ #
    # Dean Oliver possession estimate — the standard approximation used
    # across basketball analytics when play-by-play data is unavailable.
    # The 0.44 coefficient on FTA accounts for the-and-one free throws and
    # technical foul shots that don't consume a "real" possession each.
    for s in ("home", "away"):
        df[f"possessions_{s}"] = (
            df[f"FGA_{s}"]
            - df[f"OREB_{s}"]
            + df[f"TOV_{s}"]
            + 0.44 * df[f"FTA_{s}"]
        )

    # Offensive rating: points scored per 100 possessions (own possessions).
    # Defensive rating: points allowed per 100 of the *opponent's* possessions.
    # True shooting %: shooting efficiency accounting for 3-pointers and FTs.
    df["ortg_game_home"]   = 100 * df["PTS_home"] / df["possessions_home"]
    df["ortg_game_away"]   = 100 * df["PTS_away"] / df["possessions_away"]

    df["drtg_game_home"]   = 100 * df["PTS_away"] / df["possessions_away"]
    df["drtg_game_away"]   = 100 * df["PTS_home"] / df["possessions_home"]

    df["ts_pct_game_home"] = df["PTS_home"] / (2 * (df["FGA_home"] + 0.44 * df["FTA_home"]))
    df["ts_pct_game_away"] = df["PTS_away"] / (2 * (df["FGA_away"] + 0.44 * df["FTA_away"]))

    # ------------------------------------------------------------------ #
    # 3. Reshape to long team-game format                                  #
    # ------------------------------------------------------------------ #
    home_view = df[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_home",
                    "ortg_game_home", "drtg_game_home", "ts_pct_game_home"]].copy()
    home_view.columns = ["GAME_ID", "GAME_DATE", "team",
                         "ortg_game", "drtg_game", "ts_pct_game"]
    home_view["is_home"] = 1

    away_view = df[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_away",
                    "ortg_game_away", "drtg_game_away", "ts_pct_game_away"]].copy()
    away_view.columns = ["GAME_ID", "GAME_DATE", "team",
                         "ortg_game", "drtg_game", "ts_pct_game"]
    away_view["is_home"] = 0

    long_df = pd.concat([home_view, away_view], ignore_index=True)
    long_df = long_df.sort_values(["team", "GAME_DATE"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 4. Rolling efficiency per team, with shift(1) to prevent leakage    #
    # ------------------------------------------------------------------ #
    # .shift(1) ensures game N's rolling window spans games 1…N-1 only.
    # Without the shift the current game's own rating would be included,
    # which is leakage — the model would see the outcome at train time
    # but cannot know it at prediction time.
    g = long_df.groupby("team", sort=False)

    long_df["rolling_ortg_5"]   = g["ortg_game"].transform(lambda x: x.shift(1).rolling(5).mean())
    long_df["rolling_ortg_10"]  = g["ortg_game"].transform(lambda x: x.shift(1).rolling(10).mean())
    long_df["rolling_drtg_5"]   = g["drtg_game"].transform(lambda x: x.shift(1).rolling(5).mean())
    long_df["rolling_drtg_10"]  = g["drtg_game"].transform(lambda x: x.shift(1).rolling(10).mean())
    long_df["rolling_ts_pct_5"] = g["ts_pct_game"].transform(lambda x: x.shift(1).rolling(5).mean())
    long_df["rolling_ts_pct_10"]= g["ts_pct_game"].transform(lambda x: x.shift(1).rolling(10).mean())

    # ------------------------------------------------------------------ #
    # 5. Pivot back to wide and merge onto main table                      #
    # ------------------------------------------------------------------ #
    def extract_side(is_home_val, suffix):
        side = long_df[long_df["is_home"] == is_home_val][["GAME_ID"] + ROLLING_EFFICIENCY_COLS].copy()
        return side.rename(columns={c: f"{c}{suffix}" for c in ROLLING_EFFICIENCY_COLS})

    home_features = extract_side(1, "_home")
    away_features = extract_side(0, "_away")

    result = df.merge(home_features, on="GAME_ID", how="left") \
               .merge(away_features, on="GAME_ID", how="left")

    # ------------------------------------------------------------------ #
    # 6. Drop intermediate single-game columns                             #
    # ------------------------------------------------------------------ #
    result = result.drop(columns=GAME_COLS)

    # ------------------------------------------------------------------ #
    # 7. Diagnostics                                                       #
    # ------------------------------------------------------------------ #
    suffixed_rolling = (
        [f"{c}_home" for c in ROLLING_EFFICIENCY_COLS] +
        [f"{c}_away" for c in ROLLING_EFFICIENCY_COLS]
    )

    print("\n  Null counts in new rolling efficiency columns:")
    any_null = result[[f"{c}_home" for c in ROLLING_EFFICIENCY_COLS]].isnull().any(axis=1)
    print(f"  Rows with at least one null: {any_null.sum():,}  (expected ≤ ~300, early-season only)")
    for col in suffixed_rolling:
        n = result[col].isnull().sum()
        if n:
            print(f"    {col:<35} {n:>4} nulls")

    ortg = result["rolling_ortg_10_home"].dropna()
    print(f"\n  rolling_ortg_10_home plausibility check (real NBA range ≈ 95–125):")
    print(f"    min={ortg.min():.1f}  max={ortg.max():.1f}  mean={ortg.mean():.1f}  median={ortg.median():.1f}")

    # ------------------------------------------------------------------ #
    # 8. Save                                                              #
    # ------------------------------------------------------------------ #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(result):,} rows × {result.shape[1]} columns to {OUT_PATH}")

    return result


if __name__ == "__main__":
    build_efficiency_features()

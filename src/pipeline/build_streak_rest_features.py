from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
MATRIX_PATH  = PROJECT_ROOT / "data" / "processed" / "feature_matrix_v2.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "processed" / "feature_matrix_v3.csv"

VERIFY_TEAMS = ["BOS", "LAL", "GSW"]  # 3 well-known teams for manual streak check


def _compute_streak(win_series: pd.Series) -> pd.Series:
    """Compute signed current-streak entering each game, within one team's group.

    Algorithm
    ---------
    1. shift(1) — push every result one row forward so that game N only
       sees games 0..N-1 (leakage prevention).
    2. Detect value changes in the shifted series with .ne(shift(1)).cumsum()
       to assign a monotonically increasing group ID each time the result
       flips (win→loss or loss→win).
    3. cumcount()+1 within each group gives streak length (1, 2, 3, ...).
    4. Multiply by the sign of the group (+1 for wins, −1 for losses).

    The first game in each team's history is NaN because shift(1) produces
    NaN there, which propagates through the sign multiplication.

    Args:
        win_series: Per-team win/loss series (1=win, 0=loss), sorted by date.

    Returns:
        Signed streak series: +N = N-game win streak entering this game,
        −N = N-game losing streak.  NaN for the team's first game.
    """
    # shift(1): game N sees results of games 0..N-1 only.
    prev = win_series.shift(1)

    # Identify group boundaries: True wherever the value (or NaN status) changes.
    # NaN comparisons always return True, so position 0 (NaN) and position 1
    # (first real result, different from the preceding NaN) each start a new
    # group — which is exactly what we want.
    group_id   = prev.ne(prev.shift(1)).cumsum()
    streak_len = group_id.groupby(group_id).cumcount() + 1

    # Sign: +1 if the streak was wins, −1 if losses, NaN for the first position.
    sign = prev.map({1.0: 1, 0.0: -1})

    return streak_len * sign


def build_streak_rest_features() -> pd.DataFrame:
    """Add rest_advantage and current_streak features to the feature matrix.

    rest_advantage
    --------------
    days_rest_home − days_rest_away.  Positive means the home team is more
    rested; negative means the away team is.  Computed directly on the wide
    table — no reshaping required since both rest columns already exist.

    current_streak
    --------------
    Signed count of consecutive wins (+) or losses (−) entering the game.
    Requires the long team-game reshape because the streak is per-team, not
    per-game.  After reshaping and sorting by team + date, we apply the
    standard "group consecutive equal values" pattern with an upfront shift(1)
    to exclude the current game's own result (same leakage rule as every other
    rolling feature in this pipeline).

    Steps
    -----
    1. Load feature_matrix_v2.csv.
    2. Compute rest_advantage directly on the wide table.
    3. Reshape to long format; derive win from WL_home/WL_away.
    4. Sort by team + date; compute current_streak via _compute_streak.
    5. Print first-15-game samples for 3 teams as a manual sanity check.
    6. Pivot back, merge by GAME_ID, print null counts + plausibility check.
    7. Save to feature_matrix_v3.csv.

    Returns
    -------
    pd.DataFrame
        feature_matrix_v2 extended with rest_advantage,
        current_streak_home, and current_streak_away.
    """
    # ------------------------------------------------------------------ #
    # 1. Load                                                              #
    # ------------------------------------------------------------------ #
    print(f"Loading {MATRIX_PATH} ...")
    df = pd.read_csv(MATRIX_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    # ------------------------------------------------------------------ #
    # 2. rest_advantage — direct wide-table operation, no reshape needed   #
    # ------------------------------------------------------------------ #
    df["rest_advantage"] = df["days_rest_home"] - df["days_rest_away"]

    # ------------------------------------------------------------------ #
    # 3. Reshape to long team-game format                                  #
    # ------------------------------------------------------------------ #
    home_view = df[["GAME_ID", "GAME_DATE", "season", "TEAM_ABBREVIATION_home", "WL_home"]].copy()
    home_view.columns = ["GAME_ID", "GAME_DATE", "season", "team", "WL"]
    home_view["is_home"] = 1

    away_view = df[["GAME_ID", "GAME_DATE", "season", "TEAM_ABBREVIATION_away", "WL_away"]].copy()
    away_view.columns = ["GAME_ID", "GAME_DATE", "season", "team", "WL"]
    away_view["is_home"] = 0

    long_df = pd.concat([home_view, away_view], ignore_index=True)
    long_df["win"] = (long_df["WL"] == "W").astype(int)
    long_df = long_df.sort_values(["team", "season", "GAME_DATE"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 4. Compute current_streak per team per season                        #
    # ------------------------------------------------------------------ #
    # Grouping by ["team", "season"] resets the streak at every season
    # boundary — the first game of each season gets NaN, regardless of
    # how the previous season ended.  This prevents a team's end-of-season
    # form from bleeding into their opening-game context the following year.
    long_df["current_streak"] = (
        long_df.groupby(["team", "season"], sort=False)["win"]
        .transform(_compute_streak)
    )

    # ------------------------------------------------------------------ #
    # 5. Manual verification                                               #
    # ------------------------------------------------------------------ #
    print("\n  Manual streak verification (first 15 games per team):")
    print(f"  {'team':<5} {'season':>6} {'date':<12} {'win':>4}  {'streak':>7}")
    print(f"  {'─'*48}")
    for team in VERIFY_TEAMS:
        sample = (
            long_df[long_df["team"] == team]
            .sort_values(["season", "GAME_DATE"])
            .head(15)
        )
        for _, row in sample.iterrows():
            streak_str = f"{row['current_streak']:+.0f}" if pd.notna(row["current_streak"]) else "NaN"
            print(f"  {row['team']:<5} {int(row['season']):>6} {str(row['GAME_DATE'].date()):<12} "
                  f"{int(row['win']):>4}  {streak_str:>7}")
        print()

    # Season-boundary reset check: DET last 3 games of 2023, first 3 of 2024.
    # The first game of 2024 must show NaN, not a continuation of 2023's streak.
    print("  Season-boundary reset check (DET: last 3 of 2023, first 3 of 2024):")
    print(f"  {'team':<5} {'season':>6} {'date':<12} {'win':>4}  {'streak':>7}")
    print(f"  {'─'*48}")
    det = long_df[long_df["team"] == "DET"].sort_values(["season", "GAME_DATE"])
    tail_2023 = det[det["season"] == 2023].tail(3)
    head_2024 = det[det["season"] == 2024].head(3)
    for _, row in pd.concat([tail_2023, head_2024]).iterrows():
        streak_str = f"{row['current_streak']:+.0f}" if pd.notna(row["current_streak"]) else "NaN"
        print(f"  {row['team']:<5} {int(row['season']):>6} {str(row['GAME_DATE'].date()):<12} "
              f"{int(row['win']):>4}  {streak_str:>7}")
    print()

    # ------------------------------------------------------------------ #
    # 6. Pivot back and merge                                              #
    # ------------------------------------------------------------------ #
    home_feats = (
        long_df[long_df["is_home"] == 1][["GAME_ID", "current_streak"]]
        .rename(columns={"current_streak": "current_streak_home"})
    )
    away_feats = (
        long_df[long_df["is_home"] == 0][["GAME_ID", "current_streak"]]
        .rename(columns={"current_streak": "current_streak_away"})
    )

    result = df.merge(home_feats, on="GAME_ID", how="left") \
               .merge(away_feats, on="GAME_ID", how="left")

    # ------------------------------------------------------------------ #
    # Null counts                                                          #
    # ------------------------------------------------------------------ #
    new_cols = ["rest_advantage", "current_streak_home", "current_streak_away"]
    print(f"\n  Null counts:")
    for col in new_cols:
        n = result[col].isna().sum()
        print(f"    {col:<30} {n:>4} nulls  ({100*n/len(result):.1f}%)")

    # ------------------------------------------------------------------ #
    # Plausibility check                                                   #
    # ------------------------------------------------------------------ #
    # With season-bounded streaks the theoretical max is ~82 games (perfect
    # season), but realistically anything beyond ±30 would be historically
    # extraordinary.  The 2023-24 Pistons lost 28 consecutive games — the
    # NBA record — so a min near −28 in our dataset is expected and correct,
    # not a bug.  Values beyond ±33 (roughly half a season) would warrant
    # investigation.
    for col in ["current_streak_home", "current_streak_away"]:
        s = result[col].dropna()
        print(f"\n  {col} range check (expected ≈ −28 to +18, DET 28-game losing streak validates lower bound):")
        print(f"    min={s.min():.0f}  max={s.max():.0f}  mean={s.mean():.2f}")

    # ------------------------------------------------------------------ #
    # 7. Save                                                              #
    # ------------------------------------------------------------------ #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(result):,} rows × {result.shape[1]} columns to {OUT_PATH}")

    return result


if __name__ == "__main__":
    build_streak_rest_features()

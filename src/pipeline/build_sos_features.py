from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
MATRIX_PATH  = PROJECT_ROOT / "data" / "processed" / "feature_matrix.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "processed" / "feature_matrix_v2.csv"


def build_sos_features() -> pd.DataFrame:
    """Add strength-of-schedule (SOS) features to the feature matrix.

    A team's recent win rate or point differential in isolation doesn't
    distinguish between a team that dominated weak opponents and one that
    stayed competitive against strong ones.  SOS captures this by averaging
    the quality of a team's recent opponents, measured by each opponent's
    own rolling point differential at the time of the matchup.

    Leakage notes
    -------------
    opponent_quality for each game is the opponent's rolling_pt_diff_10
    value that was already in the wide table.  That value was itself built
    with .shift(1).rolling(10) in build_rolling_features.py, so it only
    reflects the opponent's form *before* the current game — already safe.

    The additional .shift(1) applied when computing sos_rolling_10 serves
    a separate purpose: it prevents the current game's opponent from being
    counted in this team's own SOS window.  Without it, game N's SOS would
    include game N's opponent, which is information about the current game
    rather than prior schedule context.

    Steps
    -----
    1. Load feature_matrix.csv.
    2. Reshape to long team-game format.  For each game, the home team row
       receives opponent_quality = rolling_pt_diff_10_away (the away team's
       pre-game rolling form), and the away team row receives the mirror.
    3. Sort by team + date; compute sos_rolling_10 as .shift(1).rolling(10)
       .mean() of opponent_quality, grouped by team.
    4. Pivot back to wide format with _home / _away suffixes; merge onto
       the main table by GAME_ID.
    5. Print null counts and plausibility check; save.

    Returns
    -------
    pd.DataFrame
        feature_matrix extended with sos_rolling_10_home and
        sos_rolling_10_away.
    """
    # ------------------------------------------------------------------ #
    # 1. Load                                                              #
    # ------------------------------------------------------------------ #
    print(f"Loading {MATRIX_PATH} ...")
    df = pd.read_csv(MATRIX_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    # ------------------------------------------------------------------ #
    # 2. Reshape to long team-game format                                  #
    # ------------------------------------------------------------------ #
    # opponent_quality is already leakage-safe — it's the opponent's
    # rolling_pt_diff_10 which was built with shift+rolling upstream.
    # We just need to pull it from the correct column for each side.
    home_view = df[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_home",
                    "rolling_pt_diff_10_away"]].copy()
    home_view.columns = ["GAME_ID", "GAME_DATE", "team", "opponent_quality"]
    home_view["is_home"] = 1

    away_view = df[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_away",
                    "rolling_pt_diff_10_home"]].copy()
    away_view.columns = ["GAME_ID", "GAME_DATE", "team", "opponent_quality"]
    away_view["is_home"] = 0

    long_df = pd.concat([home_view, away_view], ignore_index=True)
    long_df  = long_df.sort_values(["team", "GAME_DATE"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 3. Rolling SOS per team                                              #
    # ------------------------------------------------------------------ #
    # .shift(1) moves each value forward one row within the group, so that
    # game N's SOS window spans only the N-1 most recent opponents — the
    # current game's opponent is excluded.  This is necessary even though
    # opponent_quality itself is already leakage-safe: the shift here is
    # about excluding the current opponent from *this team's* schedule
    # context, not about the opponent's own internal calculation.
    g = long_df.groupby("team", sort=False)
    long_df["sos_rolling_10"] = g["opponent_quality"].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )

    # ------------------------------------------------------------------ #
    # 4. Pivot back and merge onto main table                              #
    # ------------------------------------------------------------------ #
    home_sos = (
        long_df[long_df["is_home"] == 1][["GAME_ID", "sos_rolling_10"]]
        .rename(columns={"sos_rolling_10": "sos_rolling_10_home"})
    )
    away_sos = (
        long_df[long_df["is_home"] == 0][["GAME_ID", "sos_rolling_10"]]
        .rename(columns={"sos_rolling_10": "sos_rolling_10_away"})
    )

    result = df.merge(home_sos, on="GAME_ID", how="left") \
               .merge(away_sos, on="GAME_ID", how="left")

    # ------------------------------------------------------------------ #
    # 5. Diagnostics and save                                              #
    # ------------------------------------------------------------------ #
    new_cols = ["sos_rolling_10_home", "sos_rolling_10_away"]
    for col in new_cols:
        n_null = result[col].isna().sum()
        print(f"\n  {col}: {n_null:,} nulls / {len(result):,} rows  ({100*n_null/len(result):.1f}%)")

    # SOS nulls are expected to exceed rolling_pt_diff_10 nulls (~150) because
    # two sources of missingness compound: (1) a team's first ~10 games lack
    # SOS history, and (2) any opponent whose own rolling_pt_diff_10 was null
    # (their own first 10 games) contributes a null to the window, which the
    # rolling mean propagates.  The combined effect produces more nulls but
    # they remain confined to the first portion of the 2020-21 season.
    print(f"\n  (Expected: more nulls than rolling_pt_diff_10 (~150 per side) because")
    print(f"   both teams need sufficient prior history for opponent_quality to be non-null.)")

    for col in new_cols:
        s = result[col].dropna()
        print(f"\n  {col} plausibility check (expected range ≈ −15 to +15, same scale as rolling_pt_diff_10):")
        print(f"    min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}  median={s.median():.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(result):,} rows × {result.shape[1]} columns to {OUT_PATH}")

    return result


if __name__ == "__main__":
    build_sos_features()

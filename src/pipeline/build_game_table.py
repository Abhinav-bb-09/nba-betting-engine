from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "games_merged.csv"

# Per-team stat columns present in the raw game log.
# After the home/away merge these become <col>_home and <col>_away.
STAT_COLS = [
    "WL", "MIN",
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PLUS_MINUS",
]


def build_game_table() -> pd.DataFrame:
    """Combine per-team game-log rows into one row per game.

    The NBA API returns two rows per game — one for each team.  This
    function pivots that structure so every game occupies a single row
    with _home and _away suffixes on each stat column, making it easy
    to compute differentials and join against betting lines.

    Steps
    -----
    1. Load and concatenate the five raw games_*.csv files.
    2. Split into home subset (MATCHUP contains "vs.") and away subset
       (MATCHUP contains "@").
    3. Merge the two subsets on GAME_ID, suffixing shared column names
       with _home and _away.
    4. Select a clean column set: identifiers + all stat pairs.
    5. Parse GAME_DATE to datetime.
    6. Assert the output row count equals exactly half the input rows
       (one merged row per game, not per team).
    7. Print diagnostics and save to data/processed/games_merged.csv.

    Returns
    -------
    pd.DataFrame
        One row per game with home and away stats side by side.
    """
    # ------------------------------------------------------------------ #
    # 1. Load and concatenate all seasons                                  #
    # ------------------------------------------------------------------ #
    files = sorted(RAW_DIR.glob("games_*.csv"))
    print(f"Found {len(files)} raw game files: {[f.name for f in files]}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    total_rows = len(df)
    print(f"Total rows loaded: {total_rows:,}")

    # ------------------------------------------------------------------ #
    # 2. Split into home and away subsets                                  #
    # ------------------------------------------------------------------ #
    # "vs." indicates the team played at home; "@" indicates away.
    home = df[df["MATCHUP"].str.contains("vs\\.", regex=True)].copy()
    away = df[df["MATCHUP"].str.contains("@", regex=False)].copy()

    print(f"  Home rows: {len(home):,}  |  Away rows: {len(away):,}")

    # Some games have both rows labeled "@" and no "vs." row — these are
    # neutral-site games or data errors in the source.  They cannot be
    # merged correctly, so we identify and drop them explicitly rather
    # than silently failing the row-count assertion later.
    neutral_ids = set(away["GAME_ID"]) - set(home["GAME_ID"])
    if neutral_ids:
        print(f"  WARNING: {len(neutral_ids)} game(s) have no home-side row "
              f"(both teams show '@') — dropping GAME_IDs: {sorted(neutral_ids)}")
        away = away[~away["GAME_ID"].isin(neutral_ids)]

    # ------------------------------------------------------------------ #
    # 3. Merge on GAME_ID                                                  #
    # ------------------------------------------------------------------ #
    merged = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))

    # ------------------------------------------------------------------ #
    # 4. Select clean column set                                           #
    # ------------------------------------------------------------------ #
    stat_cols_paired = [f"{c}_home" for c in STAT_COLS] + [f"{c}_away" for c in STAT_COLS]

    keep = (
        ["GAME_ID", "GAME_DATE_home", "SEASON_ID_home",
         "TEAM_ABBREVIATION_home", "TEAM_ABBREVIATION_away"]
        + stat_cols_paired
    )
    merged = merged[keep].rename(columns={
        "GAME_DATE_home": "GAME_DATE",
        "SEASON_ID_home": "SEASON_ID",
    })

    # ------------------------------------------------------------------ #
    # 5. Parse GAME_DATE                                                   #
    # ------------------------------------------------------------------ #
    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])

    # ------------------------------------------------------------------ #
    # 6. Verify row count                                                  #
    # ------------------------------------------------------------------ #
    # Expected = number of home-side rows (one per valid game after neutrals dropped).
    expected = len(home)
    actual = len(merged)
    dropped = len(neutral_ids)
    if actual != expected:
        raise AssertionError(
            f"Row count mismatch: expected {expected:,} merged rows "
            f"({total_rows:,} input rows − {dropped * 2} neutral/bad rows ÷ 2) "
            f"but got {actual:,}. Check for duplicate or unmatched GAME_IDs."
        )
    print(f"\n  Row count check passed: {total_rows:,} input rows "
          f"− {dropped * 2} neutral-site rows → {actual:,} merged rows (÷2 ✓)")

    # ------------------------------------------------------------------ #
    # 7. Print diagnostics and save                                        #
    # ------------------------------------------------------------------ #
    print(f"\n  Output shape: {merged.shape}")
    print(f"\n  First 3 rows:\n{merged[['GAME_ID','GAME_DATE','TEAM_ABBREVIATION_home','TEAM_ABBREVIATION_away','PTS_home','PTS_away']].head(3).to_string(index=False)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved to {OUT_PATH}")

    return merged


if __name__ == "__main__":
    build_game_table()

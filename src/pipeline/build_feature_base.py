from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
GAMES_PATH   = PROJECT_ROOT / "data" / "processed" / "games_merged.csv"
LINES_PATH   = PROJECT_ROOT / "data" / "processed" / "betting_lines.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "processed" / "feature_base.csv"


def build_feature_base() -> pd.DataFrame:
    """Join game results with betting lines and compute the spread-cover target.

    Steps
    -----
    1. Load games_merged.csv (one row per game, home and away stats side
       by side) and betting_lines.csv (spread, total, moneylines).
    2. Parse date columns to datetime on both so the join key is
       type-consistent.
    3. Left-join betting lines onto games on (date, home team, away team).
       A left join keeps every game row; unmatched games get NaN in the
       betting-line columns, which we can quantify before dropping.
    4. Report the match rate: how many games found a corresponding
       betting line.
    5. Drop rows with a null spread — we cannot construct a spread-cover
       target without a line, and including them would silently corrupt
       any model trained on this table.
    6. Compute home_covers_spread.

       Spread sign convention in this dataset
       ----------------------------------------
       The raw `spread` column is ALWAYS POSITIVE.  The `whos_favored`
       column carries the direction: "home" means the home team is giving
       points (they are the favourite), "away" means they are receiving
       them (they are the underdog).

       We convert to a signed spread from the home team's perspective:
           signed_spread = spread  if whos_favored == "home"  (home gives points)
           signed_spread = -spread if whos_favored == "away"  (home gets points)

       Home covers when their actual margin beats the spread they're giving:
           home_covers_spread = 1  if  (PTS_home - PTS_away) > signed_spread
                              = 0  otherwise

       Example: home -7.5 (signed_spread=+7.5) — home must win by >7.5.
       Example: home +3  (signed_spread=-3.0)  — home can lose by up to 3.

    7. Save to data/processed/feature_base.csv.
    8. Print final shape and target class distribution.

    Returns
    -------
    pd.DataFrame
        One row per matched game with all stat columns and home_covers_spread.
    """
    # ------------------------------------------------------------------ #
    # 1. Load                                                              #
    # ------------------------------------------------------------------ #
    print("Loading games_merged.csv ...")
    games = pd.read_csv(GAMES_PATH)
    print(f"  {len(games):,} game rows")

    print("Loading betting_lines.csv ...")
    lines = pd.read_csv(LINES_PATH)
    print(f"  {len(lines):,} betting-line rows")

    # ------------------------------------------------------------------ #
    # 2. Parse dates                                                       #
    # ------------------------------------------------------------------ #
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    lines["date"]      = pd.to_datetime(lines["date"])

    # ------------------------------------------------------------------ #
    # 3. Left-join betting lines onto games                                #
    # ------------------------------------------------------------------ #
    merged = games.merge(
        lines,
        left_on=["GAME_DATE", "TEAM_ABBREVIATION_home", "TEAM_ABBREVIATION_away"],
        right_on=["date", "home_team", "away_team"],
        how="left",
    ).drop(columns=["date", "home_team", "away_team"])

    # ------------------------------------------------------------------ #
    # 4. Report match rate                                                 #
    # ------------------------------------------------------------------ #
    n_matched   = merged["spread"].notna().sum()
    n_total     = len(merged)
    match_rate  = 100 * n_matched / n_total
    print(f"\n  Betting-line match rate: {n_matched:,} / {n_total:,} games ({match_rate:.1f}%)")

    # ------------------------------------------------------------------ #
    # 5. Drop unmatched rows (no spread → can't compute target)            #
    # ------------------------------------------------------------------ #
    n_dropped = n_total - n_matched
    if n_dropped:
        print(f"  Dropping {n_dropped:,} unmatched game(s) with null spread.")
    merged = merged[merged["spread"].notna()].copy()

    # ------------------------------------------------------------------ #
    # 6. Compute home_covers_spread                                        #
    # ------------------------------------------------------------------ #
    # Convert the always-positive spread to a signed value from the home
    # team's perspective (see docstring for the full sign convention).
    signed_spread = merged["spread"] * merged["whos_favored"].map({"home": 1, "away": -1})
    merged["home_covers_spread"] = (
        (merged["PTS_home"] - merged["PTS_away"]) > signed_spread
    ).astype(int)

    # ------------------------------------------------------------------ #
    # 7. Save                                                              #
    # ------------------------------------------------------------------ #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved to {OUT_PATH}")

    # ------------------------------------------------------------------ #
    # 8. Diagnostics                                                       #
    # ------------------------------------------------------------------ #
    print(f"\n  Final shape: {merged.shape}")
    print(f"\n  home_covers_spread value counts:")
    vc = merged["home_covers_spread"].value_counts().sort_index()
    for val, count in vc.items():
        label = "covers (1)" if val == 1 else "does not cover (0)"
        print(f"    {label}: {count:,}  ({100 * count / len(merged):.1f}%)")

    return merged


if __name__ == "__main__":
    build_feature_base()

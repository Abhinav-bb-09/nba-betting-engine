from pathlib import Path

import pandas as pd

PROJECT_ROOT  = Path(__file__).parents[2]
BASE_PATH     = PROJECT_ROOT / "data" / "processed" / "feature_base_v2.csv"
INJURIES_PATH = PROJECT_ROOT / "data" / "processed" / "injuries.csv"
OUT_PATH      = PROJECT_ROOT / "data" / "processed" / "feature_base_v3.csv"

# Inclusive bounds of the injury dataset's reliable coverage.
# Games outside this window must not be treated as "0 injuries" —
# the absence of records reflects missing data, not a clean slate.
INJURY_DATA_START = "2020-10-01"
INJURY_DATA_END   = "2023-04-16"


def build_injury_features() -> pd.DataFrame:
    """Add recent-injury context features to the feature base table.

    For each game we count how many distinct players from each team were
    placed on the IL in the 14 days strictly before the game date.  This
    gives the model a proxy for roster health without looking at same-day
    transactions (which would be leakage if the game hadn't yet been played).

    Because injury data only covers 2020-21 through 2022-23, we track
    coverage explicitly via has_injury_data and force the injury counts to
    NaN for out-of-coverage games.  Leaving those rows as 0 would imply
    healthy rosters when the real situation is simply unknown.

    Steps
    -----
    1. Load feature_base_v2.csv and injuries.csv.
    2. Filter injuries to placed_on_IL rows only (Relinquished is not null).
    3. Build all unique (team, game_date) pairs from both home and away sides.
    4. Cross-join those pairs against the injury records by team, then keep
       only injury events that fall in [game_date - 14 days, game_date).
       Count distinct Relinquished player names per (team, game_date).
    5. Merge counts back onto all pairs (left join), filling unmatched pairs
       with 0 — these are games with no IL placements in the 14-day window.
    6. Attach home and away counts to the main table.
    7. Add has_injury_data: 1 if game falls within coverage window, else 0.
    8. Explicitly override recent_injuries_count to NaN for out-of-coverage
       games — the 0s from step 5 are only valid within the coverage window.
    9. Print diagnostics and save.

    Returns
    -------
    pd.DataFrame
        feature_base_v2 extended with recent_injuries_count_home,
        recent_injuries_count_away, and has_injury_data.
    """
    # ------------------------------------------------------------------ #
    # 1. Load                                                              #
    # ------------------------------------------------------------------ #
    print(f"Loading {BASE_PATH} ...")
    base = pd.read_csv(BASE_PATH)
    base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
    print(f"  {len(base):,} rows, {base.shape[1]} columns")

    print(f"Loading {INJURIES_PATH} ...")
    injuries = pd.read_csv(INJURIES_PATH)
    injuries["Date"] = pd.to_datetime(injuries["Date"])
    print(f"  {len(injuries):,} injury transaction rows")

    # ------------------------------------------------------------------ #
    # 2. Keep only IL placements (Relinquished is the player being removed)#
    # ------------------------------------------------------------------ #
    placed_il = (
        injuries[injuries["status"] == "placed_on_IL"]
        [["Date", "team_abbr", "Relinquished"]]
        .copy()
    )
    print(f"  {len(placed_il):,} placed-on-IL transactions")

    # ------------------------------------------------------------------ #
    # 3. All unique (team, game_date) pairs across both sides              #
    # ------------------------------------------------------------------ #
    home_pairs = base[["GAME_DATE", "TEAM_ABBREVIATION_home"]].rename(
        columns={"TEAM_ABBREVIATION_home": "team"}
    )
    away_pairs = base[["GAME_DATE", "TEAM_ABBREVIATION_away"]].rename(
        columns={"TEAM_ABBREVIATION_away": "team"}
    )
    all_pairs = pd.concat([home_pairs, away_pairs]).drop_duplicates().reset_index(drop=True)
    print(f"  {len(all_pairs):,} unique (team, game_date) pairs")

    # ------------------------------------------------------------------ #
    # 4. Cross-join by team, filter to 14-day trailing window              #
    # ------------------------------------------------------------------ #
    # After the merge each row is one (game_date, team, injury_event).
    # We then keep only injury events that happened strictly before the
    # game and within 14 calendar days — the trailing window that captures
    # meaningful short-term roster disruption without including stale news.
    merged = all_pairs.merge(placed_il, left_on="team", right_on="team_abbr", how="left")

    in_window = merged[
        (merged["Date"] >= merged["GAME_DATE"] - pd.Timedelta(days=14)) &
        (merged["Date"] <  merged["GAME_DATE"])
    ].copy()

    counts = (
        in_window
        .groupby(["team", "GAME_DATE"])["Relinquished"]
        .nunique()
        .reset_index(name="recent_injuries_count")
    )

    # ------------------------------------------------------------------ #
    # 5. Merge counts back; fill 0 for pairs with no IL events in window  #
    # ------------------------------------------------------------------ #
    # A left join keeps every (team, game_date) pair.  Pairs not present
    # in `counts` had no IL placements in the window — they get NaN here
    # and are immediately filled to 0.  Note: this 0 is only meaningful
    # within the coverage period; the override in step 8 handles the rest.
    all_pairs_with_counts = all_pairs.merge(counts, on=["team", "GAME_DATE"], how="left")
    all_pairs_with_counts["recent_injuries_count"] = (
        all_pairs_with_counts["recent_injuries_count"].fillna(0).astype(int)
    )

    # ------------------------------------------------------------------ #
    # 6. Attach home and away counts to the main table                     #
    # ------------------------------------------------------------------ #
    home_counts = all_pairs_with_counts.rename(columns={
        "team": "TEAM_ABBREVIATION_home",
        "recent_injuries_count": "recent_injuries_count_home",
    })
    away_counts = all_pairs_with_counts.rename(columns={
        "team": "TEAM_ABBREVIATION_away",
        "recent_injuries_count": "recent_injuries_count_away",
    })

    result = base.merge(home_counts, on=["GAME_DATE", "TEAM_ABBREVIATION_home"], how="left")
    result = result.merge(away_counts, on=["GAME_DATE", "TEAM_ABBREVIATION_away"], how="left")

    # ------------------------------------------------------------------ #
    # 7. Coverage flag                                                     #
    # ------------------------------------------------------------------ #
    result["has_injury_data"] = (
        (result["GAME_DATE"] >= pd.Timestamp(INJURY_DATA_START)) &
        (result["GAME_DATE"] <= pd.Timestamp(INJURY_DATA_END))
    ).astype(int)

    # ------------------------------------------------------------------ #
    # 8. Override out-of-coverage injury counts to NaN                    #
    # ------------------------------------------------------------------ #
    # Games in 2023-24 and 2024-25 have no injury source data.  The
    # trailing-window calculation produced 0 for those rows because no
    # events matched — but 0 injuries and unknown injuries are not the
    # same thing.  Forcing NaN makes the gap explicit so that downstream
    # models either exclude these rows from injury-based predictions or
    # handle them as a distinct missing-data case.
    no_coverage = result["has_injury_data"] == 0
    result.loc[no_coverage, "recent_injuries_count_home"] = float("nan")
    result.loc[no_coverage, "recent_injuries_count_away"] = float("nan")

    # ------------------------------------------------------------------ #
    # 9. Diagnostics                                                       #
    # ------------------------------------------------------------------ #
    n_covered     = result["has_injury_data"].sum()
    n_not_covered = (result["has_injury_data"] == 0).sum()
    print(f"\n  has_injury_data breakdown:")
    print(f"    covered     (== 1): {n_covered:,} games  (2020-21 through 2022-23)")
    print(f"    not covered (== 0): {n_not_covered:,} games  (2023-24 and 2024-25)")

    covered = result[result["has_injury_data"] == 1]
    for side in ("home", "away"):
        col = f"recent_injuries_count_{side}"
        s = covered[col]
        print(f"\n  {col} distribution (covered rows):")
        print(f"    min={s.min():.0f}  max={s.max():.0f}  "
              f"mean={s.mean():.2f}  median={s.median():.0f}  "
              f"rows with >0: {(s > 0).sum():,} / {len(s):,}")

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(result):,} rows × {result.shape[1]} columns to {OUT_PATH}")

    return result


if __name__ == "__main__":
    build_injury_features()

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "NBA Player Injury Stats(1951 - 2023).csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "injuries.csv"

DATE_START = "2020-10-01"  # start of 2020-21 regular season bubble
DATE_END = "2023-04-16"    # end of 2022-23 regular season

# Maps the dataset's short team nicknames to standard NBA abbreviations.
# Confirmed against all 30 nicknames present in the filtered date window.
TEAM_NAME_MAP: dict[str, str] = {
    "76ers": "PHI",
    "Blazers": "POR",
    "Bucks": "MIL",
    "Bulls": "CHI",
    "Cavaliers": "CLE",
    "Celtics": "BOS",
    "Clippers": "LAC",
    "Grizzlies": "MEM",
    "Hawks": "ATL",
    "Heat": "MIA",
    "Hornets": "CHA",
    "Jazz": "UTA",
    "Kings": "SAC",
    "Knicks": "NYK",
    "Lakers": "LAL",
    "Magic": "ORL",
    "Mavericks": "DAL",
    "Nets": "BKN",
    "Nuggets": "DEN",
    "Pacers": "IND",
    "Pelicans": "NOP",
    "Pistons": "DET",
    "Raptors": "TOR",
    "Rockets": "HOU",
    "Spurs": "SAS",
    "Suns": "PHX",
    "Thunder": "OKC",
    "Timberwolves": "MIN",
    "Warriors": "GSW",
    "Wizards": "WAS",
}


def clean_injuries() -> pd.DataFrame:
    """Read, filter, and clean the NBA player injury history CSV.

    Steps
    -----
    1. Load the raw file and drop the spurious index column.
    2. Parse Date to datetime so the DataFrame can be joined to game logs
       and betting lines by date.
    3. Filter to the window that overlaps with our betting-lines dataset
       (2020-21 through 2022-23; moneyline seasons only).
    4. Map full team nicknames to standard 3-letter abbreviations so we
       can join against the other DataFrames.  Unknown names are printed.
    5. Derive a status column from the Relinquished / Acquired columns:
       "placed_on_IL" means the player was removed from the active roster;
       "activated" means they were returned.  Both in the same row is
       treated as placed_on_IL (edge-case guard).
    6. Print a diagnostic summary: row count, unique teams, unmapped teams,
       and null counts for key columns.
    7. Save to data/processed/ and return the cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned injury log with one row per roster transaction.
    """
    # ------------------------------------------------------------------ #
    # 1. Load and drop the unnamed index column                            #
    # ------------------------------------------------------------------ #
    print(f"Reading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=["Unnamed: 0"])
    print(f"  Raw rows: {len(df):,}")

    # ------------------------------------------------------------------ #
    # 2. Parse dates                                                       #
    # ------------------------------------------------------------------ #
    df["Date"] = pd.to_datetime(df["Date"])

    # ------------------------------------------------------------------ #
    # 3. Filter to target date window                                      #
    # ------------------------------------------------------------------ #
    mask = (df["Date"] >= DATE_START) & (df["Date"] <= DATE_END)
    df = df[mask].copy()
    print(f"  Rows after date filter ({DATE_START} → {DATE_END}): {len(df):,}")

    # ------------------------------------------------------------------ #
    # 4. Map team names to standard abbreviations                          #
    # ------------------------------------------------------------------ #
    unique_teams = set(df["Team"].dropna().unique())
    unmapped = unique_teams - set(TEAM_NAME_MAP)
    if unmapped:
        print(f"\n  WARNING — unmapped team names found: {sorted(unmapped)}")
        print("  These rows will have NaN in team_abbr.\n")
    else:
        print(f"  All {len(unique_teams)} team names mapped successfully.")

    df["team_abbr"] = df["Team"].map(TEAM_NAME_MAP)

    # ------------------------------------------------------------------ #
    # 5. Derive status column                                              #
    # ------------------------------------------------------------------ #
    # Relinquished = player removed from roster (placed on IL / inactive).
    # Acquired     = player returned to active roster.
    # When both are set (rare), we treat it as placed_on_IL.
    df["status"] = "unknown"
    df.loc[df["Acquired"].notna(), "status"] = "activated"
    df.loc[df["Relinquished"].notna(), "status"] = "placed_on_IL"

    # ------------------------------------------------------------------ #
    # 6. Diagnostic summary                                                #
    # ------------------------------------------------------------------ #
    print(f"\n  Unique teams found: {sorted(unique_teams)}")
    print(f"  Unmapped team count: {len(unmapped)}")

    key_cols = ["Date", "Team", "team_abbr", "Relinquished", "Acquired", "status"]
    print("\n  Null counts in key columns:")
    for col in key_cols:
        n_null = df[col].isna().sum()
        pct = 100 * n_null / len(df)
        print(f"    {col:<20} {n_null:>5,} nulls  ({pct:.1f}%)")

    # ------------------------------------------------------------------ #
    # 7. Save                                                              #
    # ------------------------------------------------------------------ #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(df):,} rows to {OUT_PATH}")

    return df


if __name__ == "__main__":
    clean_injuries()

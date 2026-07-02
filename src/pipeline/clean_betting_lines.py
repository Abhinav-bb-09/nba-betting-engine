from pathlib import Path

import pandas as pd

# Paths anchored to the project root regardless of working directory.
PROJECT_ROOT = Path(__file__).parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "nba_2008-2026.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "betting_lines.csv"

# Seasons labeled by ending year (2021 = the 2020-21 season, etc.)
TARGET_SEASONS = {2021, 2022, 2023, 2024, 2025}

# Full mapping from the dataset's lowercase codes to standard NBA abbreviations.
# Confirmed against all 30 codes present in the raw file.
TEAM_CODE_MAP: dict[str, str] = {
    "atl": "ATL",
    "bkn": "BKN",
    "bos": "BOS",
    "cha": "CHA",
    "chi": "CHI",
    "cle": "CLE",
    "dal": "DAL",
    "den": "DEN",
    "det": "DET",
    "gs": "GSW",
    "hou": "HOU",
    "ind": "IND",
    "lac": "LAC",
    "lal": "LAL",
    "mem": "MEM",
    "mia": "MIA",
    "mil": "MIL",
    "min": "MIN",
    "no": "NOP",
    "ny": "NYK",
    "okc": "OKC",
    "orl": "ORL",
    "phi": "PHI",
    "phx": "PHX",
    "por": "POR",
    "sa": "SAS",
    "sac": "SAC",
    "tor": "TOR",
    "utah": "UTA",
    "wsh": "WAS",
}


def clean_betting_lines() -> pd.DataFrame:
    """Read, filter, and clean the raw NBA betting lines CSV.

    Steps
    -----
    1. Load the raw file.
    2. Filter to the five seasons that overlap with our game-log data
       (2020-21 through 2024-25, labeled 2021-2025 in this dataset).
    3. Parse the date column to proper datetime so downstream code can
       join on it or resample by date.
    4. Map the dataset's shorthand team codes to standard 3-letter NBA
       abbreviations so we can join against the game-log DataFrames that
       use official abbreviations.  Any code not in the map is printed so
       gaps can be caught early.
    5. Report null counts for the columns that feed directly into the
       model (spread, total, moneyline_home, moneyline_away).
    6. Save the cleaned DataFrame to data/processed/ and return it.

    Returns
    -------
    pd.DataFrame
        The cleaned betting-lines DataFrame.
    """
    # ------------------------------------------------------------------ #
    # 1. Load raw data                                                     #
    # ------------------------------------------------------------------ #
    print(f"Reading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Raw rows: {len(df):,}")

    # ------------------------------------------------------------------ #
    # 2. Filter to target seasons                                          #
    # ------------------------------------------------------------------ #
    df = df[df["season"].isin(TARGET_SEASONS)].copy()
    print(f"  Rows after season filter ({sorted(TARGET_SEASONS)}): {len(df):,}")

    # ------------------------------------------------------------------ #
    # 3. Parse date column                                                 #
    # ------------------------------------------------------------------ #
    df["date"] = pd.to_datetime(df["date"])

    # ------------------------------------------------------------------ #
    # 4. Map team codes to standard NBA abbreviations                      #
    # ------------------------------------------------------------------ #
    all_codes = set(df["away"].unique()) | set(df["home"].unique())
    unmapped = all_codes - set(TEAM_CODE_MAP)
    if unmapped:
        print(f"\n  WARNING — unmapped team codes found: {sorted(unmapped)}")
        print("  These rows will have NaN in away_team / home_team.\n")
    else:
        print("  All team codes mapped successfully.")

    df["away_team"] = df["away"].map(TEAM_CODE_MAP)
    df["home_team"] = df["home"].map(TEAM_CODE_MAP)

    # ------------------------------------------------------------------ #
    # 5. Report nulls in betting-line columns                              #
    # ------------------------------------------------------------------ #
    betting_cols = ["spread", "total", "moneyline_home", "moneyline_away"]
    print("\n  Null counts in betting-line columns:")
    for col in betting_cols:
        n_null = df[col].isna().sum()
        pct = 100 * n_null / len(df)
        print(f"    {col:<20} {n_null:>6,} nulls  ({pct:.1f}%)")

    # ------------------------------------------------------------------ #
    # 6. Save to processed/                                                #
    # ------------------------------------------------------------------ #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved {len(df):,} rows to {OUT_PATH}")

    return df


if __name__ == "__main__":
    clean_betting_lines()

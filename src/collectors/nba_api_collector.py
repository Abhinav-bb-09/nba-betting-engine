import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats

from src.utils.base_collector import BaseCollector


class NBAAPICollector(BaseCollector):
    """Collects raw game and team stats from the NBA Stats API.

    All network calls go through fetch_with_retry so transient failures
    (rate-limit responses, timeouts) are handled automatically.
    """

    def fetch_season_games(self, season: str) -> pd.DataFrame:
        """Fetch every regular-season game log for a given season.

        Uses LeagueGameLog which returns one row per team per game,
        giving us scores, home/away flag, and basic box-score totals.
        This is the foundation for win/loss records and point spreads.

        Args:
            season: NBA season string in the format "YYYY-YY", e.g. "2023-24".

        Returns:
            DataFrame with one row per team-game entry.
        """
        self.logger.info("Fetching game log for season %s", season)

        def _fetch():
            log = leaguegamelog.LeagueGameLog(
                season=season,
                timeout=self.config["nba_api"]["timeout_seconds"],
            )
            return log.get_data_frames()[0]

        return self.fetch_with_retry(
            _fetch,
            max_retries=self.config["nba_api"]["max_retries"],
            delay_seconds=self.config["nba_api"]["rate_limit_seconds"],
        )

    def fetch_advanced_stats(self, season: str) -> pd.DataFrame:
        """Fetch per-team advanced stats for a given season.

        Uses LeagueDashTeamStats with measure_type='Advanced' to pull
        Offensive Rating (ORtg), Defensive Rating (DRtg), and True
        Shooting % (TS%). These are the strongest predictors of team
        quality and are used downstream as model features.

        Args:
            season: NBA season string in the format "YYYY-YY", e.g. "2023-24".

        Returns:
            DataFrame with one row per team containing advanced metrics.
        """
        self.logger.info("Fetching advanced stats for season %s", season)

        def _fetch():
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                timeout=self.config["nba_api"]["timeout_seconds"],
            )
            return stats.get_data_frames()[0]

        return self.fetch_with_retry(
            _fetch,
            max_retries=self.config["nba_api"]["max_retries"],
            delay_seconds=self.config["nba_api"]["rate_limit_seconds"],
        )

    def save_raw_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save a DataFrame to data/raw/ as a CSV file.

        Saving raw data before any transformation means we can re-run
        the processing pipeline without hitting the API again.

        Args:
            df: The DataFrame to persist.
            filename: Destination filename, e.g. "games_2023-24.csv".

        Returns:
            The Path where the file was written.
        """
        raw_dir = Path(__file__).parents[2] / self.config["paths"]["raw_data_dir"]
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / filename
        df.to_csv(dest, index=False)
        self.logger.info("Saved %d rows to %s", len(df), dest)
        return dest

    def run(self, seasons: list[str]) -> None:
        """Collect and save game logs and advanced stats for each season.

        Iterates over the provided seasons, fetches both data types for
        each, and saves them as CSVs in data/raw/. A rate-limit pause
        is inserted between every API call to avoid getting blocked by
        the NBA Stats API.

        Args:
            seasons: List of season strings, e.g. ["2022-23", "2023-24"].
        """
        rate_pause = self.config["nba_api"]["rate_limit_seconds"]

        for season in seasons:
            self.logger.info("--- Starting collection for season %s ---", season)

            games_df = self.fetch_season_games(season)
            self.save_raw_data(games_df, f"games_{season}.csv")

            time.sleep(rate_pause)

            advanced_df = self.fetch_advanced_stats(season)
            self.save_raw_data(advanced_df, f"advanced_stats_{season}.csv")

            time.sleep(rate_pause)

        self.logger.info("Collection complete for %d season(s).", len(seasons))

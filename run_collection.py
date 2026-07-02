from src.collectors.nba_api_collector import NBAAPICollector

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

if __name__ == "__main__":
    collector = NBAAPICollector()
    collector.run(SEASONS)

# NBA Betting Engine

A data pipeline and modeling engine for NBA betting analysis.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
data/           Raw, processed, and feature data
src/
  collectors/   Data ingestion from external APIs / sources
  pipeline/     Feature engineering and model pipeline
  utils/        Shared helpers
config/         YAML configuration
tests/          Unit and integration tests
logs/           Runtime logs
```

## Known Data Limitations

The betting lines dataset (`data/processed/betting_lines.csv`) has complete spread and total data across all 5 seasons (2020-21 through 2024-25).

Moneyline data is only reliably available for 2020-21 and 2021-22, with ~50% coverage in 2022-23, and no coverage in 2023-24 or 2024-25.

**Decision:** spread and total predictions use the full 5-season dataset. Moneyline prediction is scoped to 2020-21 through 2022-23 only, and treated as a secondary model.

Injury data (`data/processed/injuries.csv`) is sourced from a pre-scraped dataset (Pro Sports Transactions, via Kaggle) due to Cloudflare protection blocking direct scraping of the original site.

Coverage: 2020-21 through most of 2022-23 season only (through April 16, 2023). No injury data exists for 2023-24 or 2024-25.

**Decision:** injury-based features will only be available for the seasons with coverage; this will need to be accounted for explicitly in Phase 2 feature engineering (e.g. flagging rows with no injury data available, rather than treating missing injury data as "no injuries occurred").

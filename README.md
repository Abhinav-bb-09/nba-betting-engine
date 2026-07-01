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

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

## Feature Ablation Study

Tested 3 feature set variants via 3-fold chronological cross-validation (folds: 2021→2022, 2021-22→2023, 2021-23→2024; ~1,000–1,300 games per validation fold):

| Feature set | Best mean ROC-AUC |
|---|---|
| Baseline (rolling team-form + efficiency) | 0.5164 |
| Baseline + strength-of-schedule (SOS) | 0.5074 |
| Baseline + SOS + rest differential + current streak | 0.5105 |

All differences fall within the expected noise band at this sample size — none represent a statistically meaningful improvement over the baseline.

**Decision:** per Occam's razor, proceeded with the baseline feature set (`models/baseline_xgb.json`, retrained via `tune_model_cv.py` on `feature_matrix.csv`) for Phase 4 backtesting. The SOS, rest, and streak pipeline code is retained for future reference but the additional complexity is not justified by the CV evidence.

## Backtest Results (2024-25 Held-Out Test Season)

First-ever evaluation on the 2024-25 test season (never used in feature selection, hyperparameter tuning, or the ablation study):

| Metric | Value |
|---|---|
| Model accuracy | 48.73% |
| ROC-AUC | 0.4793 |
| Majority-class baseline accuracy | 50.12% |

Flat-betting simulation at standard -110 odds (breakeven win rate: 52.38%):

| Strategy | Win rate | ROI |
|---|---|---|
| Model (baseline_xgb_tuned) | 48.73% | −6.96% |
| Always bet home covers | 49.88% | −4.78% |
| Always bet away covers | 50.12% | −4.31% |

All three strategies cluster within a ~1.4 percentage-point band around 50%, consistent with sampling noise rather than a meaningful difference between them. The model is the weakest of the three — it learned patterns from 2021–2024 that partially inverted in 2024-25.

**Conclusion:** this project did not find a profitable, statistically robust edge against NBA closing lines using the tested feature set (rolling team form, efficiency ratings, injury counts, and 3 ablated features — SOS, rest differential, streak). This result is consistent with prior research establishing NBA betting markets as highly efficient, and represents an honest, rigorously-obtained empirical finding rather than a pipeline failure.

**Note on bet sizing:** Kelly criterion sizing was deliberately not used. Kelly requires a demonstrated positive expected value to size bets safely; applying it to unproven or negative edge estimates amplifies losses rather than growth. Flat betting at a fixed unit was used instead as the methodologically honest choice given the model's unproven edge.

## Docker Deployment

The Docker image requires two locally-generated artifacts that are gitignored and not included in the repository:

- `models/baseline_xgb_tuned.json` — the trained XGBoost model
- `data/processed/test.csv` — the 2024-25 test split used by `GET /demo/{game_id}`

If cloning fresh, run the full pipeline first (`src/pipeline/` scripts in order, then `src/pipeline/train_baseline_model.py` and `src/pipeline/tune_model_cv.py`) to regenerate these files before building the image.

**Build:**
```bash
docker build -t nba-betting-engine .
```

**Run:**
```bash
docker run -p 8000:8000 nba-betting-engine
```

The API is then available at `http://localhost:8000`. Visit `/docs` for the interactive Swagger UI and `/redoc` for the full reference.

**Verified build and run:** the image was built end-to-end in ~126 s (14/14 steps). The `libgomp1` apt step resolves XGBoost's OpenMP dependency inside the container — the same class of issue hit during Phase 3 local development on macOS (resolved then via `brew install libomp`). A live container was confirmed against `/health` and `/demo/{game_id}`, returning identical responses to the local dev server; the model path resolved correctly to `/app/models/baseline_xgb_tuned.json`, confirming filesystem isolation.

**Production note:** baking model weights and data artifacts directly into the image is appropriate for a demonstration project but does not scale to production. A production deployment would pull versioned artifacts from cloud storage (S3, GCS) or a model registry (e.g. MLflow, Weights & Biases) at container startup — keeping the image itself stateless and the model version independently auditable. This is consistent with the live-feature-pipeline limitation already documented for `GET /predict`: both require production data infrastructure that is intentionally out of scope for this phase.

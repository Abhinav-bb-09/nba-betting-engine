# NBA Spread Prediction — Model Card

Predicts whether the NBA home team covers the closing point spread.
→ Full documentation: [README.md](README.md)

---

## Task

**Binary classification.** Label = 1 if the home team's final margin exceeds the closing spread, 0 otherwise. Evaluated against Vegas closing lines — the strongest publicly available baseline for NBA spread markets.

---

## Data

| Source | Seasons | Rows | Notes |
|---|---|---|---|
| NBA API game logs (`nba_api`) | 2020-21 → 2024-25 | 5,993 games | Neutral-site games dropped (5 detected, removed) |
| Betting lines (Kaggle) | 2020-21 → 2024-25 | — | Spread + total complete; moneyline absent for 2023-25 |
| Injury reports (Pro Sports Transactions via Kaggle) | 2020-21 → 2022-23 | — | Cloudflare blocks live scraping; 2023-25 seasons have no injury data |

**Split:** train 2020-21 → 2023-24 (4,768 rows), test 2024-25 (1,225 rows, fully held out until final evaluation).

**Known limitations:** moneyline features excluded (insufficient coverage); injury features are NaN for ~40% of rows and flagged via `has_injury_data`; XGBoost handles both natively.

---

## Model

| Attribute | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| `max_depth` | 4 |
| `learning_rate` | 0.1 |
| `n_estimators` | 100 |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |
| Features | 29 across 5 groups (see below) |
| Training data | Seasons 2020-21 → 2023-24 |
| Held-out test | Season 2024-25 |
| Artifact | `models/baseline_xgb_tuned.json` |

**Feature groups (29 total, home + away for all rolling features):**

| Group | Features | Count |
|---|---|---|
| Rolling win/loss form | `rolling_win_pct_5/10`, `rolling_pt_diff_5/10` | 8 |
| Rest & fatigue | `days_rest`, `is_back_to_back` | 4 |
| Rolling efficiency | `rolling_ortg_5/10`, `rolling_drtg_5/10`, `rolling_ts_pct_5/10` | 12 |
| Injury context | `recent_injuries_count_home/away`, `has_injury_data` | 3 |
| Betting line inputs | `spread`, `total` | 2 |

---

## Methodology Highlights

- **Leakage-safe rolling features** — `.shift(1)` applied before every `.rolling()` call so game N's window never includes game N's own outcome; enforced via `leakage_safe_rolling_mean()` helper and unit-tested in `tests/test_leakage_prevention.py`
- **Chronological cross-validation** — 3 expanding-window folds (2021→2022, 2021-22→2023, 2021-23→2024); never random k-fold, which would allow future games to appear in training sets
- **Feature ablation study** — SOS, rest differential, and streak features built and tested separately before the final feature set was locked
- **Held-out test season** — the 2024-25 season was never used in feature selection, hyperparameter tuning, or the ablation study; first look at test results came after all modelling decisions were finalised

---

## Results

| Metric | Validation (CV mean) | Test (2024-25, held out) |
|---|---|---|
| Accuracy | — | 48.73% |
| ROC-AUC | 0.5164 | 0.4793 |
| Majority-class baseline | — | 50.12% |

---

## Feature Ablation Study

| Feature set | Mean CV ROC-AUC |
|---|---|
| Baseline (rolling form + efficiency) | 0.5164 |
| + Strength of schedule | 0.5074 |
| + SOS + rest differential + streak | 0.5105 |

All differences fall within the expected noise band at this sample size. The baseline feature set was selected per Occam's razor.

---

## Backtest (Flat Betting, −110 Odds, 2024-25 Season)

$100 per game, 1,225 games. Break-even win rate at −110 odds: **52.38%**.

| Strategy | Win Rate | ROI |
|---|---|---|
| Model (XGBoost) | 48.73% | −6.96% |
| Always bet home covers | 49.88% | −4.78% |
| Always bet away covers | 50.12% | −4.31% |
| **Break-even required** | **52.38%** | **0%** |

All three strategies cluster within ≈1.4 percentage points of each other — consistent with sampling noise rather than a meaningful signal difference between them.

---

## Conclusion

No statistically significant edge was found over Vegas closing lines using the tested feature set. This is consistent with established research on NBA betting market efficiency: closing lines rapidly incorporate public information, leaving little exploitable signal in the rolling team-form and efficiency statistics available post-game. Feature importance is nearly uniformly distributed across all 29 inputs (top-15 importance scores span only 0.038–0.042, with `spread` ranking 9th and `total` outside the top 15 entirely), confirming that the model found no single feature or shortcut to lean on — a further indicator of genuinely weak predictive signal rather than a modelling or pipeline failure.

---

## Engineering Artifacts

- **Inference API** — [`src/api/`](src/api/) (FastAPI, `/predict` and `/demo/{game_id}` endpoints)
- **Containerisation** — [`Dockerfile`](Dockerfile) (python:3.13-slim, libgomp1 fix for XGBoost)
- **Unit tests** — [`tests/`](tests/) (leakage prevention + streak calculation, pytest)
- **CI** — [`.github/workflows/tests.yml`](.github/workflows/tests.yml) (runs on every push/PR to main)
- **EDA notebook** — [`notebooks/eda_and_results.ipynb`](notebooks/eda_and_results.ipynb) (cover rates, ORtg distribution, feature importance, backtest ROI — outputs embedded)

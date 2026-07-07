"""
NBA Spread Prediction API

Serves the baseline XGBoost spread-cover model (Phase 3) via a FastAPI REST
interface.  All endpoints are read-only; the model and test dataset are loaded
once at startup and held in memory for the lifetime of the server process.

Scope and honest limitations
------------------------------
This API serves *pre-computed* features extracted from historical game logs.
Generating features for an *upcoming* game would require:

  1. A live NBA stats pipeline polling recent box scores for both teams
     (the same rolling-window logic in build_rolling_features.py and
     build_efficiency_features.py, but applied in real time).
  2. A live betting-line feed for the spread and total inputs.
  3. Injury feed coverage for 2023-24 and 2024-25 seasons (currently absent).

That real-time feature pipeline is intentionally out of scope for this phase.
The /demo endpoint sidesteps the limitation by replaying held-out test games
whose features are already stored in data/processed/test.csv.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[2]
MODEL_PATH   = PROJECT_ROOT / "models" / "baseline_xgb_tuned.json"
TEST_PATH    = PROJECT_ROOT / "data" / "processed" / "test.csv"

DISCLAIMER = (
    "This model showed no statistically significant edge over Vegas closing "
    "lines in backtesting on the held-out 2024-25 season (ROC-AUC 0.479, "
    "accuracy 48.73% vs. 52.38% required breakeven at −110 odds). "
    "This API is a demonstration of an ML deployment pipeline, "
    "not a betting recommendation."
)

# The 29 v1 feature columns the model was actually trained on.
#
# The current feature whitelist (define_model_features.py) contains 34
# non-identifier columns, but train.csv and test.csv were built from
# feature_matrix.csv (v1), which pre-dates the SOS, streak, and
# rest_advantage ablation columns.  The same version-agnostic intersection
# used in backtest_evaluate.py reduces the effective column set to these 29.
# Column order here must match training order (XGBoost uses it for alignment).
MODEL_COLS = [
    # Home rolling form (last 5 / 10 games, shifted — no leakage)
    "rolling_win_pct_5_home",    "rolling_win_pct_10_home",
    "rolling_pt_diff_5_home",    "rolling_pt_diff_10_home",
    "days_rest_home",            "is_back_to_back_home",
    # Home rolling efficiency (Dean Oliver ORtg / DRtg / TS%)
    "rolling_ortg_5_home",       "rolling_ortg_10_home",
    "rolling_drtg_5_home",       "rolling_drtg_10_home",
    "rolling_ts_pct_5_home",     "rolling_ts_pct_10_home",
    # Away rolling form
    "rolling_win_pct_5_away",    "rolling_win_pct_10_away",
    "rolling_pt_diff_5_away",    "rolling_pt_diff_10_away",
    "days_rest_away",            "is_back_to_back_away",
    # Away rolling efficiency
    "rolling_ortg_5_away",       "rolling_ortg_10_away",
    "rolling_drtg_5_away",       "rolling_drtg_10_away",
    "rolling_ts_pct_5_away",     "rolling_ts_pct_10_away",
    # Injury context — NaN for 2023-24 / 2024-25 (XGBoost handles natively)
    "recent_injuries_count_home", "recent_injuries_count_away", "has_injury_data",
    # Pre-game betting line inputs (from the sportsbook, known before tip-off)
    "spread", "total",
]

# ---------------------------------------------------------------------------
# Runtime state — populated once at startup, never mutated during serving.
#
# Loading the model per-request would re-read and deserialise the JSON file
# on every call (~50–100 ms each), making the API impractical under any load.
# Holding the deserialised XGBClassifier in module-level state costs ~20 MB
# of RAM and makes inference effectively instant (<1 ms per row).
# ---------------------------------------------------------------------------
_state: dict = {
    "model":        None,
    "model_loaded": False,
    "test_df":      None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the XGBoost model and test dataset once on startup."""
    # Model
    try:
        mdl = xgb.XGBClassifier()
        mdl.load_model(str(MODEL_PATH))
        _state["model"]        = mdl
        _state["model_loaded"] = True
        print(f"[startup] Model loaded: {MODEL_PATH.name}")
    except Exception as exc:
        print(f"[startup] WARNING: model failed to load — {exc}")

    # Test dataset (used only by /demo — not on the hot prediction path)
    try:
        _state["test_df"] = pd.read_csv(TEST_PATH)
        print(f"[startup] Test data loaded: {len(_state['test_df']):,} rows")
    except Exception as exc:
        print(f"[startup] WARNING: test dataset unavailable — {exc}")

    yield  # serve requests until shutdown


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NBA Spread Prediction API",
    description=(
        "Baseline XGBoost model for NBA spread-cover prediction. "
        "Built as a rigorous end-to-end ML pipeline demonstration — see /docs. "
        "<br><br>"
        "<strong>Disclaimer:</strong> " + DISCLAIMER
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GameFeatures(BaseModel):
    """All 29 pre-game feature inputs for the XGBoost spread-cover model.

    Every value must reflect the state BEFORE tip-off.  Passing any box-score
    outcome column (points scored, FGA, WL, etc.) would be data leakage — the
    model would be "predicting" from the final score it isn't allowed to see.

    Rolling values use a shift(1) + rolling(N) convention: the 'last 5 games'
    window excludes the current game and the N games immediately preceding it.

    Injury fields are Optional[float] because the source dataset only covers
    through April 2023.  Omitting them (or passing null) is correct for more
    recent games; XGBoost handles NaN internally via its split-finding logic.
    """
    # ── Home team rolling form ─────────────────────────────────────────────
    rolling_win_pct_5_home:   float
    rolling_win_pct_10_home:  float
    rolling_pt_diff_5_home:   float
    rolling_pt_diff_10_home:  float
    days_rest_home:           float
    is_back_to_back_home:     float

    # ── Home team rolling efficiency (Dean Oliver ORtg / DRtg / TS%) ───────
    rolling_ortg_5_home:      float
    rolling_ortg_10_home:     float
    rolling_drtg_5_home:      float
    rolling_drtg_10_home:     float
    rolling_ts_pct_5_home:    float
    rolling_ts_pct_10_home:   float

    # ── Away team rolling form ─────────────────────────────────────────────
    rolling_win_pct_5_away:   float
    rolling_win_pct_10_away:  float
    rolling_pt_diff_5_away:   float
    rolling_pt_diff_10_away:  float
    days_rest_away:           float
    is_back_to_back_away:     float

    # ── Away team rolling efficiency ───────────────────────────────────────
    rolling_ortg_5_away:      float
    rolling_ortg_10_away:     float
    rolling_drtg_5_away:      float
    rolling_drtg_10_away:     float
    rolling_ts_pct_5_away:    float
    rolling_ts_pct_10_away:   float

    # ── Injury context ─────────────────────────────────────────────────────
    recent_injuries_count_home: Optional[float] = None
    recent_injuries_count_away: Optional[float] = None
    has_injury_data:            Optional[float] = None

    # ── Pre-game betting line inputs ───────────────────────────────────────
    spread: float
    total:  float


class PredictionResponse(BaseModel):
    home_covers_spread_probability: float
    predicted_class:                int
    disclaimer:                     str


# ---------------------------------------------------------------------------
# Shared inference helper
# ---------------------------------------------------------------------------

def _predict(features_dict: dict) -> tuple[float, int]:
    """Run model inference on a dict of feature values.

    Parameters
    ----------
    features_dict : dict
        Keys must cover MODEL_COLS; None values are treated as NaN by pandas
        and handled natively by XGBoost's split logic.

    Returns
    -------
    tuple[float, int]
        (probability that home covers spread, binary predicted class)

    Raises
    ------
    HTTPException 503
        If the model did not load successfully at startup.
    """
    if not _state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health for details.",
        )
    # Cast to float64 so None values become NaN — XGBoost rejects object dtype.
    row  = pd.DataFrame([features_dict])[MODEL_COLS].astype("float64")
    prob = float(_state["model"].predict_proba(row)[0, 1])
    cls  = int(_state["model"].predict(row)[0])
    return prob, cls


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="API info and navigation")
def root():
    """Return basic API metadata and links to available endpoints."""
    return {
        "name":    "NBA Spread Prediction API",
        "version": "1.0.0",
        "model":   "baseline_xgb_tuned — XGBoost depth=4 lr=0.1 n=100, trained on 2021–2024 NBA seasons",
        "endpoints": {
            "GET  /health":          "Model load status",
            "POST /predict":         "Predict spread cover from 29 feature inputs",
            "GET  /demo/{game_id}":  "Replay a held-out 2024-25 test game by GAME_ID",
            "GET  /docs":            "Interactive Swagger UI",
            "GET  /redoc":           "ReDoc API reference",
        },
        "disclaimer": DISCLAIMER,
    }


@app.get("/health", summary="Model health check")
def health():
    """Return API status and whether the model loaded successfully at startup.

    model_loaded is False if the model file was missing or corrupt at boot.
    In that state /predict and /demo will return 503.
    """
    return {
        "status":       "ok",
        "model_loaded": _state["model_loaded"],
        "model_path":   str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict spread cover")
def predict(features: GameFeatures):
    """Predict whether the home team will cover the spread.

    Accepts the 29 pre-game feature values defined in GameFeatures and returns
    the model's probability estimate alongside the binary predicted class.

    - **predicted_class = 1**: model predicts home team covers the spread.
    - **predicted_class = 0**: model predicts home team does NOT cover.

    Injury fields (recent_injuries_count_home/away, has_injury_data) may be
    omitted or set to null for games where injury data is unavailable;
    XGBoost handles NaN natively.
    """
    prob, cls = _predict(features.model_dump())
    return PredictionResponse(
        home_covers_spread_probability=round(prob, 4),
        predicted_class=cls,
        disclaimer=DISCLAIMER,
    )


@app.get("/demo/{game_id}", summary="Replay a held-out test game")
def demo(game_id: int):
    """Look up a 2024-25 test game by GAME_ID and return prediction vs. reality.

    Retrieves pre-computed features for the given game from
    data/processed/test.csv, runs model inference, and returns both the
    prediction and the game's actual real-world outcome so the caller can
    compare directly.

    Game IDs for the 2024-25 season are 8-digit integers in the range
    22400001–22401230 (NBA API format with leading zero stripped by pandas).
    Browse test.csv or query the NBA API for specific GAME_IDs.

    Returns 404 if the GAME_ID is not found in the 2024-25 test set.
    Returns 503 if the test dataset or model failed to load at startup.
    """
    test_df = _state["test_df"]
    if test_df is None:
        raise HTTPException(
            status_code=503,
            detail="Test dataset is not available. Check server logs.",
        )

    match = test_df[test_df["GAME_ID"] == game_id]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"GAME_ID {game_id} not found in the 2024-25 test set "
                f"({len(test_df):,} games). "
                "IDs are 8-digit integers, e.g. 22400062."
            ),
        )

    row = match.iloc[0]

    # Build feature dict; NaN → None so pandas recreates NaN in the DataFrame.
    features_dict = {
        col: (float(row[col]) if pd.notna(row[col]) else None)
        for col in MODEL_COLS
    }
    prob, cls = _predict(features_dict)

    actual = int(row["home_covers_spread"])
    return {
        "game": {
            "game_id":   int(row["GAME_ID"]),
            "game_date": str(row["GAME_DATE"]),
            "home_team": str(row["TEAM_ABBREVIATION_home"]),
            "away_team": str(row["TEAM_ABBREVIATION_away"]),
            "spread":    float(row["spread"]),
            "actual_outcome": {
                "home_covers_spread": actual,
                "description": (
                    "Home team covered the spread"
                    if actual == 1
                    else "Home team did not cover the spread"
                ),
            },
        },
        "prediction": {
            "home_covers_spread_probability": round(prob, 4),
            "predicted_class":                cls,
            "correct":                        int(cls == actual),
            "disclaimer":                     DISCLAIMER,
        },
    }

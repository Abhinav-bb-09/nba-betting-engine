from pathlib import Path

import pandas as pd

PROJECT_ROOT  = Path(__file__).parents[2]
PREDS_PATH    = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"
OUT_PATH      = PROJECT_ROOT / "data" / "processed" / "backtest_summary.csv"

BET_SIZE          = 100.0   # flat $100 per game
WIN_PROFIT        = 90.91   # profit on a winning -110 bet ($100 / 1.10, rounded)
LOSS_AMOUNT       = -100.0  # loss on a losing bet
BREAKEVEN_WIN_RATE = 100 / (100 + WIN_PROFIT)  # ≈ 52.38%


def _simulate(predictions: pd.Series, actuals: pd.Series, label: str) -> dict:
    """Run a flat-betting simulation for one prediction strategy.

    Args:
        predictions: Series of 0/1 predicted class for each game.
        actuals:     Series of 0/1 actual home_covers_spread for each game.
        label:       Strategy name for reporting.

    Returns:
        Dictionary of summary statistics for this strategy.
    """
    correct = (predictions == actuals)
    wins    = int(correct.sum())
    losses  = int((~correct).sum())
    games   = wins + losses
    profit  = wins * WIN_PROFIT + losses * LOSS_AMOUNT
    roi     = profit / (games * BET_SIZE) * 100

    return {
        "strategy":         label,
        "games_bet":        games,
        "wins":             wins,
        "losses":           losses,
        "win_rate_pct":     round(100 * wins / games, 2),
        "total_profit":     round(profit, 2),
        "roi_pct":          round(roi, 2),
        "breakeven_win_rate_pct": round(BREAKEVEN_WIN_RATE * 100, 2),
    }


def simulate_flat_betting() -> pd.DataFrame:
    """Simulate flat-bet wagering on the 2024-25 NBA test season.

    Betting convention: -110 odds (standard US spread market)
    -------------------------------------------------------
    A -110 line means you must risk $110 to win $100.  On a flat $100 bet:
      - Win: profit = $100 / 1.10 ≈ +$90.91
      - Loss: profit = -$100.00
    The breakeven win rate is 100 / (100 + 90.91) ≈ 52.38%.  Any model
    that wins fewer than 52.38% of its bets will lose money in the long run
    regardless of bet sizing, because the house rake is baked into the line.

    Why flat betting (not Kelly)
    ----------------------------
    Kelly criterion maximises long-run bankroll growth by sizing bets in
    proportion to perceived edge.  It requires an *accurate* edge estimate —
    i.e. calibrated probabilities and demonstrated positive expectation.
    This model has not shown a proven edge on held-out data (ROC-AUC < 0.5
    on the test set).  Applying Kelly to unproven edge estimates amplifies
    losses, not gains.  Flat betting at a fixed unit size is the honest
    baseline: it makes the P&L purely a function of prediction accuracy,
    with no illusion of optimal sizing.

    Strategies simulated
    --------------------
    1. Model bets: wager on predicted_class for every game.
    2. Always-home baseline: always bet that the home team covers (class 1).
       Represents the naive bettor who ignores all information and backs
       the home side on every game.
    3. Always-away baseline: always bet that the home team does NOT cover
       (class 0).  Mirror of strategy 2.

    Returns
    -------
    pd.DataFrame
        One row per strategy with win rate, P&L, and ROI.
    """
    # ------------------------------------------------------------------ #
    # Load predictions                                                     #
    # ------------------------------------------------------------------ #
    print(f"Loading {PREDS_PATH.name} ...")
    df = pd.read_csv(PREDS_PATH)
    print(f"  {len(df):,} games  (2024-25 season)")

    actuals = df["home_covers_spread"]

    # ------------------------------------------------------------------ #
    # Run simulations                                                      #
    # ------------------------------------------------------------------ #
    model_preds        = df["predicted_class"]
    always_home_preds  = pd.Series(1, index=df.index)   # always predict home covers
    always_away_preds  = pd.Series(0, index=df.index)   # always predict home doesn't cover

    results = [
        _simulate(model_preds,       actuals, "Model (baseline_xgb_tuned)"),
        _simulate(always_home_preds, actuals, "Baseline: always bet home covers"),
        _simulate(always_away_preds, actuals, "Baseline: always bet away covers"),
    ]
    summary = pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    # Print summary                                                        #
    # ------------------------------------------------------------------ #
    print(f"\n  Breakeven win rate at -110 odds: {BREAKEVEN_WIN_RATE * 100:.2f}%")
    print(f"  (Any strategy below this line loses money long-term)\n")

    col_w = 38
    print(f"{'─' * 78}")
    print(f"  {'Strategy':<{col_w}} {'W':>5} {'L':>5} {'Win%':>7} {'P&L':>10} {'ROI%':>7}")
    print(f"{'─' * 78}")
    for row in results:
        above = "✓" if row["win_rate_pct"] >= BREAKEVEN_WIN_RATE * 100 else "✗"
        print(
            f"  {row['strategy']:<{col_w}} "
            f"{row['wins']:>5} {row['losses']:>5} "
            f"{row['win_rate_pct']:>6.2f}%{above} "
            f"${row['total_profit']:>9,.2f} "
            f"{row['roi_pct']:>6.2f}%"
        )
    print(f"{'─' * 78}")
    print(f"  {'Breakeven threshold':<{col_w}} {'':>5} {'':>5} {'52.38%':>7}")

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved backtest summary to {OUT_PATH.name}")

    return summary


if __name__ == "__main__":
    simulate_flat_betting()

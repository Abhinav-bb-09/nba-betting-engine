import pandas as pd


def leakage_safe_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling mean that excludes the current row from its own window.

    The standard rolling mean (series.rolling(N).mean()) includes the value at
    position i in the window used to compute the feature at position i.  For a
    sports prediction model this is data leakage: the feature for game N would
    incorporate the outcome of game N itself — information that does not exist
    at tip-off and cannot be used at inference time.

    The fix is a single .shift(1) before .rolling(): this pushes each value one
    position forward so that game N's window only spans games 0 through N-1.
    The cost is that the first `window` rows become NaN (insufficient prior
    history), which is the correct behaviour — XGBoost handles these natively.

    This function is the canonical form of the pattern used throughout this
    project's rolling-feature pipeline scripts.  Centralising it here means
    the shift(1) is applied consistently in every call site and is documented
    in exactly one place.

    Args:
        series: Input series, sorted by date within a single team's history.
        window: Number of prior games to include in the rolling average.

    Returns:
        Series of rolling means where position i averages games i-window
        through i-1, with NaN for positions with insufficient history.
    """
    return series.shift(1).rolling(window).mean()

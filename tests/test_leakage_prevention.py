"""Tests for leakage_safe_rolling_mean.

Synthetic data is used throughout so the expected values are derivable by
hand — the point is to prove the function does what the docstring claims,
not to reproduce pipeline behaviour.
"""

import pandas as pd
import pytest

from src.utils.feature_helpers import leakage_safe_rolling_mean


def test_current_row_excluded():
    """The value at position i must NOT appear in the window used to compute
    the feature at position i.

    series = [10, 20, 30, 40, 50]
    After shift(1): [NaN, 10, 20, 30, 40]
    rolling(2).mean() at index 2 uses the window [10, 20] — NOT 30.
    Expected: (10 + 20) / 2 = 15.0
    """
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = leakage_safe_rolling_mean(series, window=2)
    assert result.iloc[2] == 15.0, (
        f"Expected 15.0 (mean of prior values 10 and 20), got {result.iloc[2]}. "
        "If this is 20.0, the shift(1) is missing and the current row (30) "
        "is being averaged with 10 — classic leakage."
    )


def test_first_rows_are_nan():
    """The first `window` rows must be NaN because they have fewer than
    `window` prior observations.

    For window=3 applied to [10, 20, 30, 40, 50]:
      After shift(1): [NaN, 10, 20, 30, 40]
      Index 0: no prior history          → NaN
      Index 1: one prior value (NaN)     → NaN  (need 3 non-NaN)
      Index 2: two prior values (10, 20) → NaN  (need 3 non-NaN)
      Index 3: three prior values        → first valid result
    """
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    window = 3
    result = leakage_safe_rolling_mean(series, window=window)

    for i in range(window):
        assert pd.isna(result.iloc[i]), (
            f"Expected NaN at index {i} (insufficient prior history for window={window}), "
            f"got {result.iloc[i]}."
        )


def test_window_size_respected():
    """A rolling(3) mean must average exactly the 3 immediately preceding
    values — not more, not fewer.

    series = [10, 20, 30, 40, 50, 60]
    After shift(1): [NaN, 10, 20, 30, 40, 50]
    Index 3 window: [10, 20, 30] → mean = 20.0
    Index 4 window: [20, 30, 40] → mean = 30.0
    Index 5 window: [30, 40, 50] → mean = 40.0
    """
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    result = leakage_safe_rolling_mean(series, window=3)

    assert result.iloc[3] == pytest.approx(20.0), (
        f"Index 3 should average [10, 20, 30] = 20.0, got {result.iloc[3]}"
    )
    assert result.iloc[4] == pytest.approx(30.0), (
        f"Index 4 should average [20, 30, 40] = 30.0, got {result.iloc[4]}"
    )
    assert result.iloc[5] == pytest.approx(40.0), (
        f"Index 5 should average [30, 40, 50] = 40.0, got {result.iloc[5]}"
    )

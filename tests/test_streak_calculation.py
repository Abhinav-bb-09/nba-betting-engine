"""Tests for _compute_streak from build_streak_rest_features.py.

_compute_streak is a standalone function that takes a single team's sorted
win/loss series and returns a signed streak series.  It does not depend on
file I/O, so it is tested directly on synthetic input — no mocking needed.

Synthetic data makes the expected streaks derivable by hand, which is the
whole point: if the expected value has to be read from production output
to write the test, the test cannot catch a pre-existing bug.
"""

import math

import pandas as pd
import pytest

from src.pipeline.build_streak_rest_features import _compute_streak


def test_win_streak():
    """Verify streaks for a mixed win/loss sequence.

    Input:  [1, 1, 1, 0, 0, 1]  (W W W L L W)

    Trace through the algorithm:
      prev = shift(1)         → [NaN, 1, 1, 1, 0, 0]
      prev.ne(prev.shift(1))  → [T,   T, F, F, T, F]
        (NaN != NaN is True in pandas; 1 != NaN is True; then False, False, True, False)
      group_id = cumsum       → [1,   2, 2, 2, 3, 3]
      streak_len (cumcount+1) → [1,   1, 2, 3, 1, 2]
      sign from prev.map      → [NaN, +1,+1,+1,-1,-1]
      result = len * sign     → [NaN, +1,+2,+3,-1,-2]
    """
    wins = pd.Series([1, 1, 1, 0, 0, 1], dtype=float)
    result = _compute_streak(wins)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(+1)
    assert result.iloc[2] == pytest.approx(+2)
    assert result.iloc[3] == pytest.approx(+3)
    assert result.iloc[4] == pytest.approx(-1)
    assert result.iloc[5] == pytest.approx(-2)


def test_first_game_is_nan():
    """The first entry must always be NaN regardless of the series content.

    shift(1) produces NaN at position 0 for any input, so the sign factor
    is NaN and the streak is undefined for the opening game.
    This holds for win streaks, loss streaks, and alternating series alike.
    """
    for wins in [
        pd.Series([1, 1, 1, 1], dtype=float),
        pd.Series([0, 0, 0, 0], dtype=float),
        pd.Series([1, 0, 1, 0], dtype=float),
    ]:
        result = _compute_streak(wins)
        assert pd.isna(result.iloc[0]), (
            f"Expected NaN at index 0 for series {wins.tolist()}, "
            f"got {result.iloc[0]}"
        )


def test_streak_sign_flip():
    """A streak must reset to ±1 immediately after the result changes direction —
    it must not carry forward the magnitude of the preceding streak.

    Input: [1, 1, 1, 0, ...]

    The game at index 4 is the first loss.  The streak entering it is the
    third game of a win streak (+3 at index 3).  The streak at index 4 must
    be −1 (start of a new losing streak), NOT −4 or any continuation of the
    prior magnitude.
    """
    wins = pd.Series([1, 1, 1, 0, 0, 1], dtype=float)
    result = _compute_streak(wins)

    # The transition from win streak to first loss resets to −1, not −4.
    assert result.iloc[4] == pytest.approx(-1), (
        f"Expected −1 at the first game after a 3-game win streak, "
        f"got {result.iloc[4]}.  A non-reset value suggests streak_len "
        "is accumulating across the direction change."
    )

    # Also verify the loss streak accumulates correctly after the reset:
    # two prior losses (indices 3 and 4 in wins) → streak entering index 5 = −2.
    assert result.iloc[5] == pytest.approx(-2), (
        f"Expected −2 at index 5 (2 consecutive prior losses), "
        f"got {result.iloc[5]}."
    )

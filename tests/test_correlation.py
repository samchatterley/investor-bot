"""Tests for risk/correlation.py — all network calls are replaced by _fetch_fn injection."""

import unittest

from risk.correlation import CORRELATION_THRESHOLD, LOOKBACK_DAYS, _pearson, correlated_with_held


def _make_fetch(data: dict[str, list[float]]):
    """Return a _fetch_fn that serves pre-canned closes, ignoring symbols/days args."""

    def _fetch(symbols, days):
        return {sym: data[sym] for sym in symbols if sym in data}

    return _fetch


def _from_returns(returns: list[float], start: float = 100.0) -> list[float]:
    """Build a price series by compounding a list of daily returns."""
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


# 21 daily prices → 20 daily returns (>= _MIN_OVERLAP=10)
_FLAT = [100.0] * 21
_UP = [100.0 + i for i in range(21)]

# Alternating +5% / -5% returns — genuinely zigzag (not monotone)
_ZIGZAG_UP = _from_returns([0.05, -0.05] * 10)  # 21 prices: up then down repeatedly
_ZIGZAG_DOWN = _from_returns([-0.05, 0.05] * 10)  # opposite phase → r ≈ -1


class TestPearson(unittest.TestCase):
    def test_identical_series_returns_one(self):
        corr = _pearson(_UP, _UP)
        self.assertAlmostEqual(corr, 1.0, places=6)

    def test_perfectly_anticorrelated_returns_minus_one(self):
        # Opposite-phase zigzag: when A goes up 5%, B goes down 5%, and vice versa
        corr = _pearson(_ZIGZAG_UP, _ZIGZAG_DOWN)
        self.assertAlmostEqual(corr, -1.0, places=6)

    def test_flat_series_returns_zero(self):
        # Flat price → zero returns → zero variance → denom = 0 → returns 0.0
        corr = _pearson(_FLAT, _FLAT)
        self.assertEqual(corr, 0.0)

    def test_insufficient_overlap_returns_zero(self):
        # Fewer than _MIN_OVERLAP + 1 prices (need ≥ 11 to get ≥ 10 returns)
        short = [100.0 + i for i in range(5)]
        corr = _pearson(short, short)
        self.assertEqual(corr, 0.0)

    def test_uses_tail_when_series_differ_in_length(self):
        long_a = [100.0 + i for i in range(30)]
        short_b = [200.0 + i for i in range(15)]
        # Should not raise; overlapping tail is 15 values
        corr = _pearson(long_a, short_b)
        self.assertIsInstance(corr, float)

    def test_unrelated_series_between_minus_one_and_one(self):
        import math

        corr = _pearson(_ZIGZAG_UP, _UP)
        self.assertFalse(math.isnan(corr))
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)


class TestCorrelatedWithHeld(unittest.TestCase):
    def test_no_held_symbols_returns_false(self):
        self.assertFalse(correlated_with_held("AAPL", set()))

    def test_candidate_data_unavailable_returns_false(self):
        # Fails open when candidate price data is missing
        fetch = _make_fetch({"MSFT": _UP})
        result = correlated_with_held("AAPL", {"MSFT"}, _fetch_fn=fetch)
        self.assertFalse(result)

    def test_held_data_unavailable_skips_pair(self):
        # No data for the held symbol — pair is skipped, should not block
        fetch = _make_fetch({"AAPL": _UP})
        result = correlated_with_held("AAPL", {"MSFT"}, _fetch_fn=fetch)
        self.assertFalse(result)

    def test_high_correlation_returns_true(self):
        fetch = _make_fetch({"AAPL": _UP, "MSFT": _UP})
        result = correlated_with_held("AAPL", {"MSFT"}, threshold=0.7, _fetch_fn=fetch)
        self.assertTrue(result)

    def test_negative_correlation_not_flagged(self):
        # Anticorrelated pair (r ≈ -1) should not trigger the filter (threshold=0.7 is positive)
        fetch = _make_fetch({"AAPL": _ZIGZAG_UP, "MSFT": _ZIGZAG_DOWN})
        result = correlated_with_held("AAPL", {"MSFT"}, threshold=0.7, _fetch_fn=fetch)
        self.assertFalse(result)

    def test_below_threshold_not_flagged(self):
        # Orthogonal-ish series with low correlation
        import math

        zigzag = [100.0 + math.sin(i) * 5 for i in range(21)]
        fetch = _make_fetch({"AAPL": _UP, "MSFT": zigzag})
        result = correlated_with_held("AAPL", {"MSFT"}, threshold=0.99, _fetch_fn=fetch)
        self.assertFalse(result)

    def test_custom_threshold_respected(self):
        # With threshold=0.0, any positive correlation triggers
        fetch = _make_fetch({"AAPL": _UP, "MSFT": _UP})
        result = correlated_with_held("AAPL", {"MSFT"}, threshold=0.0, _fetch_fn=fetch)
        self.assertTrue(result)

    def test_one_correlated_held_among_many_returns_true(self):
        # MSFT anticorrelated, GOOG identical — GOOG should trigger the filter
        fetch = _make_fetch({"AAPL": _ZIGZAG_UP, "MSFT": _ZIGZAG_DOWN, "GOOG": _ZIGZAG_UP})
        result = correlated_with_held("AAPL", {"MSFT", "GOOG"}, threshold=0.7, _fetch_fn=fetch)
        self.assertTrue(result)

    def test_all_held_uncorrelated_returns_false(self):
        # All held symbols are anticorrelated with candidate — none exceed threshold
        fetch = _make_fetch({"AAPL": _ZIGZAG_UP, "MSFT": _ZIGZAG_DOWN})
        result = correlated_with_held("AAPL", {"MSFT"}, threshold=0.7, _fetch_fn=fetch)
        self.assertFalse(result)

    def test_default_threshold_is_module_constant(self):
        # Verify the default threshold wires through correctly
        fetch = _make_fetch({"AAPL": _UP, "MSFT": _UP})
        # r=1.0 will always exceed CORRELATION_THRESHOLD=0.7
        result = correlated_with_held("AAPL", {"MSFT"}, _fetch_fn=fetch)
        self.assertTrue(result)
        self.assertEqual(CORRELATION_THRESHOLD, 0.7)

    def test_lookback_days_constant(self):
        self.assertEqual(LOOKBACK_DAYS, 20)


class TestFetchCloses(unittest.TestCase):
    """Test _fetch_closes via yfinance mock — no network calls."""

    def test_download_exception_returns_empty(self):
        from unittest.mock import patch

        from risk.correlation import _fetch_closes

        with patch("risk.correlation.yf.download", side_effect=RuntimeError("network")):
            result = _fetch_closes(["AAPL"], 20)
        self.assertEqual(result, {})

    def test_empty_download_returns_empty(self):
        from unittest.mock import patch

        import pandas as pd

        from risk.correlation import _fetch_closes

        with patch("risk.correlation.yf.download", return_value=pd.DataFrame()):
            result = _fetch_closes(["AAPL"], 20)
        self.assertEqual(result, {})

    def test_no_symbols_returns_empty(self):
        from risk.correlation import _fetch_closes

        result = _fetch_closes([], 20)
        self.assertEqual(result, {})

    def test_single_symbol_non_multiindex(self):
        from unittest.mock import patch

        import pandas as pd

        from risk.correlation import _fetch_closes

        prices = [100.0 + i for i in range(15)]
        index = pd.date_range("2026-01-01", periods=15, freq="B")
        df = pd.DataFrame({"Close": prices}, index=index)

        with patch("risk.correlation.yf.download", return_value=df):
            result = _fetch_closes(["AAPL"], 20)
        self.assertIn("AAPL", result)
        self.assertEqual(len(result["AAPL"]), 15)

    def test_single_symbol_insufficient_data_excluded(self):
        from unittest.mock import patch

        import pandas as pd

        from risk.correlation import _fetch_closes

        # Only 5 rows — below _MIN_OVERLAP=10
        df = pd.DataFrame({"Close": [100.0] * 5})

        with patch("risk.correlation.yf.download", return_value=df):
            result = _fetch_closes(["AAPL"], 20)
        self.assertEqual(result, {})

    def test_multiindex_extracts_per_symbol(self):
        from unittest.mock import patch

        import pandas as pd

        from risk.correlation import _fetch_closes

        prices = [100.0 + i for i in range(15)]
        index = pd.date_range("2026-01-01", periods=15, freq="B")
        # Build a proper MultiIndex DataFrame as yfinance returns
        cols = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Close", "MSFT")])
        df = pd.DataFrame(
            {("Close", "AAPL"): prices, ("Close", "MSFT"): [200.0 + i for i in range(15)]},
            index=index,
            columns=cols,
        )

        with patch("risk.correlation.yf.download", return_value=df):
            result = _fetch_closes(["AAPL", "MSFT"], 20)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)

    def test_multiindex_symbol_not_in_result_excluded(self):
        from unittest.mock import patch

        import pandas as pd

        from risk.correlation import _fetch_closes

        # Download returns MultiIndex but only has AAPL (MSFT was requested but missing)
        index = pd.date_range("2026-01-01", periods=15, freq="B")
        cols = pd.MultiIndex.from_tuples([("Close", "AAPL")])
        df = pd.DataFrame(
            {("Close", "AAPL"): [100.0 + i for i in range(15)]},
            index=index,
            columns=cols,
        )
        with patch("risk.correlation.yf.download", return_value=df):
            result = _fetch_closes(["AAPL", "MSFT"], 20)
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)

    def test_non_multiindex_multi_symbol_returns_empty(self):
        from unittest.mock import patch

        import pandas as pd

        from risk.correlation import _fetch_closes

        # yfinance returns flat columns with >1 symbol — unexpected; should return {}
        index = pd.date_range("2026-01-01", periods=15, freq="B")
        df = pd.DataFrame({"AAPL": [100.0 + i for i in range(15)]}, index=index)
        with patch("risk.correlation.yf.download", return_value=df):
            result = _fetch_closes(["AAPL", "MSFT"], 20)
        self.assertEqual(result, {})

    def test_multiindex_symbol_below_min_overlap_excluded(self):
        from unittest.mock import patch

        import pandas as pd

        from risk.correlation import _fetch_closes

        # AAPL has 15 rows, MSFT has only 3 (via NaN padding)
        index = pd.date_range("2026-01-01", periods=15, freq="B")
        aapl_prices = [100.0 + i for i in range(15)]
        msft_prices = [float("nan")] * 12 + [200.0, 201.0, 202.0]
        cols = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Close", "MSFT")])
        df = pd.DataFrame(
            {("Close", "AAPL"): aapl_prices, ("Close", "MSFT"): msft_prices},
            index=index,
            columns=cols,
        )

        with patch("risk.correlation.yf.download", return_value=df):
            result = _fetch_closes(["AAPL", "MSFT"], 20)
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)

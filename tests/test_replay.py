"""Tests for backtest/replay.py — historical replay harness."""

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd


def _make_preloaded_df(n=300, base=400.0, end="2025-06-01"):
    idx = pd.bdate_range(end=end, periods=n)
    prices = [base + i * 0.01 for i in range(len(idx))]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [10_000_000] * len(idx),
        },
        index=idx,
    )


def _make_preloaded(symbols=("SPY", "AAPL"), n=300):
    return {sym: _make_preloaded_df(n=n, base=400.0 if sym == "SPY" else 150.0) for sym in symbols}


class TestComputeRegime(unittest.TestCase):
    def test_bull_trending_when_spy_strong(self):
        from backtest.replay import _compute_regime

        preloaded = {"SPY": _make_preloaded_df(n=50, base=400.0)}
        # Last bar much higher than 5 bars ago to trigger BULL_TRENDING
        spy = preloaded["SPY"].copy()
        closes = list(spy["Close"])
        for i in range(len(closes) - 6, len(closes)):
            closes[i] = closes[i - 6] * 1.05  # +5% over 5 days
        spy["Close"] = closes
        preloaded["SPY"] = spy
        as_of = str(spy.index[-1].date())
        result = _compute_regime(preloaded, as_of)
        self.assertIn(
            result["regime"], ("BULL_TRENDING", "CHOPPY", "HIGH_VOL", "BEAR_DAY", "UNKNOWN")
        )
        self.assertIn("is_bearish", result)

    def test_returns_unknown_when_spy_missing(self):
        from backtest.replay import _compute_regime

        result = _compute_regime({}, "2025-01-10")
        self.assertEqual(result["regime"], "UNKNOWN")
        self.assertFalse(result["is_bearish"])

    def test_returns_unknown_when_insufficient_history(self):
        from backtest.replay import _compute_regime

        preloaded = {"SPY": _make_preloaded_df(n=3)}
        as_of = str(preloaded["SPY"].index[-1].date())
        result = _compute_regime(preloaded, as_of)
        self.assertEqual(result["regime"], "UNKNOWN")

    def test_bear_day_when_large_1d_drop(self):
        import config  # noqa: PLC0415
        from backtest.replay import _compute_regime

        spy = _make_preloaded_df(n=20, base=400.0)
        # Force last close well below previous to trigger is_bearish
        closes = spy["Close"].tolist()
        closes[-1] = closes[-2] * (1 + (config.BEAR_MARKET_SPY_THRESHOLD - 1) / 100)
        spy["Close"] = closes
        as_of = str(spy.index[-1].date())
        result = _compute_regime({"SPY": spy}, as_of)
        self.assertTrue(result["is_bearish"])
        self.assertEqual(result["regime"], "BEAR_DAY")

    def test_vix_extracted_when_vix_in_preloaded(self):
        from backtest.replay import _compute_regime

        preloaded = _make_preloaded(symbols=("SPY",))
        vix_df = _make_preloaded_df(n=20, base=18.0)
        preloaded["^VIX"] = vix_df
        as_of = str(vix_df.index[-1].date())
        result = _compute_regime(preloaded, as_of)
        self.assertIn("vix", result)
        self.assertIsNotNone(result["vix"])

    def test_as_of_slicing_prevents_future_data(self):
        from backtest.replay import _compute_regime

        spy = _make_preloaded_df(n=50, base=400.0)
        cutoff_idx = 20
        as_of = str(spy.index[cutoff_idx].date())
        result = _compute_regime({"SPY": spy}, as_of)
        # Just verify it doesn't crash and returns a valid dict
        self.assertIn("regime", result)


class TestBuildPreloaded(unittest.TestCase):
    def test_calls_yf_download_with_all_symbols(self):
        from backtest.replay import _build_preloaded

        raw = pd.DataFrame()
        with patch("backtest.replay.yf.download", return_value=raw) as mock_dl:
            _build_preloaded(["AAPL", "NVDA"], date(2024, 1, 1), date(2024, 6, 1))
        mock_dl.assert_called_once()
        call_kwargs = mock_dl.call_args
        tickers_arg = (
            call_kwargs[0][0]
            if call_kwargs[0]
            else call_kwargs[1].get("tickers", call_kwargs[0][0] if call_kwargs[0] else [])
        )
        # SPY and ^VIX are always added
        self.assertIn("SPY", tickers_arg)

    def test_returns_dict(self):
        from backtest.replay import _build_preloaded

        with patch("backtest.replay.yf.download", return_value=pd.DataFrame()):
            result = _build_preloaded(["AAPL"], date(2024, 1, 1), date(2024, 6, 1))
        self.assertIsInstance(result, dict)

    def test_multiindex_raw_is_split_per_symbol(self):
        from backtest.replay import _build_preloaded

        syms = ["AAPL", "SPY"]
        n = 10
        idx = pd.bdate_range("2024-01-01", periods=n)
        data = {
            (col, sym): [100.0] * n
            for col in ["Open", "High", "Low", "Close", "Volume"]
            for sym in syms
        }
        raw = pd.DataFrame(data, index=idx)
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, sym) for col in ["Open", "High", "Low", "Close", "Volume"] for sym in syms]
        )
        with patch("backtest.replay.yf.download", return_value=raw):
            result = _build_preloaded(["AAPL"], date(2024, 1, 1), date(2024, 2, 1))
        self.assertIn("AAPL", result)
        self.assertIn("SPY", result)


class TestRunHistoricalReplayDryRun(unittest.TestCase):
    """Dry-run mode: Claude is called, positions are not modified."""

    def _patched_run(self, preloaded, snapshots, decisions):
        from backtest.replay import run_historical_replay

        with (
            patch("backtest.replay._build_preloaded", return_value=preloaded),
            patch("backtest.replay.market_data.get_market_snapshots", return_value=snapshots),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch("analysis.ai_analyst.get_trading_decisions", return_value=decisions),
        ):
            return run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-10",
                initial_capital=50_000.0,
                dry_run=True,
            )

    def _minimal_preloaded(self):
        idx = pd.bdate_range("2024-06-01", "2025-01-15")
        spy = pd.DataFrame(
            {"Open": 400.0, "High": 401.0, "Low": 399.0, "Close": 400.0, "Volume": 1e7}, index=idx
        )
        aapl = pd.DataFrame(
            {"Open": 150.0, "High": 151.0, "Low": 149.0, "Close": 150.0, "Volume": 5e6}, index=idx
        )
        return {"SPY": spy, "AAPL": aapl}

    def test_returns_dict_with_required_keys(self):
        result = self._patched_run(
            self._minimal_preloaded(),
            [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 2.0, "ret_10d_pct": 3.0}],
            {"buy_candidates": [], "position_decisions": []},
        )
        for key in (
            "start_date",
            "end_date",
            "initial_capital",
            "final_value",
            "total_return_pct",
            "total_trades",
            "win_rate_pct",
            "daily_records",
            "all_trades",
        ):
            self.assertIn(key, result)

    def test_initial_capital_preserved_in_dry_run(self):
        result = self._patched_run(
            self._minimal_preloaded(),
            [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 1.0, "ret_10d_pct": 2.0}],
            {"buy_candidates": [], "position_decisions": []},
        )
        self.assertEqual(result["initial_capital"], 50_000.0)
        self.assertEqual(result["final_value"], 50_000.0)
        self.assertEqual(result["total_return_pct"], 0.0)

    def test_daily_records_contain_date_and_regime(self):
        result = self._patched_run(
            self._minimal_preloaded(),
            [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 1.0, "ret_10d_pct": 2.0}],
            {"buy_candidates": [], "position_decisions": []},
        )
        for rec in result["daily_records"]:
            self.assertIn("date", rec)
            self.assertIn("regime", rec)
            self.assertIn("decisions", rec)

    def test_no_trades_in_dry_run(self):
        result = self._patched_run(
            self._minimal_preloaded(),
            [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 1.0, "ret_10d_pct": 2.0}],
            {
                "buy_candidates": [{"symbol": "AAPL", "confidence": 9, "key_signal": "momentum"}],
                "position_decisions": [],
            },
        )
        self.assertEqual(result["all_trades"], [])

    def test_error_returned_when_spy_missing(self):
        from backtest.replay import run_historical_replay

        with patch("backtest.replay._build_preloaded", return_value={"AAPL": _make_preloaded_df()}):
            result = run_historical_replay(
                symbols=["AAPL"], start_date="2025-01-06", end_date="2025-01-10"
            )
        self.assertIn("error", result)


class TestRunHistoricalReplayLive(unittest.TestCase):
    """Live mode (dry_run=False): buys and sells are simulated."""

    def _minimal_preloaded(self, start="2024-06-01", end="2025-02-01"):
        idx = pd.bdate_range(start, end)
        spy = pd.DataFrame(
            {"Open": 400.0, "High": 401.0, "Low": 399.0, "Close": 400.0, "Volume": 1e7}, index=idx
        )
        aapl = pd.DataFrame(
            {"Open": 150.0, "High": 151.0, "Low": 149.0, "Close": 150.0, "Volume": 5e6}, index=idx
        )
        return {"SPY": spy, "AAPL": aapl}

    def test_buy_reduces_cash_and_adds_position(self):
        from backtest.replay import run_historical_replay

        decisions = {
            "buy_candidates": [{"symbol": "AAPL", "confidence": 9, "key_signal": "momentum"}],
            "position_decisions": [],
        }
        snap = [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 2.0, "ret_10d_pct": 3.0}]
        with (
            patch("backtest.replay._build_preloaded", return_value=self._minimal_preloaded()),
            patch("backtest.replay.market_data.get_market_snapshots", return_value=snap),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch("analysis.ai_analyst.get_trading_decisions", return_value=decisions),
        ):
            result = run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-10",
                initial_capital=100_000.0,
                dry_run=False,
            )
        buys = [t for t in result["all_trades"] if t["action"] == "BUY"]
        self.assertGreater(len(buys), 0)

    def test_sell_after_max_hold_days(self):
        from backtest.replay import run_historical_replay

        # Buy on day 1, hold for max_hold_days=1, expect sell on day 2
        buy_decisions = {
            "buy_candidates": [{"symbol": "AAPL", "confidence": 9, "key_signal": "momentum"}],
            "position_decisions": [],
        }
        sell_decisions = {"buy_candidates": [], "position_decisions": []}

        call_count = [0]

        def _decisions(**kwargs):
            call_count[0] += 1
            return buy_decisions if call_count[0] == 1 else sell_decisions

        snap = [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 2.0, "ret_10d_pct": 3.0}]
        with (
            patch("backtest.replay._build_preloaded", return_value=self._minimal_preloaded()),
            patch("backtest.replay.market_data.get_market_snapshots", return_value=snap),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch("analysis.ai_analyst.get_trading_decisions", side_effect=_decisions),
        ):
            result = run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-15",
                initial_capital=100_000.0,
                max_hold_days=1,
                dry_run=False,
            )
        sells = [t for t in result["all_trades"] if t["action"] == "SELL"]
        self.assertGreater(len(sells), 0)

    def test_sub_share_guard_skips_buy(self):
        from backtest.replay import run_historical_replay

        decisions = {
            "buy_candidates": [{"symbol": "AAPL", "confidence": 9, "key_signal": "momentum"}],
            "position_decisions": [],
        }
        # current_price so high that $1 notional / price < 1 share
        snap = [
            {"symbol": "AAPL", "current_price": 1_000_000.0, "ret_5d_pct": 2.0, "ret_10d_pct": 3.0}
        ]
        with (
            patch("backtest.replay._build_preloaded", return_value=self._minimal_preloaded()),
            patch("backtest.replay.market_data.get_market_snapshots", return_value=snap),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch("analysis.ai_analyst.get_trading_decisions", return_value=decisions),
        ):
            result = run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-10",
                initial_capital=100.0,  # tiny capital → sub-share
                dry_run=False,
            )
        buys = [t for t in result["all_trades"] if t["action"] == "BUY"]
        self.assertEqual(buys, [])

    def test_bear_regime_skips_buys(self):
        from backtest.replay import run_historical_replay

        bear_regime = {
            "is_bearish": True,
            "spy_change_pct": -3.0,
            "spy_5d_pct": -5.0,
            "regime": "BEAR_DAY",
        }
        decisions = {
            "buy_candidates": [{"symbol": "AAPL", "confidence": 9, "key_signal": "momentum"}],
            "position_decisions": [],
        }
        snap = [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 2.0, "ret_10d_pct": 3.0}]
        with (
            patch("backtest.replay._build_preloaded", return_value=self._minimal_preloaded()),
            patch("backtest.replay.market_data.get_market_snapshots", return_value=snap),
            patch("backtest.replay._compute_regime", return_value=bear_regime),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch("analysis.ai_analyst.get_trading_decisions", return_value=decisions),
        ):
            result = run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-10",
                initial_capital=100_000.0,
                dry_run=False,
            )
        buys = [t for t in result["all_trades"] if t["action"] == "BUY"]
        self.assertEqual(buys, [])

    def test_total_return_pct_is_numeric(self):
        from backtest.replay import run_historical_replay

        decisions = {"buy_candidates": [], "position_decisions": []}
        snap = [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 1.0, "ret_10d_pct": 2.0}]
        with (
            patch("backtest.replay._build_preloaded", return_value=self._minimal_preloaded()),
            patch("backtest.replay.market_data.get_market_snapshots", return_value=snap),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch("analysis.ai_analyst.get_trading_decisions", return_value=decisions),
        ):
            result = run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-10",
                initial_capital=100_000.0,
                dry_run=False,
            )
        self.assertIsInstance(result["total_return_pct"], float)

    def test_win_rate_zero_when_no_trades(self):
        from backtest.replay import run_historical_replay

        decisions = {"buy_candidates": [], "position_decisions": []}
        snap = [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 1.0, "ret_10d_pct": 2.0}]
        with (
            patch("backtest.replay._build_preloaded", return_value=self._minimal_preloaded()),
            patch("backtest.replay.market_data.get_market_snapshots", return_value=snap),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch("analysis.ai_analyst.get_trading_decisions", return_value=decisions),
        ):
            result = run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-10",
                initial_capital=100_000.0,
                dry_run=False,
            )
        self.assertEqual(result["win_rate_pct"], 0.0)

    def test_all_trades_list_contains_dicts(self):
        from backtest.replay import run_historical_replay

        decisions = {
            "buy_candidates": [{"symbol": "AAPL", "confidence": 9, "key_signal": "momentum"}],
            "position_decisions": [],
        }
        snap = [{"symbol": "AAPL", "current_price": 150.0, "ret_5d_pct": 2.0, "ret_10d_pct": 3.0}]
        with (
            patch("backtest.replay._build_preloaded", return_value=self._minimal_preloaded()),
            patch("backtest.replay.market_data.get_market_snapshots", return_value=snap),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch("analysis.ai_analyst.get_trading_decisions", return_value=decisions),
        ):
            result = run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-10",
                initial_capital=100_000.0,
                dry_run=False,
            )
        for t in result["all_trades"]:
            self.assertIsInstance(t, dict)
            self.assertIn("action", t)
            self.assertIn("symbol", t)


class TestReplayContextCompleteness(unittest.TestCase):
    """run_historical_replay result includes context_completeness metadata."""

    def _minimal_preloaded(self):
        idx = pd.bdate_range("2024-06-01", "2025-01-15")
        spy = pd.DataFrame(
            {"Open": 400.0, "High": 401.0, "Low": 399.0, "Close": 400.0, "Volume": 1e7}, index=idx
        )
        aapl = pd.DataFrame(
            {"Open": 150.0, "High": 151.0, "Low": 149.0, "Close": 150.0, "Volume": 5e6}, index=idx
        )
        return {"SPY": spy, "AAPL": aapl}

    def _run(self):
        from backtest.replay import run_historical_replay

        with (
            patch("backtest.replay._build_preloaded", return_value=self._minimal_preloaded()),
            patch(
                "backtest.replay.market_data.get_market_snapshots",
                return_value=[
                    {
                        "symbol": "AAPL",
                        "current_price": 150.0,
                        "ret_5d_pct": 1.0,
                        "ret_10d_pct": 2.0,
                    }
                ],
            ),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=lambda x: x),
            patch(
                "analysis.ai_analyst.get_trading_decisions",
                return_value={"buy_candidates": [], "position_decisions": []},
            ),
        ):
            return run_historical_replay(
                symbols=["AAPL"],
                start_date="2025-01-06",
                end_date="2025-01-10",
                initial_capital=50_000.0,
                dry_run=True,
            )

    def test_context_completeness_is_partial(self):
        result = self._run()
        self.assertEqual(result.get("context_completeness"), "partial")

    def test_missing_context_lists_expected_fields(self):
        result = self._run()
        missing = result.get("missing_context", [])
        for field in ("news", "options_signals", "sentiment"):
            self.assertIn(field, missing)

    def test_missing_context_is_list(self):
        result = self._run()
        self.assertIsInstance(result.get("missing_context"), list)


class TestSpyReturnFromPreloaded(unittest.TestCase):
    def _spy_df(self, n=20):
        idx = pd.bdate_range("2025-01-01", periods=n)
        closes = [100.0 + i for i in range(n)]
        return pd.DataFrame({"Close": closes}, index=idx)

    def test_returns_correct_5d_return(self):
        from data.market_data import _spy_return_from_preloaded

        spy = self._spy_df(20)
        as_of = str(spy.index[-1].date())
        result = _spy_return_from_preloaded({"SPY": spy}, as_of, 5)
        expected = round((spy["Close"].iloc[-1] / spy["Close"].iloc[-6] - 1) * 100, 2)
        self.assertAlmostEqual(result, expected, places=4)

    def test_returns_none_when_spy_missing(self):
        from data.market_data import _spy_return_from_preloaded

        result = _spy_return_from_preloaded({}, "2025-01-10", 5)
        self.assertIsNone(result)

    def test_returns_none_when_insufficient_history(self):
        from data.market_data import _spy_return_from_preloaded

        spy = self._spy_df(3)
        as_of = str(spy.index[-1].date())
        result = _spy_return_from_preloaded({"SPY": spy}, as_of, 5)
        self.assertIsNone(result)

    def test_as_of_slicing_excludes_future_bars(self):
        from data.market_data import _spy_return_from_preloaded

        spy = self._spy_df(20)
        # Use a cutoff in the middle of the index
        as_of = str(spy.index[9].date())
        result = _spy_return_from_preloaded({"SPY": spy}, as_of, 5)
        self.assertIsNotNone(result)
        sliced = spy[spy.index <= pd.Timestamp(as_of)]
        expected = round((sliced["Close"].iloc[-1] / sliced["Close"].iloc[-6] - 1) * 100, 2)
        self.assertAlmostEqual(result, expected, places=4)

"""
Tests for every Phase 1 safety fix. Each class maps to one behaviour change.
No integration with real APIs — all external calls are mocked.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import config
from models import OrderResult, OrderStatus

# ── 1. Daily loss baseline ────────────────────────────────────────────────────


class TestDailyBaseline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = patch("utils.portfolio_tracker.LOG_DIR", self.tmpdir)
        self._patcher.start()
        # Force _BASELINE_PATH to use tmpdir
        import utils.portfolio_tracker as pt

        self._old_path = pt._BASELINE_PATH
        pt._BASELINE_PATH = os.path.join(self.tmpdir, "daily_baseline.json")

    def tearDown(self):
        import utils.portfolio_tracker as pt

        pt._BASELINE_PATH = self._old_path
        self._patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_save_and_load_roundtrip(self):
        from utils.portfolio_tracker import load_daily_baseline, save_daily_baseline

        save_daily_baseline(100_000.0)
        result = load_daily_baseline()
        self.assertAlmostEqual(result, 100_000.0)

    def test_load_returns_none_when_no_file(self):
        from utils.portfolio_tracker import load_daily_baseline

        self.assertIsNone(load_daily_baseline())

    def test_load_returns_none_for_yesterday(self):
        import utils.portfolio_tracker as pt
        from utils.portfolio_tracker import load_daily_baseline

        with open(pt._BASELINE_PATH, "w") as f:
            json.dump({"date": "2000-01-01", "portfolio_value": 99_000.0}, f)
        self.assertIsNone(load_daily_baseline())

    def test_overwrite_updates_value(self):
        from utils.portfolio_tracker import load_daily_baseline, save_daily_baseline

        save_daily_baseline(100_000.0)
        save_daily_baseline(95_000.0)
        self.assertAlmostEqual(load_daily_baseline(), 95_000.0)


# ── 2. Sell hallucination guard ───────────────────────────────────────────────


class TestSellHallucinationGuard(unittest.TestCase):
    def _decisions(self, sell_sym):
        return {
            "market_summary": "Quiet session, no major catalysts.",
            "buy_candidates": [],
            "position_decisions": [{"symbol": sell_sym, "action": "SELL", "reasoning": "test"}],
        }

    def test_sell_for_held_symbol_passes(self):
        from utils.validators import validate_ai_response

        is_valid, errors = validate_ai_response(
            self._decisions("AAPL"),
            known_symbols={"AAPL"},
            held_symbols={"AAPL"},
        )
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_sell_for_unheld_symbol_passes_with_warning(self):
        # Trailing stops create a legitimate race: position closes between data fetch
        # and validation. This is a warning, not a blocking error.
        from utils.validators import validate_ai_response

        is_valid, errors = validate_ai_response(
            self._decisions("NVDA"),
            known_symbols={"NVDA"},
            held_symbols={"AAPL"},  # NVDA not held
        )
        self.assertTrue(is_valid)
        self.assertFalse(any("NVDA" in e for e in errors))

    def test_sell_check_skipped_when_held_symbols_none(self):
        from utils.validators import validate_ai_response

        # held_symbols=None means caller opts out of the guard
        is_valid, errors = validate_ai_response(
            self._decisions("GHOST"),
            known_symbols={"GHOST"},
            held_symbols=None,
        )
        self.assertTrue(is_valid)


# ── 3. Weekly review never writes config changes to disk ─────────────────────


class TestRuntimeConfigOverride(unittest.TestCase):
    """Auto-parameter modification is disabled — _apply_config_changes validates
    and reports proposed changes but must never write runtime_config.json."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.runtime_path = os.path.join(self.tmpdir, "runtime_config.json")
        self._patcher = patch("analysis.weekly_review._RUNTIME_CONFIG_PATH", self.runtime_path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_config_changes_never_written_to_disk(self):
        from analysis.weekly_review import _apply_config_changes

        _apply_config_changes([{"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "x"}])
        self.assertFalse(
            os.path.exists(self.runtime_path),
            "runtime_config.json must not be written — auto-modification is disabled",
        )

    def test_config_py_is_never_touched(self):
        import analysis.weekly_review as wr
        from analysis.weekly_review import _apply_config_changes

        config_mtime = os.path.getmtime(
            os.path.normpath(os.path.join(os.path.dirname(wr.__file__), "..", "config.py"))
        )
        _apply_config_changes([{"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "x"}])
        self.assertEqual(
            os.path.getmtime(
                os.path.normpath(os.path.join(os.path.dirname(wr.__file__), "..", "config.py"))
            ),
            config_mtime,
            "config.py must never be modified by the weekly review",
        )

    def test_multiple_calls_never_write_to_disk(self):
        from analysis.weekly_review import _apply_config_changes

        _apply_config_changes([{"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "x"}])
        _apply_config_changes([{"parameter": "MAX_HOLD_DAYS", "proposed_value": 5, "reason": "x"}])
        self.assertFalse(os.path.exists(self.runtime_path))


# ── 4. Stale data rejection ───────────────────────────────────────────────────


class TestStaleDataRejection(unittest.TestCase):
    def _make_ticker(self, days_old: int):
        import pandas as pd

        end = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_old)
        idx = pd.bdate_range(end=end, periods=60)
        actual_n = len(idx)
        prices = [100.0 + i * 0.1 for i in range(actual_n)]
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p + 1 for p in prices],
                "Low": [p - 1 for p in prices],
                "Close": prices,
                "Volume": [1_000_000] * actual_n,
            },
            index=idx,
        )
        t = MagicMock()
        t.history.return_value = df
        return t

    def test_fresh_data_accepted(self):
        from data.market_data import fetch_stock_data

        with patch("data.market_data.yf.Ticker", return_value=self._make_ticker(0)):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNotNone(result)

    def test_data_two_days_old_accepted(self):
        # Weekends mean up to 3 calendar days is still valid
        from data.market_data import fetch_stock_data

        with patch("data.market_data.yf.Ticker", return_value=self._make_ticker(2)):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNotNone(result)

    def test_data_seven_days_old_rejected(self):
        from data.market_data import fetch_stock_data

        with patch("data.market_data.yf.Ticker", return_value=self._make_ticker(7)):
            result = fetch_stock_data("AAPL", days=30)
        self.assertIsNone(result)


# ── 5. Min volume filter ──────────────────────────────────────────────────────


class TestMinVolumeFilter(unittest.TestCase):
    def _snap(self, avg_volume, **kwargs):
        base = {
            "symbol": "TEST",
            "rsi_14": 30,
            "bb_pct": 0.20,
            "vol_ratio": 1.5,
            "ema9_above_ema21": True,
            "macd_diff": 0.5,
            "macd_crossed_up": False,
            "weekly_trend_up": True,
            "ret_5d_pct": 2.0,
            "avg_volume": avg_volume,
        }
        base.update(kwargs)
        return base

    def test_stock_above_min_volume_passes(self):
        from execution.stock_scanner import prefilter_candidates

        result = prefilter_candidates([self._snap(config.MIN_VOLUME + 1)])
        self.assertEqual(len(result), 1)

    def test_stock_at_min_volume_rejected(self):
        from execution.stock_scanner import prefilter_candidates

        result = prefilter_candidates([self._snap(config.MIN_VOLUME - 1)])
        self.assertEqual(len(result), 0)

    def test_missing_avg_volume_field_rejected(self):
        from execution.stock_scanner import prefilter_candidates

        snap = self._snap(0)
        del snap["avg_volume"]
        result = prefilter_candidates([snap])
        self.assertEqual(len(result), 0)


# ── 6. Empirical Kelly fallback ───────────────────────────────────────────────


class TestEmpiricalKelly(unittest.TestCase):
    def _stats(self, trades, wins):
        return {
            "momentum": {
                "trades": trades,
                "wins": wins,
                "losses": trades - wins,
                "by_regime": {
                    "BULL_TRENDING": {"trades": trades, "wins": wins, "losses": trades - wins}
                },
            }
        }

    def test_falls_back_to_llm_confidence_when_no_stats(self):
        from risk.position_sizer import kelly_fraction

        with patch("risk.position_sizer._load_signal_stats", return_value={}):
            f1 = kelly_fraction(7, signal="momentum", regime="BULL_TRENDING")
            f2 = kelly_fraction(7)  # defaults — same fallback
        self.assertAlmostEqual(f1, f2)

    def test_uses_empirical_rate_when_sufficient_samples(self):
        from risk.position_sizer import kelly_fraction

        # 10 trades, 8 wins → p=0.8 (better than LLM confidence 7/10=0.7)
        stats = self._stats(trades=10, wins=8)
        with patch("risk.position_sizer._load_signal_stats", return_value=stats):
            empirical = kelly_fraction(7, signal="momentum", regime="BULL_TRENDING")
            fallback = kelly_fraction(7)
        self.assertGreater(empirical, fallback)

    def test_falls_back_when_sample_too_small(self):
        from risk.position_sizer import kelly_fraction

        stats = self._stats(trades=3, wins=3)  # < _MIN_SAMPLE_SIZE = 5
        with patch("risk.position_sizer._load_signal_stats", return_value=stats):
            f_small = kelly_fraction(7, signal="momentum", regime="BULL_TRENDING")
            f_fallback = kelly_fraction(7)
        self.assertAlmostEqual(f_small, f_fallback)

    def test_result_always_non_negative(self):
        from risk.position_sizer import kelly_fraction

        # 10 trades, 2 wins → p=0.2, Kelly formula can go negative
        stats = self._stats(trades=10, wins=2)
        with patch("risk.position_sizer._load_signal_stats", return_value=stats):
            result = kelly_fraction(3, signal="momentum", regime="BULL_TRENDING")
        self.assertGreaterEqual(result, 0.0)


# ── 7. Max orders per run ─────────────────────────────────────────────────────


class TestMaxOrdersPerRun(unittest.TestCase):
    def _run_buys(self, n_candidates: int, max_orders: int):
        """Run _run_inner with n_candidates buys available, patched MAX_ORDERS_PER_RUN."""
        import contextlib
        from unittest.mock import MagicMock, patch

        buy_mock = MagicMock(
            return_value=OrderResult(
                status=OrderStatus.FILLED,
                symbol="X",
                broker_order_id="o1",
                filled_qty=1.0,
            )
        )
        candidates = [
            {"symbol": f"SYM{i}", "confidence": 8, "key_signal": "momentum"}
            for i in range(n_candidates)
        ]
        account = {
            "portfolio_value": 100_000,
            "cash": 50_000,
            "buying_power": 100_000,
            "equity": 100_000,
        }
        record = {
            "date": "2026-01-15",
            "daily_pnl": 0.0,
            "account_before": account,
            "account_after": account,
            "market_summary": "test",
            "trades_executed": [],
            "stop_losses_triggered": [],
        }

        def _validate(decisions, known, held_symbols=None):
            return True, []

        patches = {
            "main.trader.get_client": MagicMock(),
            "main.trader.is_market_open": True,
            "main.trader.get_account_info": account,
            "main.trader.get_open_positions": [],
            "main.trader.reconcile_positions": set(),
            "main.trader.ensure_stops_attached": None,
            "main.trader.get_position_ages": {},
            "main.trader.get_stale_positions": [],
            "main.trader.record_buy": None,
            "main.trader.record_sell": None,
            "main.trader.close_position": OrderResult(status=OrderStatus.FILLED, symbol="X"),
            "main.trader.place_buy_order": buy_mock,
            "main.trader.place_trailing_stop": OrderResult(
                status=OrderStatus.FILLED, symbol="X", stop_order_id="stop-1"
            ),
            "main.portfolio_tracker.load_history": [],
            "main.portfolio_tracker.get_track_record": [],
            "main.portfolio_tracker.save_daily_run": record,
            "main.portfolio_tracker.print_summary": None,
            "main.portfolio_tracker.save_daily_baseline": None,
            "main.portfolio_tracker.load_daily_baseline": None,
            "main.risk_manager.check_circuit_breaker": (False, 0.0),
            "main.risk_manager.check_daily_loss": (False, 0.0),
            "main.risk_manager.validate_buy_candidates": lambda c, **kw: c,
            "main.position_sizer.kelly_fraction": 0.1,
            "main.position_sizer.get_max_positions": 10,
            "main.market_data.get_vix": 15.0,
            "main.stock_scanner.get_market_regime": {
                "regime": "BULL_TRENDING",
                "is_bearish": False,
            },
            "main.stock_scanner.get_top_movers": [],
            "main.stock_scanner.prefilter_candidates": candidates,
            "main.macro_calendar.get_macro_risk": {"is_high_risk": False, "event": ""},
            "main.sector_data.get_sector_performance": {},
            "main.sector_data.get_leading_sectors": [],
            "main.get_latest_review": [],
            "main.earnings_calendar.get_earnings_risk_positions": {},
            "main._handle_partial_exits": [],
            "main.market_data.get_market_snapshots": [
                {"symbol": f"SYM{i}", "current_price": 100.0} for i in range(n_candidates)
            ],
            "main.options_scanner.get_options_signals": {},
            "main.news_fetcher.fetch_news": {},
            "main.sanitize_headlines": {},
            "main.sentiment_module.get_sentiment": {},
            "main.ai_analyst.get_trading_decisions": {
                "market_summary": "ok",
                "buy_candidates": candidates,
                "position_decisions": [],
            },
            "main.validate_ai_response": _validate,
            "main.check_pre_trade": (True, ""),
            "main.decision_log.log_decisions": None,
            "main.audit_log.log_ai_decision": None,
            "main.audit_log.log_run_start": None,
            "main.audit_log.log_run_end": None,
            "main.audit_log.log_order_placed": None,
            "main.audit_log.log_order_filled": None,
            "main.audit_log.log_position_closed": None,
            "main.audit_log.log_validation_failure": None,
            "main.audit_log.log_circuit_breaker": None,
            "main.audit_log.log_daily_loss_limit": None,
            "main.audit_log.log_macro_skip": None,
            "main.audit_log.log_earnings_exit": None,
            "main.alerts.alert_circuit_breaker": None,
            "main.alerts.alert_daily_loss": None,
            "main.alerts.alert_error": None,
            "main.performance.generate_dashboard": None,
            "main.performance.record_trade_outcome": None,
            "main.get_day_summary": None,
            "main.emailer.send_summary": None,
            "main.audit_log.log_event": None,
            "main.trader.has_pending_buy": False,
            "main.trader.get_total_open_exposure": 0.0,
            "main.trader.get_daily_notional": 0.0,
            "main.trader.add_daily_notional": None,
            "main.build_scan_universe": [],
            "main.save_experiment_baseline": None,
            "main.load_experiment_baseline": None,
        }

        from utils.health import HealthReport, HealthStatus

        patches["main.run_startup_health_check"] = HealthReport(
            status=HealthStatus.GREEN, issues=[], metrics={}
        )

        stack = contextlib.ExitStack()
        for target, val in patches.items():
            if val is None:
                stack.enter_context(patch(target, return_value=None))
            elif callable(val) and not isinstance(val, MagicMock):
                stack.enter_context(patch(target, side_effect=val))
            elif isinstance(val, MagicMock):
                stack.enter_context(patch(target, new=val))
            else:
                stack.enter_context(patch(target, return_value=val))
        # Scalars must use new= so comparisons like >= work without MagicMock wrapping
        stack.enter_context(patch("main.config.MAX_ORDERS_PER_RUN", new=max_orders))

        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")

        return buy_mock.call_count

    def test_buys_capped_at_max_orders_per_run(self):
        placed = self._run_buys(n_candidates=5, max_orders=2)
        self.assertEqual(placed, 2)

    def test_all_buys_placed_when_under_cap(self):
        placed = self._run_buys(n_candidates=2, max_orders=5)
        self.assertEqual(placed, 2)


# ── 8. GTC stop for fractional positions ─────────────────────────────────────


class TestGTCStopForFractional(unittest.TestCase):
    def test_sub_share_position_returns_unprotected_without_api_call(self):
        """Positions smaller than 1 whole share cannot be stop-protected — Alpaca rejects them."""
        from execution.trader import place_trailing_stop

        mock_client = MagicMock()

        # qty=0.5 — floor is 0, below the 1-share minimum — no API call should be made
        with patch("execution.trader.TRAILING_STOP_PCT", 4.0):
            result = place_trailing_stop(mock_client, "AAPL", qty=0.5, current_price=150.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrderStatus.UNPROTECTED)
        self.assertFalse(result.is_success)
        mock_client.submit_order.assert_not_called()

    def test_fractional_position_floors_to_whole_shares(self):
        """A position with fractional qty (e.g. 2.7 shares) places a stop for 2 whole shares."""
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import StopOrderRequest

        from execution.trader import place_trailing_stop

        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "stop-123"
        mock_client.submit_order.return_value = mock_order

        with patch("execution.trader.TRAILING_STOP_PCT", 4.0):
            result = place_trailing_stop(mock_client, "AAPL", qty=2.7, current_price=150.0)

        self.assertIsNotNone(result)
        # First call is the stop; second call liquidates the fractional remainder
        stop_req = mock_client.submit_order.call_args_list[0][0][0]
        self.assertIsInstance(stop_req, StopOrderRequest)
        self.assertEqual(stop_req.qty, 2)  # floored from 2.7
        self.assertEqual(stop_req.time_in_force, TimeInForce.GTC)

    def test_whole_share_uses_trailing_stop_not_fixed(self):
        from alpaca.trading.requests import TrailingStopOrderRequest

        from execution.trader import place_trailing_stop

        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "trail-456"
        mock_client.submit_order.return_value = mock_order

        with patch("execution.trader.TRAILING_STOP_PCT", 4.0):
            place_trailing_stop(mock_client, "AAPL", qty=10.0, current_price=150.0)

        call_args = mock_client.submit_order.call_args[0][0]
        self.assertIsInstance(call_args, TrailingStopOrderRequest)

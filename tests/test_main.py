"""Tests for main.py — lock, kill switch, partial exits, and run orchestration."""
import contextlib
import os
import shutil
import tempfile
import time
import unittest
import unittest.mock
from unittest.mock import MagicMock, patch

import config
from models import OrderResult, OrderStatus

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _account(value=100_000, cash=30_000):
    return {"portfolio_value": value, "cash": cash, "buying_power": cash * 2, "equity": value}


def _regime(bearish=False):
    return {"regime": "CHOPPY", "is_bearish": bearish}


def _macro(high_risk=False):
    return {"is_high_risk": high_risk, "event": ""}


def _decisions(buys=None, sells=None):
    return {
        "market_summary": "Test summary",
        "buy_candidates": buys or [],
        "position_decisions": sells or [],
    }


def _saved_record(date="2026-01-15", pnl=0.0, mode="open"):
    acc = _account()
    return {
        "date": date, "daily_pnl": pnl,
        "account_before": acc, "account_after": acc,
        "market_summary": f"{mode} check",
        "trades_executed": [], "stop_losses_triggered": [],
    }


# ── Lock tests ─────────────────────────────────────────────────────────────────

class LockBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lock_file = os.path.join(self.tmpdir, f".lock_{config.today_et().isoformat()}")
        self._patcher = patch("main._lock_file", return_value=self.lock_file)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmpdir)


class TestAcquireLock(LockBase):

    def test_creates_lock_returns_true(self):
        from main import _acquire_lock
        result = _acquire_lock()
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.lock_file))

    def test_returns_false_when_fresh_lock_exists(self):
        from main import _acquire_lock
        _acquire_lock()
        result = _acquire_lock()
        self.assertFalse(result)

    def test_stale_lock_auto_cleared(self):
        from main import _acquire_lock
        os.makedirs(self.tmpdir, exist_ok=True)
        with open(self.lock_file, "w"):
            pass
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(self.lock_file, (old_time, old_time))
        result = _acquire_lock()
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.lock_file))


class TestReleaseLock(LockBase):

    def test_removes_lock_file(self):
        from main import _acquire_lock, _release_lock
        _acquire_lock()
        _release_lock()
        self.assertFalse(os.path.exists(self.lock_file))

    def test_no_error_when_lock_already_gone(self):
        from main import _release_lock
        try:
            _release_lock()
        except Exception:
            self.fail("_release_lock raised on missing lock file")


# ── Kill switch tests ──────────────────────────────────────────────────────────

class TestRunKillSwitch(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.halt_file = os.path.join(self.tmpdir, "HALT")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _run(self, positions=None, cancel_raises=False, remaining_after=None):
        """Helper: run kill switch with optional position list and post-close broker state.

        remaining_after: list of positions still open at broker after close attempts.
                         Defaults to [] (all positions confirmed closed).
        """
        mock_client = MagicMock()
        if cancel_raises:
            mock_client.cancel_orders.side_effect = Exception("API down")
        pos = positions or []
        remaining = remaining_after if remaining_after is not None else []
        close_results = [OrderResult(status=OrderStatus.FILLED, symbol=p["symbol"]) for p in pos]

        with patch("main.trader.get_client", return_value=mock_client), \
             patch("main.trader.get_open_positions", side_effect=[pos, remaining]), \
             patch("main.trader.close_position", side_effect=close_results or [OrderResult(OrderStatus.REJECTED, "X")]), \
             patch("main.audit_log.log_position_closed"), \
             patch("main.audit_log.log_kill_switch"), \
             patch("main.alerts.alert_error"), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main.config.LOG_DIR", self.tmpdir):
            from main import _run_kill_switch
            _run_kill_switch()
        return mock_client

    def test_writes_halt_file(self):
        self._run()
        self.assertTrue(os.path.exists(self.halt_file))

    def test_halt_file_contains_halted_marker(self):
        self._run()
        with open(self.halt_file) as _f:
            self.assertIn("HALTED", _f.read())

    def test_halt_file_contains_liquidation_complete_true(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 100.0, "unrealized_plpc": 2.0}]
        self._run(positions=positions, remaining_after=[])
        with open(self.halt_file) as _f:
            content = _f.read()
        self.assertIn("Liquidation complete: True", content)

    def test_halt_file_contains_per_symbol_status(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 100.0, "unrealized_plpc": 2.0}]
        self._run(positions=positions, remaining_after=[])
        with open(self.halt_file) as _f:
            content = _f.read()
        self.assertIn("AAPL", content)
        self.assertIn("broker=closed", content)

    def test_halt_file_marks_still_open_positions(self):
        positions = [
            {"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0},
            {"symbol": "NVDA", "unrealized_pl": 0.0, "unrealized_plpc": 0.0},
        ]
        close_results = [
            OrderResult(status=OrderStatus.FILLED, symbol="AAPL"),
            OrderResult(status=OrderStatus.TIMEOUT, symbol="NVDA"),
        ]
        remaining_after = [{"symbol": "NVDA", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        with patch("main.trader.get_client", return_value=MagicMock()), \
             patch("main.trader.get_open_positions", side_effect=[positions, remaining_after]), \
             patch("main.trader.close_position", side_effect=close_results), \
             patch("main.audit_log.log_position_closed"), \
             patch("main.audit_log.log_kill_switch"), \
             patch("main.alerts.alert_error"), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main.config.LOG_DIR", self.tmpdir):
            from main import _run_kill_switch
            _run_kill_switch()
        with open(self.halt_file) as _f:
            content = _f.read()
        self.assertIn("Liquidation complete: False", content)
        self.assertIn("NVDA", content)
        self.assertIn("broker=open", content)
        self.assertIn("AAPL", content)
        self.assertIn("broker=closed", content)

    def test_alert_warns_when_positions_remain_open(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        close_results = [OrderResult(status=OrderStatus.TIMEOUT, symbol="AAPL")]
        remaining_after = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        alert_mock = MagicMock()
        with patch("main.trader.get_client", return_value=MagicMock()), \
             patch("main.trader.get_open_positions", side_effect=[positions, remaining_after]), \
             patch("main.trader.close_position", side_effect=close_results), \
             patch("main.audit_log.log_position_closed"), \
             patch("main.audit_log.log_kill_switch"), \
             patch("main.alerts.alert_error", alert_mock), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main.config.LOG_DIR", self.tmpdir):
            from main import _run_kill_switch
            _run_kill_switch()
        call_args = alert_mock.call_args[0][1]
        self.assertIn("WARNING", call_args)

    def test_cancels_all_orders(self):
        client = self._run()
        client.cancel_orders.assert_called_once()

    def test_closes_each_open_position(self):
        positions = [
            {"symbol": "AAPL", "unrealized_pl": 100.0, "unrealized_plpc": 2.0},
            {"symbol": "NVDA", "unrealized_pl": -50.0, "unrealized_plpc": -1.0},
        ]
        close_mock = MagicMock(side_effect=[
            OrderResult(status=OrderStatus.FILLED, symbol="AAPL"),
            OrderResult(status=OrderStatus.FILLED, symbol="NVDA"),
        ])
        with patch("main.trader.get_client", return_value=MagicMock()), \
             patch("main.trader.get_open_positions", side_effect=[positions, []]), \
             patch("main.trader.close_position", close_mock), \
             patch("main.audit_log.log_position_closed"), \
             patch("main.audit_log.log_kill_switch"), \
             patch("main.alerts.alert_error"), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main.config.LOG_DIR", self.tmpdir):
            from main import _run_kill_switch
            _run_kill_switch()
        self.assertEqual(close_mock.call_count, 2)

    def test_cancel_failure_does_not_abort_position_close(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        self._run(positions=positions, cancel_raises=True)
        self.assertTrue(os.path.exists(self.halt_file))

    def test_sends_alert_email(self):
        alert_mock = MagicMock()
        with patch("main.trader.get_client", return_value=MagicMock()), \
             patch("main.trader.get_open_positions", side_effect=[[], []]), \
             patch("main.trader.close_position"), \
             patch("main.audit_log.log_kill_switch"), \
             patch("main.alerts.alert_error", alert_mock), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main.config.LOG_DIR", self.tmpdir):
            from main import _run_kill_switch
            _run_kill_switch()
        alert_mock.assert_called_once()


# ── Clear halt tests ───────────────────────────────────────────────────────────

class TestRunClearHalt(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.halt_file = os.path.join(self.tmpdir, "HALT")
        self._patcher = patch("main.config.HALT_FILE", self.halt_file)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_removes_halt_file(self):
        from main import _run_clear_halt
        with open(self.halt_file, "w") as f:
            f.write("HALTED")
        _run_clear_halt()
        self.assertFalse(os.path.exists(self.halt_file))

    def test_no_error_without_halt_file(self):
        from main import _run_clear_halt
        try:
            _run_clear_halt()
        except Exception:
            self.fail("_run_clear_halt raised with no halt file present")

    def test_logs_halt_cleared(self):
        from main import _run_clear_halt
        with open(self.halt_file, "w") as f:
            f.write("HALTED")
        with patch("main.audit_log.log_halt_cleared") as mock_log:
            _run_clear_halt()
        mock_log.assert_called_once()


# ── Partial exits tests ────────────────────────────────────────────────────────

class TestHandlePartialExits(unittest.TestCase):

    def _pos(self, symbol, plpc, qty=10.0, market_value=1000.0, current_price=100.0):
        return {
            "symbol": symbol, "unrealized_plpc": plpc,
            "qty": qty, "market_value": market_value, "current_price": current_price,
        }

    def test_no_exits_below_threshold(self):
        from main import _handle_partial_exits
        positions = [self._pos("AAPL", plpc=config.PARTIAL_PROFIT_PCT - 1)]
        result = _handle_partial_exits(MagicMock(), positions, dry_run=False)
        self.assertEqual(result, [])

    def test_partial_exit_triggered_at_threshold(self):
        from main import _handle_partial_exits
        with patch("main.trader.cancel_open_orders"), \
             patch("main.trader.get_position_meta", return_value={}), \
             patch("main.trader.place_partial_sell", return_value=OrderResult(OrderStatus.FILLED, "AAPL", broker_order_id="x", filled_qty=5.0)), \
             patch("main.trader.record_partial_exit"), \
             patch("main.trader.place_trailing_stop"), \
             patch("main.audit_log.log_order_placed"), \
             patch("main.audit_log.log_order_filled"):
            result = _handle_partial_exits(
                MagicMock(),
                [self._pos("AAPL", plpc=config.PARTIAL_PROFIT_PCT + 1)],
                dry_run=False,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "AAPL")
        self.assertEqual(result[0]["action"], "PARTIAL SELL")

    def test_dry_run_returns_trade_without_placing_order(self):
        from main import _handle_partial_exits
        with patch("main.trader.cancel_open_orders") as cancel_mock:
            result = _handle_partial_exits(
                MagicMock(),
                [self._pos("AAPL", plpc=config.PARTIAL_PROFIT_PCT + 5)],
                dry_run=True,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["detail"], "dry run")
        cancel_mock.assert_not_called()

    def test_empty_positions_returns_empty(self):
        from main import _handle_partial_exits
        result = _handle_partial_exits(MagicMock(), [], dry_run=False)
        self.assertEqual(result, [])

    def test_multiple_positions_only_exits_above_threshold(self):
        from main import _handle_partial_exits
        positions = [
            self._pos("AAPL", plpc=config.PARTIAL_PROFIT_PCT + 5),
            self._pos("NVDA", plpc=config.PARTIAL_PROFIT_PCT - 1),
        ]
        with patch("main.trader.cancel_open_orders"), \
             patch("main.trader.get_position_meta", return_value={}), \
             patch("main.trader.place_partial_sell", return_value=OrderResult(OrderStatus.FILLED, "AAPL", broker_order_id="x", filled_qty=5.0)), \
             patch("main.trader.record_partial_exit"), \
             patch("main.trader.place_trailing_stop"), \
             patch("main.audit_log.log_order_placed"), \
             patch("main.audit_log.log_order_filled"):
            result = _handle_partial_exits(MagicMock(), positions, dry_run=False)
        symbols = [r["symbol"] for r in result]
        self.assertIn("AAPL", symbols)
        self.assertNotIn("NVDA", symbols)


# ── run() guard tests ──────────────────────────────────────────────────────────

class TestRunGuards(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lock_file = os.path.join(self.tmpdir, f".lock_{config.today_et().isoformat()}")
        self.halt_file = os.path.join(self.tmpdir, "HALT")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_exits_when_halt_file_present(self):
        with open(self.halt_file, "w") as f:
            f.write("HALTED")
        with patch("main.config.validate"), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main._lock_file", return_value=self.lock_file), \
             patch("sys.exit") as mock_exit:
            from main import run
            run()
        mock_exit.assert_called_with(1)

    def test_exits_on_missing_alpaca_key(self):
        with patch("main.config.validate"), \
             patch("main.config.ALPACA_API_KEY", ""), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main._lock_file", return_value=self.lock_file), \
             patch("sys.exit") as mock_exit:
            from main import run
            run()
        mock_exit.assert_called_with(1)

    def test_exits_on_missing_anthropic_key(self):
        with patch("main.config.validate"), \
             patch("main.config.ALPACA_API_KEY", "valid-key"), \
             patch("main.config.ANTHROPIC_API_KEY", ""), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main._lock_file", return_value=self.lock_file), \
             patch("sys.exit") as mock_exit:
            from main import run
            run()
        mock_exit.assert_called_with(1)

    def test_returns_without_running_if_lock_held(self):
        with open(self.lock_file, "w"):
            pass
        mock_inner = MagicMock()
        with patch("main.config.validate"), \
             patch("main.config.ALPACA_API_KEY", "valid-key"), \
             patch("main.config.ANTHROPIC_API_KEY", "valid-key"), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main._lock_file", return_value=self.lock_file), \
             patch("main._run_inner", mock_inner):
            from main import run
            run()
        mock_inner.assert_not_called()

    def test_releases_lock_after_unhandled_exception(self):
        with patch("main.config.validate"), \
             patch("main.config.ALPACA_API_KEY", "valid-key"), \
             patch("main.config.ANTHROPIC_API_KEY", "valid-key"), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main._lock_file", return_value=self.lock_file), \
             patch("main._run_inner", side_effect=RuntimeError("unexpected")), \
             patch("main.alerts.alert_error"):
            from main import run
            run()
        self.assertFalse(os.path.exists(self.lock_file))

    def test_alert_sent_on_unhandled_exception(self):
        alert_mock = MagicMock()
        with patch("main.config.validate"), \
             patch("main.config.ALPACA_API_KEY", "valid-key"), \
             patch("main.config.ANTHROPIC_API_KEY", "valid-key"), \
             patch("main.config.HALT_FILE", self.halt_file), \
             patch("main._lock_file", return_value=self.lock_file), \
             patch("main._run_inner", side_effect=RuntimeError("boom")), \
             patch("main.alerts.alert_error", alert_mock):
            from main import run
            run()
        alert_mock.assert_called_once()
        self.assertIn("main.run", alert_mock.call_args[0][0])

    def test_exits_when_live_mode_without_accept_flag(self):
        with (
            patch("main.config.validate"),
            patch("main.config.ALPACA_API_KEY", "valid-key"),
            patch("main.config.ANTHROPIC_API_KEY", "valid-key"),
            patch("main.config.IS_PAPER", False),
            patch("main.config.LIVE_CONFIRM", ""),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("sys.exit") as mock_exit,
        ):
            from main import run
            run(dry_run=False)
        mock_exit.assert_called_with(1)

    def test_live_mode_with_correct_flag_does_not_exit(self):
        mock_inner = MagicMock()
        with (
            patch("main.config.validate"),
            patch("main.config.ALPACA_API_KEY", "valid-key"),
            patch("main.config.ANTHROPIC_API_KEY", "valid-key"),
            patch("main.config.IS_PAPER", False),
            patch("main.config.LIVE_CONFIRM", "I-ACCEPT-REAL-MONEY-RISK"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("main._run_inner", mock_inner),
        ):
            from main import run
            run(dry_run=False)
        mock_inner.assert_called_once()

    def test_exits_when_config_validate_raises(self):
        with (
            patch("main.config.validate", side_effect=ValueError("bad config")),
            patch("main.config.ALPACA_API_KEY", "valid-key"),
            patch("main.config.ANTHROPIC_API_KEY", "valid-key"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("sys.exit") as mock_exit,
        ):
            from main import run
            run()
        mock_exit.assert_called_with(1)


# ── _run_inner helpers ─────────────────────────────────────────────────────────

class RunInnerBase(unittest.TestCase):
    """Provides _patch_all() context manager that stubs every external dependency."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _patch_all(self, **overrides):
        """Return an ExitStack with all _run_inner dependencies patched."""
        defaults = {
            "main.trader.get_client":                   MagicMock(),
            "main.trader.is_market_open":               True,
            "main.trader.get_account_info":             _account(),
            "main.trader.get_open_positions":           [],
            "main.trader.reconcile_positions":          None,
            "main.trader.ensure_stops_attached":        None,
            "main.trader.get_position_ages":            {},
            "main.trader.get_stale_positions":          [],
            "main.trader.record_buy":                   None,
            "main.trader.record_sell":                  None,
            "main.trader.close_position":               OrderResult(status=OrderStatus.FILLED, symbol="X"),
            "main.trader.place_buy_order":              OrderResult(status=OrderStatus.FILLED, symbol="AAPL", broker_order_id="x", filled_qty=1.0),
            "main.trader.place_trailing_stop":          None,
            "main.portfolio_tracker.load_history":           [],
            "main.portfolio_tracker.get_track_record":       [],
            "main.portfolio_tracker.save_daily_run":         _saved_record(),
            "main.portfolio_tracker.print_summary":          None,
            "main.portfolio_tracker.save_daily_baseline":    None,
            "main.portfolio_tracker.load_daily_baseline":    None,
            "main.risk_manager.check_circuit_breaker":  (False, 0.0),
            "main.risk_manager.check_daily_loss":       (False, 0.0),
            "main.risk_manager.validate_buy_candidates": lambda c, **kw: c,
            "main.position_sizer.risk_budget_size":     500.0,
            "main.position_sizer.get_max_positions":    5,
            "main.market_data.get_vix":                 18.0,
            "main.stock_scanner.get_market_regime":     _regime(),
            "main.stock_scanner.get_top_movers":        [],
            "main.stock_scanner.prefilter_candidates":  [],
            "main.macro_calendar.get_macro_risk":       _macro(),
            "main.sector_data.get_sector_performance":  {},
            "main.sector_data.get_leading_sectors":     [],
            "main.get_latest_review":                   "",
            "main.earnings_calendar.get_earnings_risk_positions": {},
            "main._handle_partial_exits":               [],
            "main.market_data.get_market_snapshots":    [{"symbol": "AAPL", "current_price": 150.0}],
            "main.options_scanner.get_options_signals": {},
            "main.news_fetcher.fetch_news":             {},
            "main.sanitize_headlines":                  {},
            "main.sentiment_module.get_sentiment":      {},
            "main.ai_analyst.get_trading_decisions":    _decisions(),
            "main.validate_ai_response":                (True, []),
            "main.check_pre_trade":                     (True, ""),
            "main.decision_log.log_decisions":          None,
            "main.audit_log.log_ai_decision":           None,
            "main.audit_log.log_run_start":             None,
            "main.audit_log.log_run_end":               None,
            "main.audit_log.log_order_placed":          None,
            "main.audit_log.log_order_filled":          None,
            "main.audit_log.log_position_closed":       None,
            "main.audit_log.log_validation_failure":    None,
            "main.audit_log.log_circuit_breaker":       None,
            "main.audit_log.log_daily_loss_limit":      None,
            "main.audit_log.log_macro_skip":            None,
            "main.audit_log.log_earnings_exit":         None,
            "main.alerts.alert_circuit_breaker":        None,
            "main.alerts.alert_daily_loss":             None,
            "main.alerts.alert_error":                  None,
            "main.performance.generate_dashboard":      None,
            "main.performance.record_trade_outcome":    None,
            "main.get_day_summary":                     {"date": "2026-01-15", "account_before": _account(), "account_after": _account(), "daily_pnl": 0.0},
            "main.emailer.send_summary":                None,
        }
        defaults.update(overrides)

        stack = contextlib.ExitStack()
        mocks = {}
        for target, return_val in defaults.items():
            if isinstance(return_val, Exception):
                m = stack.enter_context(patch(target, side_effect=return_val))
            elif isinstance(return_val, MagicMock):
                # Use new= so callers can assert directly on the passed mock object
                m = stack.enter_context(patch(target, new=return_val))
            elif callable(return_val):
                m = stack.enter_context(patch(target, side_effect=return_val))
            elif return_val is None:
                m = stack.enter_context(patch(target, return_value=None))
            else:
                m = stack.enter_context(patch(target, return_value=return_val))
            mocks[target] = m
        return stack, mocks


class TestRunInnerMarketClosed(RunInnerBase):

    def test_returns_early_without_saving(self):
        save_mock = MagicMock()
        stack, mocks = self._patch_all(**{
            "main.trader.is_market_open": False,
            "main.portfolio_tracker.save_daily_run": save_mock,
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        save_mock.assert_not_called()


class TestRunInnerMiddayClose(RunInnerBase):

    def test_midday_does_not_send_email(self):
        email_mock = MagicMock()
        stack, mocks = self._patch_all(**{"main.emailer.send_summary": email_mock})
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="midday", today="2026-01-15")
        email_mock.assert_not_called()

    def test_close_sends_email(self):
        email_mock = MagicMock()
        stack, mocks = self._patch_all(**{"main.emailer.send_summary": email_mock})
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="close", today="2026-01-15")
        email_mock.assert_called_once()

    def test_close_dry_run_does_not_send_email(self):
        email_mock = MagicMock()
        stack, mocks = self._patch_all(**{"main.emailer.send_summary": email_mock})
        with stack:
            from main import _run_inner
            _run_inner(dry_run=True, mode="close", today="2026-01-15")
        email_mock.assert_not_called()

    def test_midday_saves_record(self):
        save_mock = MagicMock(return_value=_saved_record(mode="midday"))
        stack, mocks = self._patch_all(**{"main.portfolio_tracker.save_daily_run": save_mock})
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="midday", today="2026-01-15")
        save_mock.assert_called_once()


class TestRunInnerCircuitBreaker(RunInnerBase):

    def test_circuit_breaker_blocks_buys(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(**{
            "main.risk_manager.check_circuit_breaker": (True, -15.0),
            "main.trader.place_buy_order": buy_mock,
            "main.ai_analyst.get_trading_decisions": decisions,
            "main.stock_scanner.prefilter_candidates": [{"symbol": "AAPL", "current_price": 150.0}],
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_circuit_breaker_sends_alert(self):
        alert_mock = MagicMock()
        stack, mocks = self._patch_all(**{
            "main.risk_manager.check_circuit_breaker": (True, -15.0),
            "main.alerts.alert_circuit_breaker": alert_mock,
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        alert_mock.assert_called_once()


class TestRunInnerDailyLoss(RunInnerBase):

    def test_daily_loss_closes_positions(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": -500.0, "unrealized_plpc": -5.0,
                      "qty": 10.0, "market_value": 1000.0, "current_price": 100.0}]
        close_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        stack, mocks = self._patch_all(**{
            "main.risk_manager.check_daily_loss": (True, -6.0),
            "main.trader.get_open_positions": positions,
            "main.trader.close_position": close_mock,
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        close_mock.assert_called_once()

    def test_daily_loss_dry_run_skips_close(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": -500.0, "unrealized_plpc": -5.0,
                      "qty": 10.0, "market_value": 1000.0, "current_price": 100.0}]
        close_mock = MagicMock()
        stack, mocks = self._patch_all(**{
            "main.risk_manager.check_daily_loss": (True, -6.0),
            "main.trader.get_open_positions": positions,
            "main.trader.close_position": close_mock,
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=True, mode="open", today="2026-01-15")
        close_mock.assert_not_called()

    def test_daily_loss_sends_alert(self):
        alert_mock = MagicMock()
        stack, mocks = self._patch_all(**{
            "main.risk_manager.check_daily_loss": (True, -6.0),
            "main.alerts.alert_daily_loss": alert_mock,
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        alert_mock.assert_called_once()


class TestRunInnerOpenAborts(RunInnerBase):

    def test_no_snapshots_aborts_without_saving(self):
        save_mock = MagicMock()
        stack, mocks = self._patch_all(**{
            "main.market_data.get_market_snapshots": [],
            "main.portfolio_tracker.save_daily_run": save_mock,
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        save_mock.assert_not_called()

    def test_ai_returns_none_aborts_without_saving(self):
        save_mock = MagicMock()
        stack, mocks = self._patch_all(**{
            "main.ai_analyst.get_trading_decisions": None,
            "main.portfolio_tracker.save_daily_run": save_mock,
            "main.stock_scanner.prefilter_candidates": [{"symbol": "AAPL", "current_price": 150.0}],
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        save_mock.assert_not_called()


class TestRunInnerBuyFiltering(RunInnerBase):

    def test_validation_failure_removes_out_of_universe_candidates(self):
        """Validation failure is fail-closed — all Claude buys are blocked, including out-of-universe candidates."""
        buy_mock = MagicMock()
        # GHOST is not in prefilter_candidates, so it's not in ai_known_symbols
        decisions = _decisions(buys=[{"symbol": "GHOST", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(**{
            "main.ai_analyst.get_trading_decisions": decisions,
            "main.validate_ai_response": (False, ["BUY candidate 'GHOST' not in scanned universe — rejecting"]),
            "main.trader.place_buy_order": buy_mock,
            "main.stock_scanner.prefilter_candidates": [{"symbol": "AAPL", "current_price": 150.0}],
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_validation_failure_blocks_all_claude_buys(self):
        """Validation failure is fail-closed — AAPL (valid) is also blocked when GHOST (invalid) taints the response."""
        buy_mock = MagicMock()
        decisions = _decisions(buys=[
            {"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"},
            {"symbol": "GHOST", "confidence": 7, "key_signal": "momentum"},
        ])
        stack, mocks = self._patch_all(**{
            "main.ai_analyst.get_trading_decisions": decisions,
            "main.validate_ai_response": (False, ["BUY candidate 'GHOST' not in scanned universe — rejecting"]),
            "main.trader.place_buy_order": buy_mock,
            "main.stock_scanner.prefilter_candidates": [{"symbol": "AAPL", "current_price": 150.0}],
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_structural_validation_failure_blocks_claude_driven_sells(self):
        """Structural (schema) errors zero position_decisions — Claude-recommended sells are blocked."""
        sell_mock = MagicMock()
        decisions = _decisions(
            buys=[{"symbol": "MSFT", "confidence": 8, "key_signal": "momentum"}],
            sells=[{"symbol": "AAPL", "action": "SELL", "confidence": 7, "reasoning": "momentum weakening"}],
        )
        stack, mocks = self._patch_all(**{
            "main.trader.get_open_positions": [{"symbol": "AAPL", "qty": 10, "unrealized_plpc": -0.02}],
            "main.ai_analyst.get_trading_decisions": decisions,
            # Structural error (Pydantic field path) — not a "BUY candidate '" prefix
            "main.validate_ai_response": (False, ["position_decisions → 0 → action: Input should be 'HOLD' or 'SELL'"]),
            "main.trader.close_position": sell_mock,
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        sell_mock.assert_not_called()

    def test_buy_domain_error_preserves_claude_sells(self):
        """BUY domain errors (out-of-universe) block buys only — Claude sell decisions still execute."""
        sell_mock = MagicMock()
        sell_mock.return_value = OrderResult(status=OrderStatus.FILLED, symbol="AAPL")
        decisions = _decisions(
            buys=[{"symbol": "GHOST", "confidence": 8, "key_signal": "momentum"}],
            sells=[{"symbol": "AAPL", "action": "SELL", "confidence": 7, "reasoning": "momentum weakening"}],
        )
        stack, mocks = self._patch_all(**{
            "main.trader.get_open_positions": [{"symbol": "AAPL", "qty": 10, "unrealized_plpc": -0.02}],
            "main.ai_analyst.get_trading_decisions": decisions,
            "main.validate_ai_response": (False, ["BUY candidate 'GHOST' not in scanned universe — rejecting"]),
            "main.trader.close_position": sell_mock,
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        # AAPL sell must still fire despite the invalid buy candidate
        sell_mock.assert_called_once()

    def test_bearish_regime_blocks_buys(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(**{
            "main.stock_scanner.get_market_regime": _regime(bearish=True),
            "main.ai_analyst.get_trading_decisions": decisions,
            "main.trader.place_buy_order": buy_mock,
            "main.stock_scanner.prefilter_candidates": [{"symbol": "AAPL", "current_price": 150.0}],
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_macro_risk_blocks_buys(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(**{
            "main.macro_calendar.get_macro_risk": _macro(high_risk=True),
            "main.ai_analyst.get_trading_decisions": decisions,
            "main.trader.place_buy_order": buy_mock,
            "main.stock_scanner.prefilter_candidates": [{"symbol": "AAPL", "current_price": 150.0}],
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_dry_run_does_not_place_buy_orders(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(**{
            "main.ai_analyst.get_trading_decisions": decisions,
            "main.trader.place_buy_order": buy_mock,
            "main.stock_scanner.prefilter_candidates": [{"symbol": "AAPL", "current_price": 150.0}],
            "main.trader.get_account_info": _account(100_000, 50_000),
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=True, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_sell_executed_in_open_mode(self):
        close_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        positions = [{"symbol": "AAPL", "unrealized_pl": 100.0, "unrealized_plpc": 2.0,
                      "qty": 10.0, "market_value": 1000.0, "current_price": 100.0}]
        decisions = _decisions(sells=[{"symbol": "AAPL", "action": "SELL", "reasoning": "stale"}])
        stack, mocks = self._patch_all(**{
            "main.trader.get_open_positions": positions,
            "main.ai_analyst.get_trading_decisions": decisions,
            "main.trader.close_position": close_mock,
            "main.trader.get_position_meta": MagicMock(return_value={"signal": "m", "regime": "X", "confidence": 7}),
        })
        with stack:
            from main import _run_inner
            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        close_mock.assert_called()


if __name__ == "__main__":
    unittest.main()

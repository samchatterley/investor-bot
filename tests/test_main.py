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
        "date": date,
        "daily_pnl": pnl,
        "account_before": acc,
        "account_after": acc,
        "market_summary": f"{mode} check",
        "trades_executed": [],
        "stop_losses_triggered": [],
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

    def _run(self, positions=None, cancel_raises=False, remaining_after=None, requery_raises=False):
        """Run kill switch and return the mock client.

        remaining_after: broker positions still open after close attempts (default: []).
        requery_raises: make the post-liquidation get_open_positions raise.
        """
        mock_client = MagicMock()
        if cancel_raises:
            mock_client.cancel_orders.side_effect = Exception("API down")
        pos = positions or []
        close_results = [OrderResult(status=OrderStatus.FILLED, symbol=p["symbol"]) for p in pos]

        if requery_raises:
            get_pos_side_effect = [pos, Exception("broker offline")]
        else:
            remaining = remaining_after if remaining_after is not None else []
            get_pos_side_effect = [pos, remaining]

        def get_positions_mock(client_arg):
            val = get_pos_side_effect.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with (
            patch("main.trader.get_client", return_value=mock_client),
            patch("main.trader.get_open_positions", side_effect=get_positions_mock),
            patch(
                "main.trader.close_position",
                side_effect=close_results or [OrderResult(OrderStatus.REJECTED, "X")],
            ),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        return mock_client

    def _halt_json(self) -> dict:
        import json

        with open(self.halt_file) as _f:
            return json.load(_f)

    # ── Basic smoke tests ──────────────────────────────────────────────────────

    def test_writes_halt_file(self):
        self._run()
        self.assertTrue(os.path.exists(self.halt_file))

    def test_halt_file_is_valid_json(self):
        self._run()
        data = self._halt_json()
        self.assertTrue(data["halted"])

    def test_halt_file_has_timestamp(self):
        self._run()
        self.assertIn("timestamp", self._halt_json())

    def test_cancels_all_orders(self):
        client = self._run()
        client.cancel_orders.assert_called_once()

    def test_sends_alert_email(self):
        alert_mock = MagicMock()
        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=[[], []]),
            patch("main.trader.close_position"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error", alert_mock),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        alert_mock.assert_called_once()

    def test_cancel_failure_does_not_abort_position_close(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        self._run(positions=positions, cancel_raises=True)
        self.assertTrue(os.path.exists(self.halt_file))

    def test_closes_each_open_position(self):
        positions = [
            {"symbol": "AAPL", "unrealized_pl": 100.0, "unrealized_plpc": 2.0},
            {"symbol": "NVDA", "unrealized_pl": -50.0, "unrealized_plpc": -1.0},
        ]
        close_mock = MagicMock(
            side_effect=[
                OrderResult(status=OrderStatus.FILLED, symbol="AAPL"),
                OrderResult(status=OrderStatus.FILLED, symbol="NVDA"),
            ]
        )
        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=[positions, []]),
            patch("main.trader.close_position", close_mock),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        self.assertEqual(close_mock.call_count, 2)

    # ── Liquidation-complete matrix ────────────────────────────────────────────

    def test_liquidation_complete_true_when_all_positions_gone(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 100.0, "unrealized_plpc": 2.0}]
        self._run(positions=positions, remaining_after=[])
        self.assertTrue(self._halt_json()["liquidation_complete"])

    def test_liquidation_complete_false_when_position_remains_after_filled_order(self):
        """Regression: order API says FILLED but broker still shows the position."""
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        remaining_after = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        close_results = [OrderResult(status=OrderStatus.FILLED, symbol="AAPL")]
        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=[positions, remaining_after]),
            patch("main.trader.close_position", side_effect=close_results),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        data = self._halt_json()
        self.assertFalse(data["liquidation_complete"])
        self.assertIn("AAPL", data["positions_remaining"])

    def test_liquidation_complete_false_when_timeout_and_position_remains(self):
        positions = [{"symbol": "MSFT", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        remaining_after = [{"symbol": "MSFT", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        close_results = [OrderResult(status=OrderStatus.TIMEOUT, symbol="MSFT")]
        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=[positions, remaining_after]),
            patch("main.trader.close_position", side_effect=close_results),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        data = self._halt_json()
        self.assertFalse(data["liquidation_complete"])

    def test_liquidation_complete_false_when_rejected_and_position_remains(self):
        positions = [{"symbol": "NVDA", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        remaining_after = [{"symbol": "NVDA", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        close_results = [
            OrderResult(status=OrderStatus.REJECTED, symbol="NVDA", rejection_reason="not found")
        ]
        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=[positions, remaining_after]),
            patch("main.trader.close_position", side_effect=close_results),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        data = self._halt_json()
        self.assertFalse(data["liquidation_complete"])

    def test_mixed_symbols_partial_incomplete(self):
        """Two positions: AAPL closed, MSFT still open — incomplete."""
        positions = [
            {"symbol": "AAPL", "unrealized_pl": 100.0, "unrealized_plpc": 2.0},
            {"symbol": "MSFT", "unrealized_pl": -20.0, "unrealized_plpc": -0.5},
        ]
        close_results = [
            OrderResult(status=OrderStatus.FILLED, symbol="AAPL"),
            OrderResult(status=OrderStatus.TIMEOUT, symbol="MSFT"),
        ]
        remaining_after = [{"symbol": "MSFT", "unrealized_pl": -20.0, "unrealized_plpc": -0.5}]
        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=[positions, remaining_after]),
            patch("main.trader.close_position", side_effect=close_results),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        data = self._halt_json()
        self.assertFalse(data["liquidation_complete"])
        self.assertEqual(data["positions_remaining"], ["MSFT"])
        self.assertEqual(data["symbols"]["AAPL"]["broker_status"], "closed")
        self.assertEqual(data["symbols"]["MSFT"]["broker_status"], "open")

    def test_order_timeout_but_broker_confirms_closed(self):
        """Broker says position is gone even though order API timed out — treat as closed."""
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        close_results = [OrderResult(status=OrderStatus.TIMEOUT, symbol="AAPL")]
        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=[positions, []]),
            patch("main.trader.close_position", side_effect=close_results),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        data = self._halt_json()
        self.assertTrue(data["liquidation_complete"])
        self.assertEqual(data["symbols"]["AAPL"]["broker_status"], "closed")

    # ── Re-query failure ───────────────────────────────────────────────────────

    def test_requery_failure_sets_liquidation_complete_unknown(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        self._run(positions=positions, requery_raises=True)
        data = self._halt_json()
        self.assertEqual(data["liquidation_complete"], "UNKNOWN")

    def test_requery_failure_records_verification_error(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        self._run(positions=positions, requery_raises=True)
        data = self._halt_json()
        self.assertIn("verification_error", data)
        self.assertTrue(len(data["verification_error"]) > 0)

    def test_requery_failure_sets_broker_status_unknown_per_symbol(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        self._run(positions=positions, requery_raises=True)
        data = self._halt_json()
        self.assertEqual(data["symbols"]["AAPL"]["broker_status"], "unknown")

    def test_requery_failure_alert_warns_cannot_verify(self):
        positions = [{"symbol": "AAPL", "unrealized_pl": 0.0, "unrealized_plpc": 0.0}]
        alert_mock = MagicMock()

        def get_positions_mock(client_arg):
            calls = getattr(get_positions_mock, "_calls", 0)
            get_positions_mock._calls = calls + 1
            if calls == 0:
                return positions
            raise Exception("broker offline")

        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=get_positions_mock),
            patch(
                "main.trader.close_position", return_value=OrderResult(OrderStatus.FILLED, "AAPL")
            ),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error", alert_mock),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()
        msg = alert_mock.call_args[0][1]
        self.assertIn("WARNING", msg)
        self.assertIn("verify", msg.lower())

    # ── Regression: order API optimism cannot override broker state ────────────

    def test_kill_switch_does_not_report_complete_when_close_order_submitted_but_position_remains_open(
        self,
    ):
        """Regression test: if close_position returns FILLED but the broker still shows the
        position in get_open_positions, the HALT file must say incomplete and the alert must
        warn the operator. The kill switch may not report liquidation as complete based solely
        on the order API response."""
        positions = [{"symbol": "AAPL", "unrealized_pl": 50.0, "unrealized_plpc": 1.0}]
        remaining_after = [{"symbol": "AAPL", "unrealized_pl": 50.0, "unrealized_plpc": 1.0}]
        close_results = [OrderResult(status=OrderStatus.FILLED, symbol="AAPL", filled_qty=10.0)]
        alert_mock = MagicMock()
        with (
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.get_open_positions", side_effect=[positions, remaining_after]),
            patch("main.trader.close_position", side_effect=close_results),
            patch("main.audit_log.log_position_closed"),
            patch("main.audit_log.log_kill_switch"),
            patch("main.alerts.alert_error", alert_mock),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
        ):
            from main import _run_kill_switch

            _run_kill_switch()

        data = self._halt_json()
        self.assertFalse(
            data["liquidation_complete"], "HALT file must not claim complete when position remains"
        )
        self.assertIn("AAPL", data.get("positions_remaining", []))
        self.assertEqual(data["symbols"]["AAPL"]["broker_status"], "open")

        alert_msg = alert_mock.call_args[0][1]
        self.assertIn("WARNING", alert_msg, "Alert must warn operator when position remains open")


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
            "symbol": symbol,
            "unrealized_plpc": plpc,
            "qty": qty,
            "market_value": market_value,
            "current_price": current_price,
        }

    def test_no_exits_below_threshold(self):
        from main import _handle_partial_exits

        positions = [self._pos("AAPL", plpc=config.PARTIAL_PROFIT_PCT - 1)]
        result = _handle_partial_exits(MagicMock(), positions, dry_run=False)
        self.assertEqual(result, [])

    def test_partial_exit_triggered_at_threshold(self):
        from main import _handle_partial_exits

        with (
            patch("main.trader.cancel_open_orders"),
            patch("main.trader.get_position_meta", return_value={}),
            patch(
                "main.trader.place_partial_sell",
                return_value=OrderResult(
                    OrderStatus.FILLED, "AAPL", broker_order_id="x", filled_qty=5.0
                ),
            ),
            patch("main.trader.record_partial_exit"),
            patch("main.trader.place_trailing_stop"),
            patch("main.audit_log.log_order_placed"),
            patch("main.audit_log.log_order_filled"),
        ):
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
        with (
            patch("main.trader.cancel_open_orders"),
            patch("main.trader.get_position_meta", return_value={}),
            patch(
                "main.trader.place_partial_sell",
                return_value=OrderResult(
                    OrderStatus.FILLED, "AAPL", broker_order_id="x", filled_qty=5.0
                ),
            ),
            patch("main.trader.record_partial_exit"),
            patch("main.trader.place_trailing_stop"),
            patch("main.audit_log.log_order_placed"),
            patch("main.audit_log.log_order_filled"),
        ):
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
        with (
            patch("main.config.validate"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("sys.exit") as mock_exit,
        ):
            from main import run

            run()
        mock_exit.assert_called_with(1)

    def test_exits_on_missing_alpaca_key(self):
        with (
            patch("main.config.validate"),
            patch("main.config.ALPACA_API_KEY", ""),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("sys.exit") as mock_exit,
        ):
            from main import run

            run()
        mock_exit.assert_called_with(1)

    def test_exits_on_missing_anthropic_key(self):
        with (
            patch("main.config.validate"),
            patch("main.config.ALPACA_API_KEY", "valid-key"),
            patch("main.config.ANTHROPIC_API_KEY", ""),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("sys.exit") as mock_exit,
        ):
            from main import run

            run()
        mock_exit.assert_called_with(1)

    def test_returns_without_running_if_lock_held(self):
        with open(self.lock_file, "w"):
            pass
        mock_inner = MagicMock()
        with (
            patch("main.config.validate"),
            patch("main.config.ALPACA_API_KEY", "valid-key"),
            patch("main.config.ANTHROPIC_API_KEY", "valid-key"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("main._run_inner", mock_inner),
        ):
            from main import run

            run()
        mock_inner.assert_not_called()

    def test_releases_lock_after_unhandled_exception(self):
        with (
            patch("main.config.validate"),
            patch("main.config.ALPACA_API_KEY", "valid-key"),
            patch("main.config.ANTHROPIC_API_KEY", "valid-key"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("main._run_inner", side_effect=RuntimeError("unexpected")),
            patch("main.alerts.alert_error"),
        ):
            from main import run

            run()
        self.assertFalse(os.path.exists(self.lock_file))

    def test_alert_sent_on_unhandled_exception(self):
        alert_mock = MagicMock()
        with (
            patch("main.config.validate"),
            patch("main.config.ALPACA_API_KEY", "valid-key"),
            patch("main.config.ANTHROPIC_API_KEY", "valid-key"),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main._lock_file", return_value=self.lock_file),
            patch("main._run_inner", side_effect=RuntimeError("boom")),
            patch("main.alerts.alert_error", alert_mock),
        ):
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
            "main.trader.get_client": MagicMock(),
            "main.trader.is_market_open": True,
            "main.trader.get_account_info": _account(),
            "main.trader.get_open_positions": [],
            "main.trader.reconcile_positions": set(),
            "main.trader.ensure_stops_attached": None,
            "main.trader.get_position_ages": {},
            "main.trader.get_stale_positions": [],
            "main.trader.record_buy": None,
            "main.trader.record_sell": None,
            "main.trader.close_position": OrderResult(status=OrderStatus.FILLED, symbol="X"),
            "main.trader.place_buy_order": OrderResult(
                status=OrderStatus.FILLED, symbol="AAPL", broker_order_id="x", filled_qty=1.0
            ),
            "main.trader.place_trailing_stop": None,
            "main.portfolio_tracker.load_history": [],
            "main.portfolio_tracker.get_track_record": [],
            "main.portfolio_tracker.save_daily_run": _saved_record(),
            "main.portfolio_tracker.print_summary": None,
            "main.portfolio_tracker.save_daily_baseline": None,
            "main.portfolio_tracker.load_daily_baseline": None,
            "main.risk_manager.check_circuit_breaker": (False, 0.0),
            "main.risk_manager.check_daily_loss": (False, 0.0),
            "main.risk_manager.validate_buy_candidates": lambda c, **kw: c,
            "main.position_sizer.risk_budget_size": 500.0,
            "main.position_sizer.get_max_positions": 5,
            "main.market_data.get_vix": 18.0,
            "main.stock_scanner.get_market_regime": _regime(),
            "main.stock_scanner.get_top_movers": [],
            "main.stock_scanner.prefilter_candidates": [],
            "main.macro_calendar.get_macro_risk": _macro(),
            "main.sector_data.get_sector_performance": {},
            "main.sector_data.get_leading_sectors": [],
            "main.get_latest_review": "",
            "main.earnings_calendar.get_earnings_risk_positions": {},
            "main._handle_partial_exits": [],
            "main.market_data.get_market_snapshots": [{"symbol": "AAPL", "current_price": 150.0}],
            "main.options_scanner.get_options_signals": {},
            "main.news_fetcher.fetch_news": {},
            "main.sanitize_headlines": {},
            "main.sentiment_module.get_sentiment": {},
            "main.ai_analyst.get_trading_decisions": _decisions(),
            "main.validate_ai_response": (True, []),
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
            "main.get_day_summary": {
                "date": "2026-01-15",
                "account_before": _account(),
                "account_after": _account(),
                "daily_pnl": 0.0,
            },
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
        # Health check defaults to GREEN so tests focused on buy/sell logic aren't blocked.
        if "main.run_startup_health_check" not in overrides:
            from utils.health import HealthReport, HealthStatus

            defaults["main.run_startup_health_check"] = HealthReport(
                status=HealthStatus.GREEN, issues=[], metrics={}
            )
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
            elif ".config." in target:
                # Config attributes are scalars, not callables — must use new= so the
                # attribute itself holds the value rather than a MagicMock wrapper.
                m = stack.enter_context(patch(target, new=return_val))
            else:
                m = stack.enter_context(patch(target, return_value=return_val))
            mocks[target] = m
        return stack, mocks


class TestRunInnerMarketClosed(RunInnerBase):
    def test_returns_early_without_saving(self):
        save_mock = MagicMock()
        stack, mocks = self._patch_all(
            **{
                "main.trader.is_market_open": False,
                "main.portfolio_tracker.save_daily_run": save_mock,
            }
        )
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
        stack, mocks = self._patch_all(
            **{
                "main.risk_manager.check_circuit_breaker": (True, -15.0),
                "main.trader.place_buy_order": buy_mock,
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_circuit_breaker_sends_alert(self):
        alert_mock = MagicMock()
        stack, mocks = self._patch_all(
            **{
                "main.risk_manager.check_circuit_breaker": (True, -15.0),
                "main.alerts.alert_circuit_breaker": alert_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        alert_mock.assert_called_once()


class TestRunInnerDailyLoss(RunInnerBase):
    def test_daily_loss_closes_positions(self):
        positions = [
            {
                "symbol": "AAPL",
                "unrealized_pl": -500.0,
                "unrealized_plpc": -5.0,
                "qty": 10.0,
                "market_value": 1000.0,
                "current_price": 100.0,
            }
        ]
        close_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        stack, mocks = self._patch_all(
            **{
                "main.risk_manager.check_daily_loss": (True, -6.0),
                "main.trader.get_open_positions": positions,
                "main.trader.close_position": close_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        close_mock.assert_called_once()

    def test_daily_loss_dry_run_skips_close(self):
        positions = [
            {
                "symbol": "AAPL",
                "unrealized_pl": -500.0,
                "unrealized_plpc": -5.0,
                "qty": 10.0,
                "market_value": 1000.0,
                "current_price": 100.0,
            }
        ]
        close_mock = MagicMock()
        stack, mocks = self._patch_all(
            **{
                "main.risk_manager.check_daily_loss": (True, -6.0),
                "main.trader.get_open_positions": positions,
                "main.trader.close_position": close_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=True, mode="open", today="2026-01-15")
        close_mock.assert_not_called()

    def test_daily_loss_sends_alert(self):
        alert_mock = MagicMock()
        stack, mocks = self._patch_all(
            **{
                "main.risk_manager.check_daily_loss": (True, -6.0),
                "main.alerts.alert_daily_loss": alert_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        alert_mock.assert_called_once()


class TestRunInnerOpenAborts(RunInnerBase):
    def test_no_snapshots_aborts_without_saving(self):
        save_mock = MagicMock()
        stack, mocks = self._patch_all(
            **{
                "main.market_data.get_market_snapshots": [],
                "main.portfolio_tracker.save_daily_run": save_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        save_mock.assert_not_called()

    def test_ai_returns_none_aborts_without_saving(self):
        save_mock = MagicMock()
        stack, mocks = self._patch_all(
            **{
                "main.ai_analyst.get_trading_decisions": None,
                "main.portfolio_tracker.save_daily_run": save_mock,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        save_mock.assert_not_called()


class TestRunInnerBuyFiltering(RunInnerBase):
    def test_validation_failure_removes_out_of_universe_candidates(self):
        """Validation failure is fail-closed — all Claude buys are blocked, including out-of-universe candidates."""
        buy_mock = MagicMock()
        # GHOST is not in prefilter_candidates, so it's not in ai_known_symbols
        decisions = _decisions(
            buys=[{"symbol": "GHOST", "confidence": 8, "key_signal": "momentum"}]
        )
        stack, mocks = self._patch_all(
            **{
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.validate_ai_response": (
                    False,
                    ["BUY candidate 'GHOST' not in scanned universe — rejecting"],
                ),
                "main.trader.place_buy_order": buy_mock,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_validation_failure_blocks_all_claude_buys(self):
        """Validation failure is fail-closed — AAPL (valid) is also blocked when GHOST (invalid) taints the response."""
        buy_mock = MagicMock()
        decisions = _decisions(
            buys=[
                {"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"},
                {"symbol": "GHOST", "confidence": 7, "key_signal": "momentum"},
            ]
        )
        stack, mocks = self._patch_all(
            **{
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.validate_ai_response": (
                    False,
                    ["BUY candidate 'GHOST' not in scanned universe — rejecting"],
                ),
                "main.trader.place_buy_order": buy_mock,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_structural_validation_failure_blocks_claude_driven_sells(self):
        """Structural (schema) errors zero position_decisions — Claude-recommended sells are blocked."""
        sell_mock = MagicMock()
        decisions = _decisions(
            buys=[{"symbol": "MSFT", "confidence": 8, "key_signal": "momentum"}],
            sells=[
                {
                    "symbol": "AAPL",
                    "action": "SELL",
                    "confidence": 7,
                    "reasoning": "momentum weakening",
                }
            ],
        )
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": [
                    {"symbol": "AAPL", "qty": 10, "unrealized_plpc": -0.02}
                ],
                "main.ai_analyst.get_trading_decisions": decisions,
                # Structural error (Pydantic field path) — not a "BUY candidate '" prefix
                "main.validate_ai_response": (
                    False,
                    ["position_decisions → 0 → action: Input should be 'HOLD' or 'SELL'"],
                ),
                "main.trader.close_position": sell_mock,
            }
        )
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
            sells=[
                {
                    "symbol": "AAPL",
                    "action": "SELL",
                    "confidence": 7,
                    "reasoning": "momentum weakening",
                }
            ],
        )
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": [
                    {"symbol": "AAPL", "qty": 10, "unrealized_plpc": -0.02}
                ],
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.validate_ai_response": (
                    False,
                    ["BUY candidate 'GHOST' not in scanned universe — rejecting"],
                ),
                "main.trader.close_position": sell_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        # AAPL sell must still fire despite the invalid buy candidate
        sell_mock.assert_called_once()

    def test_buy_candidates_pydantic_error_treated_as_domain_not_structural(self):
        """buy_candidates field errors (e.g. reasoning too long) are buy-domain errors —
        sells are still executed, only buys are blocked."""
        sell_mock = MagicMock()
        sell_mock.return_value = OrderResult(status=OrderStatus.FILLED, symbol="AAPL")
        decisions = _decisions(
            buys=[{"symbol": "MSFT", "confidence": 8, "key_signal": "momentum"}],
            sells=[{"symbol": "AAPL", "action": "SELL", "confidence": 7, "reasoning": "weak"}],
        )
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": [
                    {"symbol": "AAPL", "qty": 10, "unrealized_plpc": -0.02}
                ],
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.validate_ai_response": (
                    False,
                    ["buy_candidates → 0 → reasoning: String should have at most 2000 characters"],
                ),
                "main.trader.close_position": sell_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        sell_mock.assert_called_once()

    def test_bearish_regime_blocks_buys(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.stock_scanner.get_market_regime": _regime(bearish=True),
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": buy_mock,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_macro_risk_blocks_buys(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.macro_calendar.get_macro_risk": _macro(high_risk=True),
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": buy_mock,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_dry_run_does_not_place_buy_orders(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": buy_mock,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.trader.get_account_info": _account(100_000, 50_000),
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=True, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()

    def test_sell_executed_in_open_mode(self):
        close_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        positions = [
            {
                "symbol": "AAPL",
                "unrealized_pl": 100.0,
                "unrealized_plpc": 2.0,
                "qty": 10.0,
                "market_value": 1000.0,
                "current_price": 100.0,
            }
        ]
        decisions = _decisions(sells=[{"symbol": "AAPL", "action": "SELL", "reasoning": "stale"}])
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": positions,
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.close_position": close_mock,
                "main.trader.get_position_meta": MagicMock(
                    return_value={"signal": "m", "regime": "X", "confidence": 7}
                ),
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        close_mock.assert_called()


class TestLockFile(unittest.TestCase):
    """Line 51: _lock_file() uses config.today_et() to build the path."""

    def test_lock_file_contains_today_date(self):
        from main import _lock_file

        with (
            patch("main.config.today_et", return_value=config.today_et()),
            patch("main.config.LOG_DIR", "/tmp/testlogdir"),
        ):
            result = _lock_file()
        self.assertIn(config.today_et().isoformat(), result)
        self.assertIn(".lock_", result)


class TestAcquireLockStaleThenSecondFailure(LockBase):
    """Lines 70-72: stale lock removed, but second O_EXCL open also fails (race)."""

    def test_stale_lock_removal_race_returns_none(self):
        """If another process grabs the lock between removal and re-creation, return None."""
        from main import _acquire_lock

        os.makedirs(self.tmpdir, exist_ok=True)
        with open(self.lock_file, "w"):
            pass
        old_time = time.time() - 7200  # stale
        os.utime(self.lock_file, (old_time, old_time))

        real_open = os.open

        call_count = [0]

        def patched_open(path, flags, mode=0o644):
            call_count[0] += 1
            if call_count[0] == 2:
                raise FileExistsError("race condition")
            return real_open(path, flags, mode)

        with patch("os.open", side_effect=patched_open):
            result = _acquire_lock()
        self.assertIsNone(result)


class TestHandlePartialExitsAlreadyTaken(unittest.TestCase):
    """Lines 212-213: partial exit already taken — skip without placing order."""

    def _pos(self, symbol, plpc, qty=10.0, market_value=1000.0, current_price=100.0):
        return {
            "symbol": symbol,
            "unrealized_plpc": plpc,
            "qty": qty,
            "market_value": market_value,
            "current_price": current_price,
        }

    def test_skips_when_partial_exit_already_taken(self):
        from main import _handle_partial_exits

        sell_mock = unittest.mock.MagicMock()
        # meta includes partial_exit_taken_at — should trigger the skip branch
        with (
            patch(
                "main.trader.get_position_meta",
                return_value={"partial_exit_taken_at": "2026-01-15T10:00:00+00:00"},
            ),
            patch("main.trader.place_partial_sell", sell_mock),
        ):
            result = _handle_partial_exits(
                MagicMock(),
                [self._pos("AAPL", plpc=config.PARTIAL_PROFIT_PCT + 5)],
                dry_run=False,
            )
        # No trade should be executed — the position was already partially exited
        self.assertEqual(result, [])
        sell_mock.assert_not_called()


class TestRunInnerEarningsExit(RunInnerBase):
    """Lines 366-383: earnings exit execution path — symbol held AND has earnings risk."""

    def test_earnings_exit_closes_held_position(self):
        close_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        positions = [
            {
                "symbol": "AAPL",
                "unrealized_pl": 50.0,
                "unrealized_plpc": 1.0,
                "qty": 10.0,
                "market_value": 1500.0,
                "current_price": 150.0,
            }
        ]
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": positions,
                "main.earnings_calendar.get_earnings_risk_positions": {"AAPL": "2026-01-17"},
                "main.trader.close_position": close_mock,
                "main.trader.get_position_meta": MagicMock(
                    return_value={"signal": "momentum", "regime": "BULL_TRENDING", "confidence": 8}
                ),
                "main._handle_partial_exits": [],
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        close_mock.assert_called()
        # Verify the call was for AAPL
        self.assertEqual(close_mock.call_args[0][1], "AAPL")

    def test_earnings_exit_not_triggered_for_unheld_symbol(self):
        close_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        # AAPL has earnings risk but is NOT held
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": [],
                "main.earnings_calendar.get_earnings_risk_positions": {"AAPL": "2026-01-17"},
                "main.trader.close_position": close_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        close_mock.assert_not_called()

    def test_earnings_exit_failure_logs_error(self):
        """Line 383: earnings close returns non-success → error logged."""
        close_mock = MagicMock(
            return_value=OrderResult(
                status=OrderStatus.REJECTED, symbol="AAPL", rejection_reason="api error"
            )
        )
        positions = [
            {
                "symbol": "AAPL",
                "unrealized_pl": 50.0,
                "unrealized_plpc": 1.0,
                "qty": 10.0,
                "market_value": 1500.0,
                "current_price": 150.0,
            }
        ]
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": positions,
                "main.earnings_calendar.get_earnings_risk_positions": {"AAPL": "2026-01-17"},
                "main.trader.close_position": close_mock,
                "main._handle_partial_exits": [],
            }
        )
        with stack, self.assertLogs("main", level="ERROR") as cm:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        log_output = "\n".join(cm.output)
        self.assertIn("Earnings exit FAILED", log_output)
        self.assertIn("AAPL", log_output)


class TestRunInnerOptionsSignals(RunInnerBase):
    """Line 425: options_sigs non-empty → logger.info('Options signals fetched...')."""

    def test_options_signals_fetched_log_when_non_empty(self):
        stack, mocks = self._patch_all(
            **{
                "main.options_scanner.get_options_signals": {
                    "AAPL": {"put_call_ratio": 0.5, "unusual_calls": True}
                },
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
            }
        )
        with stack, self.assertLogs("main", level="INFO") as cm:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        log_output = "\n".join(cm.output)
        self.assertIn("Options signals fetched", log_output)


class TestRunInnerStaleTimeBased(RunInnerBase):
    """Lines 508-510: stale symbol NOT in position_decisions → time-based reason."""

    def test_stale_symbol_without_ai_decision_uses_time_based_reason(self):
        close_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        positions = [
            {
                "symbol": "AAPL",
                "unrealized_pl": 50.0,
                "unrealized_plpc": 1.0,
                "qty": 10.0,
                "market_value": 1500.0,
                "current_price": 150.0,
            }
        ]
        # get_position_meta must return a signal so the stale threshold can be looked up
        meta_mock = MagicMock(
            return_value={"signal": "momentum", "regime": "BULL_TRENDING", "confidence": 7}
        )
        # position_ages must be high enough to be >= MAX_HOLD_DAYS
        ages = {"AAPL": config.MAX_HOLD_DAYS + 10}
        stack, mocks = self._patch_all(
            **{
                # Return positions consistently across all calls inside _run_inner
                "main.trader.get_open_positions": positions,
                # No sell decision from AI for AAPL
                "main.ai_analyst.get_trading_decisions": _decisions(sells=[]),
                "main.trader.get_position_ages": ages,
                "main.trader.close_position": close_mock,
                "main.trader.get_position_meta": meta_mock,
                "main.market_data.get_market_snapshots": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        # AAPL should be closed via the time-based path
        close_mock.assert_called()


class TestRunInnerSellFailed(RunInnerBase):
    """Lines 527-531: sell execution where result is NOT success → error logged."""

    def test_failed_sell_logs_error_and_alert(self):
        close_mock = MagicMock(
            return_value=OrderResult(
                status=OrderStatus.REJECTED, symbol="AAPL", rejection_reason="not found"
            )
        )
        alert_mock = MagicMock()
        positions = [
            {
                "symbol": "AAPL",
                "unrealized_pl": 50.0,
                "unrealized_plpc": 1.0,
                "qty": 10.0,
                "market_value": 1500.0,
                "current_price": 150.0,
            }
        ]
        decisions = _decisions(sells=[{"symbol": "AAPL", "action": "SELL", "reasoning": "stale"}])
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": positions,
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.close_position": close_mock,
                "main.alerts.alert_error": alert_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        alert_mock.assert_called()
        # The alert should mention the sell failure
        call_args_str = str(alert_mock.call_args)
        self.assertIn("AAPL", call_args_str)


class TestRunInnerDryRunSell(RunInnerBase):
    """Lines 530-531: dry_run=True sell → executed_symbols.add + all_trades.append."""

    def test_dry_run_sell_records_trade_without_closing(self):
        close_mock = MagicMock()
        positions = [
            {
                "symbol": "AAPL",
                "unrealized_pl": 50.0,
                "unrealized_plpc": 1.0,
                "qty": 10.0,
                "market_value": 1500.0,
                "current_price": 150.0,
            }
        ]
        decisions = _decisions(sells=[{"symbol": "AAPL", "action": "SELL", "reasoning": "stale"}])
        save_mock = MagicMock(return_value=_saved_record())
        stack, mocks = self._patch_all(
            **{
                "main.trader.get_open_positions": positions,
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.close_position": close_mock,
                "main.portfolio_tracker.save_daily_run": save_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=True, mode="open", today="2026-01-15")
        close_mock.assert_not_called()
        # save_daily_run should have been called with a trade recorded
        call_kwargs = save_mock.call_args[1] if save_mock.call_args else {}
        trades = call_kwargs.get("trades_executed", [])
        symbols = [t["symbol"] for t in trades]
        self.assertIn("AAPL", symbols)


class TestRunInnerRegimeMaxOrders(RunInnerBase):
    """Lines 561-562: HIGH_VOL regime → regime_max_orders = 2, regime_conf_bump = 1."""

    def test_high_vol_regime_takes_high_vol_branch(self):
        """Lines 561-562: HIGH_VOL path sets regime_max_orders=2 and conf_bump=1."""
        buy_mock = MagicMock(
            return_value=OrderResult(
                status=OrderStatus.FILLED, symbol="AAPL", broker_order_id="x", filled_qty=1.0
            )
        )
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.stock_scanner.get_market_regime": {"regime": "HIGH_VOL", "is_bearish": False},
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": buy_mock,
                "main.market_data.get_market_snapshots": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.trader.get_account_info": _account(100_000, 50_000),
                "main.trader.get_open_positions": [],
                "main.position_sizer.get_max_positions": 5,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_called()

    def test_normal_regime_uses_max_orders_per_run(self):
        buy_mock = MagicMock(
            return_value=OrderResult(
                status=OrderStatus.FILLED, symbol="AAPL", broker_order_id="x", filled_qty=1.0
            )
        )
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        # Use a normal regime (not CHOPPY, not HIGH_VOL)
        stack, mocks = self._patch_all(
            **{
                "main.stock_scanner.get_market_regime": {
                    "regime": "BULL_TRENDING",
                    "is_bearish": False,
                },
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": buy_mock,
                "main.market_data.get_market_snapshots": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.trader.get_account_info": _account(100_000, 50_000),
                "main.trader.get_open_positions": [],
                "main.position_sizer.get_max_positions": 5,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_called()


class TestRunInnerPreTradeCheckFailed(RunInnerBase):
    """Lines 608-609: pre-trade check fails → warning logged and buy skipped."""

    def test_pre_trade_check_failure_skips_buy(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.stock_scanner.get_market_regime": {
                    "regime": "BULL_TRENDING",
                    "is_bearish": False,
                },
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": buy_mock,
                "main.check_pre_trade": (False, "daily notional exceeded"),
                "main.market_data.get_market_snapshots": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.trader.get_account_info": _account(100_000, 50_000),
                "main.trader.get_open_positions": [],
                "main.position_sizer.get_max_positions": 5,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()


class TestRunInnerStopFailedAfterBuy(RunInnerBase):
    """Stop placement failure after buy → error logged."""

    def test_stop_failure_after_buy_logs_error(self):
        alert_mock = MagicMock()
        buy_result = OrderResult(
            status=OrderStatus.FILLED, symbol="AAPL", broker_order_id="x", filled_qty=5.0
        )
        stop_result = OrderResult(
            status=OrderStatus.STOP_FAILED, symbol="AAPL", rejection_reason="api error"
        )
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.stock_scanner.get_market_regime": {
                    "regime": "BULL_TRENDING",
                    "is_bearish": False,
                },
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": MagicMock(return_value=buy_result),
                "main.trader.place_trailing_stop": MagicMock(return_value=stop_result),
                "main.market_data.get_market_snapshots": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.trader.get_account_info": _account(100_000, 50_000),
                "main.trader.get_open_positions": [],
                "main.position_sizer.get_max_positions": 5,
                "main.alerts.alert_error": alert_mock,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        # alert_error should have been called for STOP FAILED
        alert_calls = [str(c) for c in alert_mock.call_args_list]
        self.assertTrue(
            any("STOP" in c for c in alert_calls), f"Expected STOP FAILED alert, got: {alert_calls}"
        )


class TestRunInnerBuyZeroFilledQty(RunInnerBase):
    """Lines 636-637: successful buy with zero filled_qty → no stop placed."""

    def test_zero_filled_qty_does_not_place_stop(self):
        stop_mock = MagicMock()
        # filled_qty=0.0 means no shares were actually filled
        buy_result = OrderResult(
            status=OrderStatus.FILLED, symbol="AAPL", broker_order_id="x", filled_qty=0.0
        )
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.stock_scanner.get_market_regime": {
                    "regime": "BULL_TRENDING",
                    "is_bearish": False,
                },
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": MagicMock(return_value=buy_result),
                "main.trader.place_trailing_stop": stop_mock,
                "main.market_data.get_market_snapshots": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.trader.get_account_info": _account(100_000, 50_000),
                "main.trader.get_open_positions": [],
                "main.position_sizer.get_max_positions": 5,
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        # No stop should be placed when filled_qty == 0
        stop_mock.assert_not_called()


class TestRunInnerSubShareGuard(RunInnerBase):
    """Sub-share guard: notional ≥ 1.0 but notional/price < 1 share → skip and warn."""

    def test_sub_share_position_skips_buy(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.stock_scanner.get_market_regime": {
                    "regime": "BULL_TRENDING",
                    "is_bearish": False,
                },
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": buy_mock,
                # $5 notional at $10/share = 0.5 shares — sub-share, cannot be stop-protected
                "main.position_sizer.risk_budget_size": 5.0,
                "main.market_data.get_market_snapshots": [
                    {"symbol": "AAPL", "current_price": 10.0}
                ],
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 10.0}
                ],
                "main.trader.get_account_info": _account(100_000, 50_000),
                "main.trader.get_open_positions": [],
                "main.position_sizer.get_max_positions": 5,
            }
        )
        with stack, self.assertLogs("main", level="WARNING") as cm:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()
        self.assertTrue(any("sub-share" in line for line in cm.output))


class TestRunInnerNotionalTooSmall(RunInnerBase):
    """Line 647: notional < 1.0 → logger.warning('Skipping ...: $X too small')."""

    def test_too_small_notional_logs_warning_and_skips(self):
        buy_mock = MagicMock()
        decisions = _decisions(buys=[{"symbol": "AAPL", "confidence": 8, "key_signal": "momentum"}])
        stack, mocks = self._patch_all(
            **{
                "main.stock_scanner.get_market_regime": {
                    "regime": "BULL_TRENDING",
                    "is_bearish": False,
                },
                "main.ai_analyst.get_trading_decisions": decisions,
                "main.trader.place_buy_order": buy_mock,
                # risk_budget_size returns tiny notional below 1.0
                "main.position_sizer.risk_budget_size": 0.50,
                "main.market_data.get_market_snapshots": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.trader.get_account_info": _account(100_000, 50_000),
                "main.trader.get_open_positions": [],
                "main.position_sizer.get_max_positions": 5,
            }
        )
        with stack, self.assertLogs("main", level="WARNING") as cm:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()
        log_output = "\n".join(cm.output)
        self.assertIn("too small", log_output)


class TestMaxPositionsCappedByConfig(RunInnerBase):
    """MAX_POSITIONS from config must always cap get_max_positions() — never exceeded."""

    def test_config_max_positions_caps_sizer(self):
        """get_max_positions returns 3 but config.MAX_POSITIONS=2 — only 2 slots used."""
        buy_mock = MagicMock(
            return_value=OrderResult(
                status=OrderStatus.FILLED, symbol="AAPL", broker_order_id="x", filled_qty=1.0
            )
        )
        stop_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        stack, mocks = self._patch_all(
            **{
                "main.trader.place_buy_order": buy_mock,
                "main.trader.place_trailing_stop": stop_mock,
                "main.position_sizer.get_max_positions": 3,
                "main.config.MAX_POSITIONS": 2,
                "main.trader.get_open_positions": [
                    {
                        "symbol": "MSFT",
                        "market_value": 50.0,
                        "unrealized_plpc": 0.0,
                        "unrealized_pl": 0.0,
                        "qty": 1,
                        "current_price": 50.0,
                    },
                    {
                        "symbol": "GOOG",
                        "market_value": 50.0,
                        "unrealized_plpc": 0.0,
                        "unrealized_pl": 0.0,
                        "qty": 1,
                        "current_price": 50.0,
                    },
                ],
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.ai_analyst.get_trading_decisions": _decisions(
                    buys=[
                        {
                            "symbol": "AAPL",
                            "confidence": 8,
                            "reasoning": "Strong breakout signal above key resistance.",
                            "key_signal": "momentum",
                        }
                    ]
                ),
                "main.validate_ai_response": (True, []),
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        # 2 positions already open and MAX_POSITIONS=2 → 0 slots → no buy
        buy_mock.assert_not_called()


class TestExperimentDrawdownCap(RunInnerBase):
    """MAX_EXPERIMENT_DRAWDOWN_USD must block new buys when exceeded."""

    def test_drawdown_cap_blocks_buys(self):
        buy_mock = MagicMock()
        stack, mocks = self._patch_all(
            **{
                "main.trader.place_buy_order": buy_mock,
                # Start equity $1000, current $940 → $60 loss exceeds $50 cap
                "main.load_experiment_baseline": 1000.0,
                "main.trader.get_account_info": _account(940, 300),
                "main.config.MAX_EXPERIMENT_DRAWDOWN_USD": 50.0,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.ai_analyst.get_trading_decisions": _decisions(
                    buys=[
                        {
                            "symbol": "AAPL",
                            "confidence": 8,
                            "reasoning": "Strong breakout signal above key resistance.",
                            "key_signal": "momentum",
                        }
                    ]
                ),
                "main.validate_ai_response": (True, []),
            }
        )
        with stack, self.assertLogs("main", level="CRITICAL") as cm:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_not_called()
        self.assertTrue(any("drawdown" in line.lower() for line in cm.output))

    def test_drawdown_cap_not_triggered_when_under_limit(self):
        buy_mock = MagicMock(
            return_value=OrderResult(
                status=OrderStatus.FILLED, symbol="AAPL", broker_order_id="x", filled_qty=1.0
            )
        )
        stop_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="AAPL"))
        stack, mocks = self._patch_all(
            **{
                "main.trader.place_buy_order": buy_mock,
                "main.trader.place_trailing_stop": stop_mock,
                # $10 loss well under $50 cap
                "main.load_experiment_baseline": 1000.0,
                "main.trader.get_account_info": _account(990, 500),
                "main.config.MAX_EXPERIMENT_DRAWDOWN_USD": 50.0,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.ai_analyst.get_trading_decisions": _decisions(
                    buys=[
                        {
                            "symbol": "AAPL",
                            "confidence": 8,
                            "reasoning": "Strong breakout signal above key resistance.",
                            "key_signal": "momentum",
                        }
                    ]
                ),
                "main.validate_ai_response": (True, []),
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        buy_mock.assert_called_once()


class TestPartialTimeoutImmediateStopCheck(RunInnerBase):
    """PARTIAL or TIMEOUT buy result must trigger immediate ensure_stops_attached."""

    def _run_ambiguous(self, status: OrderStatus):
        ensure_mock = MagicMock(return_value=True)
        buy_mock = MagicMock(
            return_value=OrderResult(
                status=status, symbol="AAPL", broker_order_id="x", filled_qty=0.5
            )
        )
        stack, mocks = self._patch_all(
            **{
                "main.trader.place_buy_order": buy_mock,
                "main.trader.ensure_stops_attached": ensure_mock,
                "main.stock_scanner.prefilter_candidates": [
                    {"symbol": "AAPL", "current_price": 150.0}
                ],
                "main.ai_analyst.get_trading_decisions": _decisions(
                    buys=[
                        {
                            "symbol": "AAPL",
                            "confidence": 8,
                            "reasoning": "Strong breakout signal above key resistance.",
                            "key_signal": "momentum",
                        }
                    ]
                ),
                "main.validate_ai_response": (True, []),
            }
        )
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        return ensure_mock

    def test_partial_fill_triggers_immediate_stop_check(self):
        ensure_mock = self._run_ambiguous(OrderStatus.PARTIAL)
        # ensure_stops_attached called at least once during the buy loop (not only end-of-run)
        self.assertGreaterEqual(ensure_mock.call_count, 1)

    def test_timeout_triggers_immediate_stop_check(self):
        ensure_mock = self._run_ambiguous(OrderStatus.TIMEOUT)
        self.assertGreaterEqual(ensure_mock.call_count, 1)


class TestUnexpectedBrokerPositionsHalt(RunInnerBase):
    """Unexpected broker positions (not in local DB) must halt in live mode."""

    def test_unexpected_positions_write_halt_and_exit_in_live_mode(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        halt_file = os.path.join(tmpdir, ".HALTED")
        stack, mocks = self._patch_all(
            **{
                # reconcile_positions returns unexpected symbol set
                "main.trader.reconcile_positions": {"XUNKNOWN"},
                "main.config.IS_PAPER": False,
                "main.config.HALT_FILE": halt_file,
                "main.config.LOG_DIR": tmpdir,
            }
        )
        with stack, self.assertRaises(SystemExit):
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")
        import shutil as _shutil

        _shutil.rmtree(tmpdir, ignore_errors=True)

    def test_unexpected_positions_not_fatal_in_paper_mode(self):
        """Paper mode should log but not halt on unexpected positions."""
        stack, mocks = self._patch_all(
            **{
                "main.trader.reconcile_positions": {"XUNKNOWN"},
                "main.config.IS_PAPER": True,
            }
        )
        # Should not raise SystemExit
        with stack:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-01-15")


if __name__ == "__main__":
    unittest.main()

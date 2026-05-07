"""Tests for execution/trader.py — order placement, fill polling, reconciliation."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from models import OrderResult, OrderStatus


def _make_trailing_stop_order(symbol, status="new"):
    o = MagicMock()
    o.symbol = symbol
    o.status = status
    o.id = f"stop-{symbol}"
    return o


def _mock_order(order_id="order-123", status="new", filled_qty=None, filled_avg_price="0.0"):
    o = MagicMock()
    o.id = order_id
    o.status = status
    o.filled_qty = filled_qty
    o.filled_avg_price = filled_avg_price
    return o


def _mock_position(symbol, qty, current_price=150.0):
    p = MagicMock()
    p.symbol = symbol
    p.qty = str(qty)
    p.avg_entry_price = str(current_price * 0.95)
    p.current_price = str(current_price)
    p.unrealized_pl = str(qty * current_price * 0.05)
    p.unrealized_plpc = str(0.05)
    p.market_value = str(qty * current_price)
    return p


def _mock_account(portfolio_value=100_000, cash=20_000, buying_power=40_000, equity=100_000):
    a = MagicMock()
    a.portfolio_value = str(portfolio_value)
    a.cash = str(cash)
    a.buying_power = str(buying_power)
    a.equity = str(equity)
    return a


def _meta_patcher(tmpdir):
    """Return patchers that isolate trader metadata tests to a temp SQLite DB."""
    import utils.db as db_module

    db_path = os.path.join(tmpdir, "test.db")
    return [
        patch.object(db_module, "_DB_PATH", db_path),
        patch.object(db_module, "_initialized", False),
        patch.object(db_module, "_migrate_json_state", lambda: None),
    ]


class TestOrderResult(unittest.TestCase):
    def test_is_success_true_for_filled(self):
        r = OrderResult(status=OrderStatus.FILLED, symbol="AAPL")
        self.assertTrue(r.is_success)

    def test_is_success_false_for_rejected(self):
        r = OrderResult(status=OrderStatus.REJECTED, symbol="AAPL", rejection_reason="API error")
        self.assertFalse(r.is_success)

    def test_is_success_false_for_timeout(self):
        r = OrderResult(status=OrderStatus.TIMEOUT, symbol="AAPL")
        self.assertFalse(r.is_success)

    def test_is_success_false_for_stop_failed(self):
        r = OrderResult(status=OrderStatus.STOP_FAILED, symbol="AAPL")
        self.assertFalse(r.is_success)

    def test_is_success_false_for_unprotected(self):
        r = OrderResult(status=OrderStatus.UNPROTECTED, symbol="AAPL")
        self.assertFalse(r.is_success)

    def test_default_values(self):
        r = OrderResult(status=OrderStatus.FILLED, symbol="MSFT")
        self.assertAlmostEqual(r.filled_qty, 0.0)
        self.assertAlmostEqual(r.filled_avg_price, 0.0)
        self.assertIsNone(r.broker_order_id)
        self.assertIsNone(r.rejection_reason)
        self.assertIsNone(r.stop_order_id)

    def test_all_statuses_present(self):
        expected = {"FILLED", "PARTIAL", "TIMEOUT", "REJECTED", "STOP_FAILED", "UNPROTECTED"}
        actual = {s.value for s in OrderStatus}
        self.assertEqual(actual, expected)


class TestGetAccountInfo(unittest.TestCase):
    def test_returns_float_values(self):
        from execution.trader import get_account_info

        client = MagicMock()
        client.get_account.return_value = _mock_account(100_000, 20_000, 40_000, 100_000)
        result = get_account_info(client)
        self.assertAlmostEqual(result["portfolio_value"], 100_000.0)
        self.assertAlmostEqual(result["cash"], 20_000.0)
        self.assertAlmostEqual(result["buying_power"], 40_000.0)

    def test_returns_required_keys(self):
        from execution.trader import get_account_info

        client = MagicMock()
        client.get_account.return_value = _mock_account()
        result = get_account_info(client)
        for key in ["portfolio_value", "cash", "buying_power", "equity"]:
            self.assertIn(key, result)


class TestGetOpenPositions(unittest.TestCase):
    def test_returns_list_of_position_dicts(self):
        from execution.trader import get_open_positions

        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("AAPL", 10, 180.0)]
        result = get_open_positions(client)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "AAPL")

    def test_unrealised_plpc_converted_to_percent(self):
        from execution.trader import get_open_positions

        pos = _mock_position("AAPL", 10, 180.0)
        pos.unrealized_plpc = "0.05"  # 5% as decimal
        client = MagicMock()
        client.get_all_positions.return_value = [pos]
        result = get_open_positions(client)
        self.assertAlmostEqual(result[0]["unrealized_plpc"], 5.0)

    def test_empty_positions_returns_empty_list(self):
        from execution.trader import get_open_positions

        client = MagicMock()
        client.get_all_positions.return_value = []
        result = get_open_positions(client)
        self.assertEqual(result, [])


class TestPlaceBuyOrder(unittest.TestCase):
    def setUp(self):

        self.tmpdir = tempfile.mkdtemp()
        self.patchers = _meta_patcher(self.tmpdir)
        for p in self.patchers:
            p.start()
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)

    def test_returns_none_for_tiny_order(self):
        from execution.trader import place_buy_order

        client = MagicMock()
        result = place_buy_order(client, "AAPL", 0.50)
        self.assertIsNone(result)
        client.submit_order.assert_not_called()

    def test_places_order_for_valid_notional(self):
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-abc")
        with patch("execution.trader.wait_for_fill", return_value=(28.5, 175.25)):
            result = place_buy_order(client, "AAPL", 5_000.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.symbol, "AAPL")
        self.assertAlmostEqual(result.filled_qty, 28.5)
        self.assertEqual(result.status, OrderStatus.FILLED)

    def test_buy_order_captures_fill_avg_price(self):
        """OrderResult.filled_avg_price is populated from the Alpaca fill response."""
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-abc")
        with patch("execution.trader.wait_for_fill", return_value=(10.0, 234.56)):
            result = place_buy_order(client, "AAPL", 3_000.0)
        self.assertAlmostEqual(result.filled_avg_price, 234.56)

    def test_returns_rejected_on_api_error(self):
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.side_effect = Exception("insufficient funds")
        result = place_buy_order(client, "AAPL", 5_000.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("insufficient funds", result.rejection_reason)

    def test_returns_timeout_when_fill_does_not_arrive(self):
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-abc")
        final_order = MagicMock()
        final_order.status = "pending_new"
        final_order.filled_qty = None
        client.get_order_by_id.return_value = final_order
        with patch("execution.trader.wait_for_fill", return_value=None):
            result = place_buy_order(client, "AAPL", 5_000.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrderStatus.TIMEOUT)
        self.assertEqual(result.broker_order_id, "order-abc")

    def test_buy_partial_fill_detected_after_timeout(self):
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-partial")
        partial = MagicMock()
        partial.status = "partially_filled"
        partial.filled_qty = 5.0
        client.get_order_by_id.return_value = partial
        with patch("execution.trader.wait_for_fill", return_value=None):
            result = place_buy_order(client, "AAPL", 5_000.0)
        self.assertEqual(result.status, OrderStatus.PARTIAL)
        self.assertAlmostEqual(result.filled_qty, 5.0)

    def test_buy_filled_detected_on_final_check(self):
        """wait_for_fill times out but get_order_by_id shows 'filled' → FILLED result.

        Covers pre-market orders that fill at market open after the wait window closes.
        """
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-late")
        final_order = MagicMock()
        final_order.status = "filled"
        final_order.filled_qty = 23.5
        final_order.filled_avg_price = 420.69
        client.get_order_by_id.return_value = final_order
        with patch("execution.trader.wait_for_fill", return_value=None):
            result = place_buy_order(client, "NVDA", 10_000.0)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertAlmostEqual(result.filled_qty, 23.5)
        self.assertAlmostEqual(result.filled_avg_price, 420.69)
        self.assertEqual(result.broker_order_id, "order-late")

    def test_buy_client_order_id_is_symbol_date_stable(self):
        """client_order_id must be stable across same-day reruns for idempotency."""
        from config import today_et
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-xyz")
        with patch("execution.trader.wait_for_fill", return_value=(10.0, 0.0)):
            place_buy_order(client, "AAPL", 3_000.0)
        submitted = client.submit_order.call_args[0][0]
        expected = f"ib-AAPL-BUY-{today_et().isoformat()}"
        self.assertEqual(submitted.client_order_id, expected)

    def test_buy_client_order_id_always_set(self):
        """client_order_id is always set regardless of run_id arg."""
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-xyz")
        with patch("execution.trader.wait_for_fill", return_value=(10.0, 0.0)):
            place_buy_order(client, "AAPL", 3_000.0)
        submitted = client.submit_order.call_args[0][0]
        self.assertTrue(submitted.client_order_id)

    def test_order_includes_broker_order_id(self):
        from execution.trader import place_buy_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-xyz")
        with patch("execution.trader.wait_for_fill", return_value=(10.0, 0.0)):
            result = place_buy_order(client, "MSFT", 3_000.0)
        self.assertEqual(result.broker_order_id, "order-xyz")
        self.assertTrue(result.is_success)


class TestWaitForFill(unittest.TestCase):
    def test_returns_qty_when_immediately_filled(self):
        from execution.trader import wait_for_fill

        filled_order = _mock_order(status="filled", filled_qty="28.571", filled_avg_price="175.50")
        client = MagicMock()
        client.get_order_by_id.return_value = filled_order
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertIsNotNone(result)
        qty, avg_price = result
        self.assertAlmostEqual(qty, 28.571)
        self.assertAlmostEqual(avg_price, 175.50)

    def test_returns_avg_price_from_fill(self):
        from execution.trader import wait_for_fill

        filled_order = _mock_order(status="filled", filled_qty="10.0", filled_avg_price="234.56")
        client = MagicMock()
        client.get_order_by_id.return_value = filled_order
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-abc", max_wait=3)
        self.assertIsNotNone(result)
        _, avg_price = result
        self.assertAlmostEqual(avg_price, 234.56)

    def test_returns_zero_avg_price_when_field_missing(self):
        from execution.trader import wait_for_fill

        filled_order = _mock_order(status="filled", filled_qty="5.0")
        filled_order.filled_avg_price = None
        client = MagicMock()
        client.get_order_by_id.return_value = filled_order
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-abc", max_wait=3)
        self.assertIsNotNone(result)
        _, avg_price = result
        self.assertAlmostEqual(avg_price, 0.0)

    def test_returns_none_on_timeout(self):
        from execution.trader import wait_for_fill

        pending_order = _mock_order(status="new", filled_qty=None)
        client = MagicMock()
        client.get_order_by_id.return_value = pending_order
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertIsNone(result)

    def test_does_not_return_early_on_partially_filled(self):
        """partially_filled is not a terminal success — must keep polling until filled."""
        from execution.trader import wait_for_fill

        calls = [
            _mock_order(status="partially_filled", filled_qty="5.0"),
            _mock_order(status="partially_filled", filled_qty="8.0"),
            _mock_order(status="filled", filled_qty="10.0"),
        ]
        client = MagicMock()
        client.get_order_by_id.side_effect = calls
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 10.0)
        self.assertEqual(client.get_order_by_id.call_count, 3)

    def test_returns_none_when_partially_filled_then_times_out(self):
        from execution.trader import wait_for_fill

        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order(
            status="partially_filled", filled_qty="5.0"
        )
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertIsNone(result)

    def test_exits_early_on_rejected(self):
        from execution.trader import wait_for_fill

        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order(status="rejected", filled_qty=None)
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=30)
        self.assertIsNone(result)
        self.assertEqual(client.get_order_by_id.call_count, 1)

    def test_exits_early_on_cancelled(self):
        from execution.trader import wait_for_fill

        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order(status="cancelled", filled_qty=None)
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=30)
        self.assertIsNone(result)
        self.assertEqual(client.get_order_by_id.call_count, 1)

    def test_exits_early_on_expired(self):
        from execution.trader import wait_for_fill

        client = MagicMock()
        client.get_order_by_id.return_value = _mock_order(status="expired", filled_qty=None)
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=30)
        self.assertIsNone(result)
        self.assertEqual(client.get_order_by_id.call_count, 1)

    def test_retries_after_api_error_and_succeeds(self):
        from execution.trader import wait_for_fill

        client = MagicMock()
        client.get_order_by_id.side_effect = [
            Exception("transient"),
            _mock_order(status="filled", filled_qty="10.0"),
        ]
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 10.0)

    def test_returns_none_when_api_errors_until_timeout(self):
        from execution.trader import wait_for_fill

        client = MagicMock()
        client.get_order_by_id.side_effect = Exception("API down")
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertIsNone(result)

    def test_poll_exception_handled_gracefully(self):
        from execution.trader import wait_for_fill

        client = MagicMock()
        client.get_order_by_id.side_effect = Exception("API timeout")
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertIsNone(result)


def _mock_filled_order(order_id, filled_qty=10.0, filled_avg_price=0.0):
    """Return a mock order in terminal 'filled' state for use with wait_for_fill."""
    o = MagicMock()
    o.id = order_id
    o.status = "filled"
    o.filled_qty = str(filled_qty)
    o.filled_avg_price = str(filled_avg_price)
    return o


class TestPlaceSellOrder(unittest.TestCase):
    def test_places_sell_order(self):
        from execution.trader import place_sell_order

        client = MagicMock()
        client.submit_order.return_value = _mock_order("sell-123", status="new")
        client.get_order_by_id.return_value = _mock_filled_order("sell-123", filled_qty=15.5)
        with patch("execution.trader.time.sleep"):
            result = place_sell_order(client, "AAPL", 15.5)
        self.assertTrue(result.is_success)
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.broker_order_id, "sell-123")

    def test_returns_rejected_on_error(self):
        from execution.trader import place_sell_order

        client = MagicMock()
        client.submit_order.side_effect = Exception("position not found")
        result = place_sell_order(client, "AAPL", 15.5)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("position not found", result.rejection_reason)


class TestClosePosition(unittest.TestCase):
    def test_closes_successfully(self):
        from execution.trader import close_position

        client = MagicMock()
        client.close_position.return_value = _mock_order("close-123", status="new")
        client.get_order_by_id.return_value = _mock_filled_order("close-123", filled_qty=10.0)
        with patch("execution.trader.time.sleep"):
            result = close_position(client, "AAPL")
        client.close_position.assert_called_once_with("AAPL")
        self.assertTrue(result.is_success)
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.status, OrderStatus.FILLED)

    def test_returns_rejected_on_error(self):
        from execution.trader import close_position

        client = MagicMock()
        client.close_position.side_effect = Exception("not found")
        result = close_position(client, "AAPL")
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertFalse(result.is_success)
        self.assertIn("not found", result.rejection_reason)

    def test_cancels_open_orders_before_closing(self):
        # Trailing stop holds shares: close_position must cancel all orders first
        # or Alpaca returns "insufficient qty available for order"
        from execution.trader import close_position

        call_order = []
        client = MagicMock()

        pos_held = MagicMock()
        pos_held.qty = "10"
        pos_held.qty_available = "0"
        pos_free = MagicMock()
        pos_free.qty = "10"
        pos_free.qty_available = "10"
        client.get_open_position.side_effect = [pos_held, pos_free]
        client.cancel_orders.side_effect = lambda: call_order.append("cancel")

        def fake_close(symbol):
            call_order.append("close")
            return _mock_order("close-123", status="new")

        client.close_position.side_effect = fake_close
        client.get_order_by_id.return_value = _mock_filled_order("close-123")
        with patch("execution.trader.time.sleep"):
            close_position(client, "AAPL")
        self.assertEqual(call_order, ["cancel", "close"])

    def test_close_succeeds_even_when_cancel_raises(self):
        from execution.trader import close_position

        client = MagicMock()
        client.get_orders.side_effect = Exception("API down")  # cancel_open_orders fails
        client.close_position.return_value = _mock_order("close-456", status="new")
        client.get_order_by_id.return_value = _mock_filled_order("close-456")
        with patch("execution.trader.time.sleep"):
            result = close_position(client, "AAPL")
        client.close_position.assert_called_once_with("AAPL")
        self.assertTrue(result.is_success)


class TestCancelOpenOrders(unittest.TestCase):
    def _make_held_pos(self, symbol="AAPL", qty=10, qty_available=0):
        pos = MagicMock()
        pos.qty = str(qty)
        pos.qty_available = str(qty_available)
        return pos

    def test_cancels_matching_open_orders(self):
        # When shares are held (qty_available < qty), cancel_orders() fires.
        from execution.trader import cancel_open_orders

        client = MagicMock()
        client.get_open_position.side_effect = [
            self._make_held_pos(qty_available=0),  # initial: shares held
            self._make_held_pos(qty_available=10),  # after cancel: freed
        ]
        with patch("execution.trader.time.sleep"):
            cancel_open_orders(client, "AAPL")
        client.cancel_orders.assert_called_once()

    def test_cancel_all_called_once_when_shares_held(self):
        # cancel_open_orders uses cancel-all (not symbol-scoped); safe because
        # ensure_stops_attached re-covers other positions at end of run.
        from execution.trader import cancel_open_orders

        client = MagicMock()
        client.get_open_position.side_effect = [
            self._make_held_pos(qty_available=0),
            self._make_held_pos(qty_available=10),
        ]
        with patch("execution.trader.time.sleep"):
            cancel_open_orders(client, "AAPL")
        client.cancel_orders.assert_called_once()

    def test_no_cancel_when_shares_available(self):
        # If qty_available == qty, no orders hold shares — skip cancel-all.
        from execution.trader import cancel_open_orders

        client = MagicMock()
        client.get_open_position.return_value = self._make_held_pos(qty_available=10)
        cancel_open_orders(client, "AAPL")
        client.cancel_orders.assert_not_called()

    def test_handles_exception_gracefully(self):
        from execution.trader import cancel_open_orders

        client = MagicMock()
        client.get_open_position.side_effect = Exception("API error")
        try:
            cancel_open_orders(client, "AAPL")
        except Exception:
            self.fail("cancel_open_orders raised unexpectedly")


class TestReconcilePositions(unittest.TestCase):
    def setUp(self):
        import utils.db as db_module

        self.tmpdir = tempfile.mkdtemp()
        self.patchers = _meta_patcher(self.tmpdir)
        for p in self.patchers:
            p.start()
        db_module._initialized = False
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)

    def test_removes_stale_metadata_for_closed_positions(self):
        from execution.trader import get_position_meta, reconcile_positions, record_buy

        record_buy("AAPL", 180.0)
        # Alpaca shows no open positions
        client = MagicMock()
        client.get_all_positions.return_value = []
        reconcile_positions(client)
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["signal"], "unknown")  # reverted to default

    def test_adds_placeholder_for_untracked_position(self):
        from execution.trader import get_position_meta, reconcile_positions

        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("NVDA", 10)]
        reconcile_positions(client)
        meta = get_position_meta("NVDA")
        # Placeholder entry should exist with a valid entry_date
        self.assertIn("entry_date", meta)

    def test_keeps_existing_metadata_for_held_positions(self):
        from execution.trader import get_position_meta, reconcile_positions, record_buy

        record_buy("AAPL", 180.0, signal="momentum", confidence=8)
        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("AAPL", 10)]
        reconcile_positions(client)
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["signal"], "momentum")
        self.assertEqual(meta["confidence"], 8)

    def test_api_exception_does_not_crash(self):
        from execution.trader import reconcile_positions

        client = MagicMock()
        client.get_all_positions.side_effect = Exception("broker down")
        try:
            reconcile_positions(client)
        except Exception:
            self.fail("reconcile_positions raised unexpectedly on API error")


class TestEnsureStopsAttached(unittest.TestCase):
    def _make_stop_order(self, symbol, order_type, qty):
        from alpaca.trading.enums import OrderSide

        o = MagicMock()
        o.symbol = symbol
        o.order_type = order_type
        o.side = OrderSide.SELL
        o.qty = str(qty)
        return o

    def test_no_positions_does_nothing(self):
        from execution.trader import ensure_stops_attached

        client = MagicMock()
        client.get_all_positions.return_value = []
        ensure_stops_attached(client)
        client.submit_order.assert_not_called()

    def test_attaches_stop_for_uncovered_position(self):

        from execution.trader import ensure_stops_attached

        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("AAPL", 10.0)]
        client.get_orders.return_value = []  # no existing stops

        with patch("execution.trader.place_trailing_stop") as mock_stop:
            ensure_stops_attached(client)
            mock_stop.assert_called_once()
            args = mock_stop.call_args
            self.assertEqual(args[0][1], "AAPL")

    def test_does_not_duplicate_stop_when_already_covered(self):
        from alpaca.trading.enums import OrderSide, OrderType

        from execution.trader import ensure_stops_attached

        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("AAPL", 10.0)]

        existing_stop = MagicMock()
        existing_stop.symbol = "AAPL"
        existing_stop.order_type = OrderType.STOP
        existing_stop.side = OrderSide.SELL
        existing_stop.qty = "10.0"
        client.get_orders.return_value = [existing_stop]

        with patch("execution.trader.place_trailing_stop") as mock_stop:
            ensure_stops_attached(client)
            mock_stop.assert_not_called()

    def test_handles_api_exception_gracefully(self):
        from execution.trader import ensure_stops_attached

        client = MagicMock()
        client.get_all_positions.side_effect = Exception("API down")
        try:
            ensure_stops_attached(client)
        except Exception:
            self.fail("ensure_stops_attached raised unexpectedly")


class TestIsMarketOpen(unittest.TestCase):
    def test_returns_true_when_market_open(self):
        from execution.trader import is_market_open

        client = MagicMock()
        client.get_clock.return_value = MagicMock(is_open=True)
        self.assertTrue(is_market_open(client))

    def test_returns_false_when_market_closed(self):
        from execution.trader import is_market_open

        client = MagicMock()
        client.get_clock.return_value = MagicMock(is_open=False)
        self.assertFalse(is_market_open(client))


class TestGetPositionSignal(unittest.TestCase):
    def setUp(self):
        import utils.db as db_module

        self.tmpdir = tempfile.mkdtemp()
        self.patchers = _meta_patcher(self.tmpdir)
        for p in self.patchers:
            p.start()
        db_module._initialized = False
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)

    def test_returns_signal_for_known_position(self):
        from execution.trader import get_position_signal, record_buy

        record_buy("AAPL", 180.0, signal="momentum")
        self.assertEqual(get_position_signal("AAPL"), "momentum")

    def test_returns_unknown_for_missing_position(self):
        from execution.trader import get_position_signal

        self.assertEqual(get_position_signal("GHOST"), "unknown")


class TestPlacePartialSell(unittest.TestCase):
    def test_places_partial_sell_order(self):
        from execution.trader import place_partial_sell

        client = MagicMock()
        client.submit_order.return_value = MagicMock(id="order-partial")
        client.get_order_by_id.return_value = _mock_filled_order("order-partial", filled_qty=5.0)
        with patch("execution.trader.time.sleep"):
            result = place_partial_sell(client, "AAPL", 5.0)
        self.assertTrue(result.is_success)
        self.assertEqual(result.symbol, "AAPL")
        self.assertGreater(result.filled_qty, 0)
        self.assertEqual(result.broker_order_id, "order-partial")

    def test_returns_none_for_zero_qty(self):
        from execution.trader import place_partial_sell

        client = MagicMock()
        result = place_partial_sell(client, "AAPL", 0.0)
        self.assertIsNone(result)
        client.submit_order.assert_not_called()

    def test_returns_none_for_negative_qty(self):
        from execution.trader import place_partial_sell

        client = MagicMock()
        result = place_partial_sell(client, "AAPL", -1.0)
        self.assertIsNone(result)
        client.submit_order.assert_not_called()

    def test_returns_rejected_on_error(self):
        from execution.trader import place_partial_sell

        client = MagicMock()
        client.submit_order.side_effect = Exception("insufficient shares")
        result = place_partial_sell(client, "AAPL", 5.0)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertFalse(result.is_success)


# ── Shared terminal-state invariant: only FILLED → is_success ────────────────
#
# For each of close_position, place_sell_order, place_partial_sell:
#   submitted → filled                    → FILLED, is_success
#   submitted → partially_filled → filled → FILLED, is_success (polls past partial)
#   submitted → partially_filled → timeout→ PARTIAL, not success, qty recorded
#   submitted → rejected                  → REJECTED, not success
#   submitted → cancelled                 → TIMEOUT, not success
#   submitted → expired                   → TIMEOUT, not success


def _seq(*statuses_and_qtys):
    """Build a list of mock orders from (status, filled_qty) pairs."""
    orders = []
    for status, qty in statuses_and_qtys:
        o = MagicMock()
        o.status = status
        o.filled_qty = str(qty) if qty is not None else None
        o.filled_avg_price = "0.0"
        orders.append(o)
    return orders


class TestClosePositionTerminalStates(unittest.TestCase):
    def _close(self, order_sequence, final_order=None):
        """Run close_position with a given sequence of get_order_by_id responses."""
        from execution.trader import close_position

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "close-001"
        client.close_position.return_value = submitted
        client.get_orders.return_value = []
        if final_order is not None:
            client.get_order_by_id.side_effect = order_sequence + [final_order]
        else:
            client.get_order_by_id.side_effect = order_sequence
        with patch("execution.trader.time.sleep"):
            return close_position(client, "AAPL")

    def test_submitted_then_filled_is_success(self):
        result = self._close(_seq(("new", None), ("filled", 10.0)))
        self.assertTrue(result.is_success)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertAlmostEqual(result.filled_qty, 10.0)

    def test_partially_filled_then_filled_is_success(self):
        """Must keep polling past partial fill — only full fill is success."""
        result = self._close(_seq(("partially_filled", 5.0), ("filled", 10.0)))
        self.assertTrue(result.is_success)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertAlmostEqual(result.filled_qty, 10.0)

    def test_partially_filled_then_timeout_is_partial_not_success(self):
        """Timeout while partially filled → PARTIAL status, qty recorded, not success."""
        from execution.trader import close_position

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "close-001"
        client.close_position.return_value = submitted
        client.get_orders.return_value = []
        partial_order = MagicMock()
        partial_order.status = "partially_filled"
        partial_order.filled_qty = "7.0"
        client.get_order_by_id.return_value = partial_order
        with patch("execution.trader.wait_for_fill", return_value=None):
            result = close_position(client, "AAPL")
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, OrderStatus.PARTIAL)
        self.assertAlmostEqual(result.filled_qty, 7.0)

    def test_rejected_is_not_success(self):
        result = self._close(_seq(("rejected", None)))
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, OrderStatus.TIMEOUT)

    def test_cancelled_is_not_success(self):
        result = self._close(_seq(("cancelled", None)))
        self.assertFalse(result.is_success)

    def test_expired_is_not_success(self):
        result = self._close(_seq(("expired", None)))
        self.assertFalse(result.is_success)

    def test_submit_exception_is_rejected(self):
        from execution.trader import close_position

        client = MagicMock()
        client.get_orders.return_value = []
        client.close_position.side_effect = Exception("position not found")
        result = close_position(client, "AAPL")
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertFalse(result.is_success)
        self.assertIsNotNone(result.rejection_reason)


class TestPlaceSellOrderTerminalStates(unittest.TestCase):
    def _sell(self, order_sequence, final_order=None, qty=10.0):
        from execution.trader import place_sell_order

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "sell-001"
        client.submit_order.return_value = submitted
        if final_order is not None:
            client.get_order_by_id.side_effect = order_sequence + [final_order]
        else:
            client.get_order_by_id.side_effect = order_sequence
        with patch("execution.trader.time.sleep"):
            return place_sell_order(client, "AAPL", qty)

    def test_submitted_then_filled_is_success(self):
        result = self._sell(_seq(("new", None), ("filled", 10.0)))
        self.assertTrue(result.is_success)
        self.assertEqual(result.status, OrderStatus.FILLED)

    def test_partially_filled_then_filled_is_success(self):
        result = self._sell(_seq(("partially_filled", 5.0), ("filled", 10.0)))
        self.assertTrue(result.is_success)
        self.assertEqual(result.status, OrderStatus.FILLED)

    def test_partially_filled_then_timeout_is_partial_not_success(self):
        from execution.trader import place_sell_order

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "sell-001"
        client.submit_order.return_value = submitted
        partial_order = MagicMock()
        partial_order.status = "partially_filled"
        partial_order.filled_qty = "4.0"
        client.get_order_by_id.return_value = partial_order
        with patch("execution.trader.wait_for_fill", return_value=None):
            result = place_sell_order(client, "AAPL", 10.0)
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, OrderStatus.PARTIAL)
        self.assertAlmostEqual(result.filled_qty, 4.0)

    def test_rejected_is_not_success(self):
        result = self._sell(_seq(("rejected", None)))
        self.assertFalse(result.is_success)

    def test_cancelled_is_not_success(self):
        result = self._sell(_seq(("cancelled", None)))
        self.assertFalse(result.is_success)

    def test_expired_is_not_success(self):
        result = self._sell(_seq(("expired", None)))
        self.assertFalse(result.is_success)

    def test_api_error_once_then_filled(self):
        from execution.trader import place_sell_order

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "sell-001"
        client.submit_order.return_value = submitted
        client.get_order_by_id.side_effect = [
            Exception("transient"),
            _mock_filled_order("sell-001", filled_qty=10.0),
        ]
        with patch("execution.trader.time.sleep"):
            result = place_sell_order(client, "AAPL", 10.0)
        self.assertTrue(result.is_success)

    def test_api_errors_until_timeout(self):
        from execution.trader import place_sell_order

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "sell-001"
        client.submit_order.return_value = submitted
        client.get_order_by_id.side_effect = Exception("API down")
        with patch("execution.trader.time.sleep"):
            result = place_sell_order(client, "AAPL", 10.0)
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, OrderStatus.TIMEOUT)


class TestPlacePartialSellTerminalStates(unittest.TestCase):
    def _partial_sell(self, order_sequence, final_order=None, qty=5.0):
        from execution.trader import place_partial_sell

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "psell-001"
        client.submit_order.return_value = submitted
        if final_order is not None:
            client.get_order_by_id.side_effect = order_sequence + [final_order]
        else:
            client.get_order_by_id.side_effect = order_sequence
        with patch("execution.trader.time.sleep"):
            return place_partial_sell(client, "AAPL", qty)

    def test_submitted_then_filled_is_success(self):
        result = self._partial_sell(_seq(("new", None), ("filled", 5.0)))
        self.assertTrue(result.is_success)
        self.assertEqual(result.status, OrderStatus.FILLED)

    def test_partially_filled_then_filled_is_success(self):
        result = self._partial_sell(_seq(("partially_filled", 2.0), ("filled", 5.0)))
        self.assertTrue(result.is_success)

    def test_partially_filled_then_timeout_is_partial_not_success(self):
        from execution.trader import place_partial_sell

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "psell-001"
        client.submit_order.return_value = submitted
        partial_order = MagicMock()
        partial_order.status = "partially_filled"
        partial_order.filled_qty = "3.0"
        client.get_order_by_id.return_value = partial_order
        with patch("execution.trader.wait_for_fill", return_value=None):
            result = place_partial_sell(client, "AAPL", 5.0)
        self.assertFalse(result.is_success)
        self.assertEqual(result.status, OrderStatus.PARTIAL)
        self.assertAlmostEqual(result.filled_qty, 3.0)

    def test_rejected_is_not_success(self):
        result = self._partial_sell(_seq(("rejected", None)))
        self.assertFalse(result.is_success)

    def test_cancelled_is_not_success(self):
        result = self._partial_sell(_seq(("cancelled", None)))
        self.assertFalse(result.is_success)

    def test_expired_is_not_success(self):
        result = self._partial_sell(_seq(("expired", None)))
        self.assertFalse(result.is_success)


class TestPlaceTrailingStop(unittest.TestCase):
    def _make_order(self, order_id="stop-123"):
        o = MagicMock()
        o.id = order_id
        return o

    def test_returns_none_for_zero_qty(self):
        from execution.trader import place_trailing_stop

        result = place_trailing_stop(MagicMock(), "AAPL", 0.0)
        self.assertIsNone(result)

    def test_returns_none_for_negative_qty(self):
        from execution.trader import place_trailing_stop

        result = place_trailing_stop(MagicMock(), "AAPL", -1.0)
        self.assertIsNone(result)

    def test_whole_shares_places_trailing_stop_order(self):
        from alpaca.trading.requests import TrailingStopOrderRequest

        from execution.trader import place_trailing_stop

        client = MagicMock()
        client.submit_order.return_value = self._make_order("stop-trail")
        result = place_trailing_stop(client, "AAPL", 10.0, current_price=150.0)
        self.assertTrue(result.is_success)
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.stop_order_id, "stop-trail")
        submitted = client.submit_order.call_args[0][0]
        self.assertIsInstance(submitted, TrailingStopOrderRequest)

    def test_fractional_shares_places_fixed_stop_order(self):
        from alpaca.trading.requests import StopOrderRequest

        from execution.trader import place_trailing_stop

        client = MagicMock()
        client.submit_order.return_value = self._make_order("stop-fixed")
        result = place_trailing_stop(client, "AAPL", 2.5, current_price=150.0)
        self.assertTrue(result.is_success)
        self.assertEqual(result.stop_order_id, "stop-fixed")
        # First call is the stop order; second call liquidates the fractional remainder
        stop_req = client.submit_order.call_args_list[0][0][0]
        self.assertIsInstance(stop_req, StopOrderRequest)

    def test_fractional_remainder_is_liquidated(self):
        from alpaca.trading.requests import MarketOrderRequest

        from execution.trader import place_trailing_stop

        client = MagicMock()
        client.submit_order.return_value = self._make_order("stop-fixed")
        place_trailing_stop(client, "AAPL", 2.5, current_price=150.0)
        # Two calls: stop for whole shares, then market sell for remainder
        self.assertEqual(client.submit_order.call_count, 2)
        remainder_req = client.submit_order.call_args_list[1][0][0]
        self.assertIsInstance(remainder_req, MarketOrderRequest)
        self.assertAlmostEqual(remainder_req.qty, 0.5, places=4)

    def test_fractional_stop_price_below_current(self):
        from config import TRAILING_STOP_PCT
        from execution.trader import place_trailing_stop

        client = MagicMock()
        client.submit_order.return_value = self._make_order()
        place_trailing_stop(client, "AAPL", 2.5, current_price=200.0)
        # First call is the stop order
        stop_req = client.submit_order.call_args_list[0][0][0]
        expected_stop = round(200.0 * (1 - TRAILING_STOP_PCT / 100), 2)
        self.assertAlmostEqual(stop_req.stop_price, expected_stop, places=2)

    def test_fractional_without_current_price_returns_stop_failed(self):
        from execution.trader import place_trailing_stop

        result = place_trailing_stop(MagicMock(), "AAPL", 2.5, current_price=None)
        self.assertEqual(result.status, OrderStatus.STOP_FAILED)
        self.assertFalse(result.is_success)

    def test_returns_stop_failed_on_api_error(self):
        from execution.trader import place_trailing_stop

        client = MagicMock()
        client.submit_order.side_effect = Exception("order rejected")
        result = place_trailing_stop(client, "AAPL", 10.0, current_price=150.0)
        self.assertEqual(result.status, OrderStatus.STOP_FAILED)
        self.assertFalse(result.is_success)

    def test_whole_shares_trailing_stop_success_returns_filled(self):
        """GTC trailing stop path (non-fractional qty) — success branch."""
        from execution.trader import place_trailing_stop

        client = MagicMock()
        client.submit_order.return_value = self._make_order("stop-gtc-123")
        result = place_trailing_stop(client, "MSFT", 5.0, current_price=300.0)
        self.assertTrue(result.is_success)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(result.stop_order_id, "stop-gtc-123")

    def test_whole_shares_trailing_stop_exception_returns_stop_failed(self):
        """GTC trailing stop path (non-fractional qty) — exception branch (lines 175-177)."""
        from execution.trader import place_trailing_stop

        client = MagicMock()
        client.submit_order.side_effect = Exception("trailing stop rejected")
        result = place_trailing_stop(client, "MSFT", 5.0, current_price=300.0)
        self.assertEqual(result.status, OrderStatus.STOP_FAILED)
        self.assertFalse(result.is_success)
        self.assertIn("trailing stop rejected", result.rejection_reason)

    def test_fractional_stop_exception_returns_stop_failed(self):
        """Fixed stop path (fractional qty) — exception branch (lines 158-160)."""
        from execution.trader import place_trailing_stop

        client = MagicMock()
        client.submit_order.side_effect = Exception("stop order api error")
        result = place_trailing_stop(client, "AAPL", 2.5, current_price=150.0)
        self.assertEqual(result.status, OrderStatus.STOP_FAILED)
        self.assertFalse(result.is_success)
        self.assertIn("stop order api error", result.rejection_reason)


class TestReconcilePositionsMissingSymbol(unittest.TestCase):
    """Test reconcile_positions adds placeholder for symbol in broker but not in DB."""

    def setUp(self):
        import utils.db as db_module

        self.tmpdir = tempfile.mkdtemp()
        self.patchers = _meta_patcher(self.tmpdir)
        for p in self.patchers:
            p.start()
        db_module._initialized = False
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)

    def test_adds_placeholder_metadata_for_broker_symbol_not_in_db(self):
        """Lines 364-365: symbols in broker but not in DB get a placeholder INSERT."""
        from execution.trader import get_position_meta, reconcile_positions

        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("TSLA", 5)]
        reconcile_positions(client)
        meta = get_position_meta("TSLA")
        # Placeholder entry created — entry_date must be present
        self.assertIn("entry_date", meta)
        self.assertEqual(meta["signal"], "unknown")

    def test_reconcile_db_exception_does_not_crash(self):
        """Lines 364-365: db exception inside the inner block is caught."""
        from execution.trader import reconcile_positions

        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("AAPL", 10)]
        with patch("execution.trader._db", side_effect=Exception("db error")):
            try:
                reconcile_positions(client)
            except Exception:
                self.fail("reconcile_positions raised on DB error")


class TestGetPositionSignalException(unittest.TestCase):
    """Lines 257-258: get_position_signal exception path returns 'unknown'."""

    def test_returns_unknown_when_db_raises(self):
        from execution.trader import get_position_signal

        with patch("execution.trader._db", side_effect=Exception("db down")):
            result = get_position_signal("AAPL")
        self.assertEqual(result, "unknown")


class TestRecordPartialExitException(unittest.TestCase):
    """Lines 278-286: record_partial_exit exception path — no raise, warning logged."""

    def test_does_not_raise_when_db_raises(self):
        from execution.trader import record_partial_exit

        with patch("execution.trader._db", side_effect=Exception("db error")):
            try:
                record_partial_exit("AAPL")
            except Exception:
                self.fail("record_partial_exit raised unexpectedly on DB error")


class TestGetPositionMetaException(unittest.TestCase):
    """Lines 299-300: get_position_meta exception path returns defaults."""

    def test_returns_defaults_when_db_raises(self):
        from execution.trader import get_position_meta

        with patch("execution.trader._db", side_effect=Exception("db error")):
            meta = get_position_meta("AAPL")
        self.assertEqual(meta["signal"], "unknown")
        self.assertEqual(meta["regime"], "UNKNOWN")
        self.assertEqual(meta["confidence"], 0)
        self.assertEqual(meta["entry_price"], 0.0)


class TestLoadAllPositionsException(unittest.TestCase):
    """Lines 310-311: _load_all_positions exception path returns empty dict."""

    def test_returns_empty_dict_when_db_raises(self):
        from execution.trader import _load_all_positions

        with patch("execution.trader._db", side_effect=Exception("db error")):
            result = _load_all_positions()
        self.assertEqual(result, {})


class TestGetPositionAgesException(unittest.TestCase):
    """Lines 327-328: get_position_ages exception on bad entry_date — defaults to age 1."""

    def setUp(self):
        import utils.db as db_module

        self.tmpdir = tempfile.mkdtemp()
        self.patchers = _meta_patcher(self.tmpdir)
        for p in self.patchers:
            p.start()
        db_module._initialized = False
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)

    def test_bad_entry_date_defaults_to_age_one(self):
        """Lines 327-328: bad entry_date causes Exception → age defaults to 1."""
        from execution.trader import get_position_ages

        # Insert a row with a bad entry_date that cannot be parsed
        from utils.db import get_db

        with get_db() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, entry_date, entry_price, signal, regime, confidence) "
                "VALUES (?,?,?,?,?,?)",
                ("BADDATE", "not-a-date", 100.0, "unknown", "UNKNOWN", 0),
            )
        ages = get_position_ages()
        self.assertIn("BADDATE", ages)
        self.assertEqual(ages["BADDATE"], 1)


class TestEnsureStopsAttachedException(unittest.TestCase):
    """Line 397: ensure_stops_attached exception path and sub-share continue."""

    def test_does_not_raise_when_get_all_positions_raises(self):
        from execution.trader import ensure_stops_attached

        client = MagicMock()
        client.get_all_positions.side_effect = Exception("broker unavailable")
        try:
            ensure_stops_attached(client)
        except Exception:
            self.fail("ensure_stops_attached raised unexpectedly")

    def test_does_not_raise_when_get_orders_raises(self):
        from execution.trader import ensure_stops_attached

        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("AAPL", 10)]
        client.get_orders.side_effect = Exception("orders API down")
        try:
            ensure_stops_attached(client)
        except Exception:
            self.fail("ensure_stops_attached raised unexpectedly on get_orders error")

    def test_sub_share_uncovered_skips_without_placing_stop(self):
        """Line 397: whole_uncovered < 1 (entirely sub-share) → continue, no stop placed."""
        from execution.trader import ensure_stops_attached

        # Position of 0.5 shares — entirely sub-share, whole_uncovered = floor(0.5) = 0
        pos = MagicMock()
        pos.symbol = "AAPL"
        pos.qty = "0.5"  # entirely sub-share
        pos.current_price = "150.0"
        client = MagicMock()
        client.get_all_positions.return_value = [pos]
        client.get_orders.return_value = []  # no existing stops → uncovered = 0.5
        with patch("execution.trader.place_trailing_stop") as stop_mock:
            ensure_stops_attached(client)
        stop_mock.assert_not_called()


class TestGetDailyNotionalException(unittest.TestCase):
    """Lines 462-463: get_daily_notional exception path returns 0.0."""

    def test_returns_zero_when_db_raises(self):
        from execution.trader import get_daily_notional

        with patch("execution.trader._db", side_effect=Exception("db error")):
            result = get_daily_notional("2026-01-15")
        self.assertEqual(result, 0.0)


class TestAddDailyNotionalException(unittest.TestCase):
    """Lines 477-478: add_daily_notional exception path — no raise, warning logged."""

    def test_does_not_raise_when_db_raises(self):
        from execution.trader import add_daily_notional

        with patch("execution.trader._db", side_effect=Exception("db error")):
            try:
                add_daily_notional("2026-01-15", 500.0)
            except Exception:
                self.fail("add_daily_notional raised unexpectedly on DB error")


class TestAutoCancelTimeoutIntents(unittest.TestCase):
    """auto_cancel_timeout_intents resolves timeout intents with no broker position."""

    def setUp(self):

        self.tmpdir = tempfile.mkdtemp()
        self.patchers = _meta_patcher(self.tmpdir)
        for p in self.patchers:
            p.start()
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)

    def _insert_intent(self, symbol, status, trade_date="2026-05-05"):
        from utils.order_ledger import create_intent, update_intent

        client_id = f"ib-{symbol}-BUY-{trade_date}"
        create_intent(symbol, "BUY", trade_date, 1000.0, client_id)
        if status != "pending":
            update_intent(client_id, status, broker_order_id=f"broker-{symbol}")
        return client_id

    def test_cancels_timeout_intent_with_no_broker_position(self):
        from utils.order_ledger import auto_cancel_timeout_intents, get_unresolved_intents

        self._insert_intent("AAPL", "timeout")
        resolved = auto_cancel_timeout_intents(broker_symbols=set(), trade_date="2026-05-05")
        self.assertEqual(resolved, 1)
        remaining = get_unresolved_intents(trade_date="2026-05-05")
        self.assertEqual(len(remaining), 0)

    def test_preserves_timeout_intent_when_broker_has_position(self):
        from utils.order_ledger import auto_cancel_timeout_intents, get_unresolved_intents

        self._insert_intent("AAPL", "timeout")
        resolved = auto_cancel_timeout_intents(broker_symbols={"AAPL"}, trade_date="2026-05-05")
        self.assertEqual(resolved, 0)
        remaining = get_unresolved_intents(trade_date="2026-05-05")
        self.assertEqual(len(remaining), 1)

    def test_does_not_cancel_submitted_intents(self):
        from utils.order_ledger import auto_cancel_timeout_intents, get_unresolved_intents

        self._insert_intent("AAPL", "submitted")
        resolved = auto_cancel_timeout_intents(broker_symbols=set(), trade_date="2026-05-05")
        self.assertEqual(resolved, 0)
        remaining = get_unresolved_intents(trade_date="2026-05-05")
        self.assertEqual(len(remaining), 1)

    def test_handles_exception_gracefully(self):
        from utils.order_ledger import auto_cancel_timeout_intents

        with patch("utils.order_ledger.get_unresolved_intents", side_effect=Exception("db fail")):
            try:
                result = auto_cancel_timeout_intents(broker_symbols=set(), trade_date="2026-05-05")
            except Exception:
                self.fail("auto_cancel_timeout_intents raised unexpectedly")
        self.assertEqual(result, 0)


class TestReconcileFilledIntents(unittest.TestCase):
    """reconcile_filled_intents marks timeout intents filled when broker position confirmed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patchers = _meta_patcher(self.tmpdir)
        for p in self.patchers:
            p.start()
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)

    def _insert_intent(self, symbol, status, trade_date="2026-05-07"):
        from utils.order_ledger import create_intent, update_intent

        client_id = f"ib-{symbol}-BUY-{trade_date}"
        create_intent(symbol, "BUY", trade_date, 1000.0, client_id)
        if status != "pending":
            update_intent(client_id, status, broker_order_id=f"broker-{symbol}")
        return client_id

    def test_resolves_timeout_when_broker_has_position(self):
        from utils.order_ledger import get_unresolved_intents, reconcile_filled_intents

        self._insert_intent("NVDA", "timeout")
        resolved = reconcile_filled_intents(broker_symbols={"NVDA"}, trade_date="2026-05-07")
        self.assertEqual(resolved, 1)
        remaining = get_unresolved_intents(trade_date="2026-05-07")
        self.assertEqual(len(remaining), 0)

    def test_preserves_timeout_when_broker_has_no_position(self):
        from utils.order_ledger import get_unresolved_intents, reconcile_filled_intents

        self._insert_intent("NVDA", "timeout")
        resolved = reconcile_filled_intents(broker_symbols=set(), trade_date="2026-05-07")
        self.assertEqual(resolved, 0)
        remaining = get_unresolved_intents(trade_date="2026-05-07")
        self.assertEqual(len(remaining), 1)

    def test_does_not_resolve_submitted_intents(self):
        from utils.order_ledger import get_unresolved_intents, reconcile_filled_intents

        self._insert_intent("NVDA", "submitted")
        resolved = reconcile_filled_intents(broker_symbols={"NVDA"}, trade_date="2026-05-07")
        self.assertEqual(resolved, 0)
        remaining = get_unresolved_intents(trade_date="2026-05-07")
        self.assertEqual(len(remaining), 1)

    def test_resolves_multiple_timeout_intents(self):
        from utils.order_ledger import get_unresolved_intents, reconcile_filled_intents

        for sym in ("AMAT", "TSM", "NVDA"):
            self._insert_intent(sym, "timeout")
        resolved = reconcile_filled_intents(
            broker_symbols={"AMAT", "TSM", "NVDA"}, trade_date="2026-05-07"
        )
        self.assertEqual(resolved, 3)
        remaining = get_unresolved_intents(trade_date="2026-05-07")
        self.assertEqual(len(remaining), 0)

    def test_handles_exception_gracefully(self):
        from utils.order_ledger import reconcile_filled_intents

        with patch("utils.order_ledger.get_unresolved_intents", side_effect=Exception("db fail")):
            try:
                result = reconcile_filled_intents(broker_symbols={"NVDA"}, trade_date="2026-05-07")
            except Exception:
                self.fail("reconcile_filled_intents raised unexpectedly")
        self.assertEqual(result, 0)

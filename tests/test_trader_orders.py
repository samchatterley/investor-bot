"""Tests for execution/trader.py — order placement, fill polling, reconciliation."""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call


def _mock_order(order_id="order-123", status="new", filled_qty=None):
    o = MagicMock()
    o.id = order_id
    o.status = status
    o.filled_qty = filled_qty
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
        with patch("execution.trader.wait_for_fill", return_value=28.5):
            result = place_buy_order(client, "AAPL", 5_000.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["filled_qty"], 28.5)

    def test_returns_none_on_api_error(self):
        from execution.trader import place_buy_order
        client = MagicMock()
        client.submit_order.side_effect = Exception("insufficient funds")
        with patch("execution.trader.wait_for_fill", return_value=None):
            result = place_buy_order(client, "AAPL", 5_000.0)
        self.assertIsNone(result)

    def test_order_includes_order_id_and_notional(self):
        from execution.trader import place_buy_order
        client = MagicMock()
        client.submit_order.return_value = _mock_order("order-xyz")
        with patch("execution.trader.wait_for_fill", return_value=10.0):
            result = place_buy_order(client, "MSFT", 3_000.0)
        self.assertEqual(result["order_id"], "order-xyz")
        self.assertAlmostEqual(result["notional"], 3_000.0)


class TestWaitForFill(unittest.TestCase):

    def test_returns_qty_when_immediately_filled(self):
        from execution.trader import wait_for_fill
        filled_order = _mock_order(status="filled", filled_qty="28.571")
        client = MagicMock()
        client.get_order_by_id.return_value = filled_order
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertAlmostEqual(result, 28.571)

    def test_returns_none_on_timeout(self):
        from execution.trader import wait_for_fill
        pending_order = _mock_order(status="new", filled_qty=None)
        client = MagicMock()
        client.get_order_by_id.return_value = pending_order
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertIsNone(result)

    def test_returns_qty_on_partial_fill(self):
        from execution.trader import wait_for_fill
        partial_order = _mock_order(status="partially_filled", filled_qty="10.0")
        client = MagicMock()
        client.get_order_by_id.return_value = partial_order
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertAlmostEqual(result, 10.0)

    def test_poll_exception_handled_gracefully(self):
        from execution.trader import wait_for_fill
        client = MagicMock()
        client.get_order_by_id.side_effect = Exception("API timeout")
        with patch("execution.trader.time.sleep"):
            result = wait_for_fill(client, "order-123", max_wait=3)
        self.assertIsNone(result)


class TestPlaceSellOrder(unittest.TestCase):

    def test_places_sell_order(self):
        from execution.trader import place_sell_order
        client = MagicMock()
        client.submit_order.return_value = _mock_order("sell-123", status="new")
        result = place_sell_order(client, "AAPL", 15.5)
        self.assertIsNotNone(result)
        self.assertEqual(result["symbol"], "AAPL")

    def test_returns_none_on_error(self):
        from execution.trader import place_sell_order
        client = MagicMock()
        client.submit_order.side_effect = Exception("position not found")
        result = place_sell_order(client, "AAPL", 15.5)
        self.assertIsNone(result)


class TestClosePosition(unittest.TestCase):

    def test_closes_successfully(self):
        from execution.trader import close_position
        client = MagicMock()
        result = close_position(client, "AAPL")
        client.close_position.assert_called_once_with("AAPL")
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["status"], "closed")

    def test_returns_none_on_error(self):
        from execution.trader import close_position
        client = MagicMock()
        client.close_position.side_effect = Exception("not found")
        result = close_position(client, "AAPL")
        self.assertIsNone(result)


class TestCancelOpenOrders(unittest.TestCase):

    def _make_order(self, symbol, status="new"):
        o = MagicMock()
        o.symbol = symbol
        o.status = status
        o.id = f"order-{symbol}"
        return o

    def test_cancels_matching_open_orders(self):
        from execution.trader import cancel_open_orders
        client = MagicMock()
        client.get_orders.return_value = [self._make_order("AAPL", "new")]
        cancel_open_orders(client, "AAPL")
        client.cancel_order_by_id.assert_called_once_with("order-AAPL")

    def test_does_not_cancel_other_symbols(self):
        from execution.trader import cancel_open_orders
        client = MagicMock()
        client.get_orders.return_value = [
            self._make_order("AAPL", "new"),
            self._make_order("NVDA", "new"),
        ]
        cancel_open_orders(client, "AAPL")
        # Only AAPL should be cancelled
        args = [c.args[0] for c in client.cancel_order_by_id.call_args_list]
        self.assertIn("order-AAPL", args)
        self.assertNotIn("order-NVDA", args)

    def test_does_not_cancel_filled_orders(self):
        from execution.trader import cancel_open_orders
        client = MagicMock()
        client.get_orders.return_value = [self._make_order("AAPL", "filled")]
        cancel_open_orders(client, "AAPL")
        client.cancel_order_by_id.assert_not_called()

    def test_handles_exception_gracefully(self):
        from execution.trader import cancel_open_orders
        client = MagicMock()
        client.get_orders.side_effect = Exception("API error")
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
        from execution.trader import record_buy, reconcile_positions, get_position_meta
        record_buy("AAPL", 180.0)
        # Alpaca shows no open positions
        client = MagicMock()
        client.get_all_positions.return_value = []
        reconcile_positions(client)
        meta = get_position_meta("AAPL")
        self.assertEqual(meta["signal"], "unknown")  # reverted to default

    def test_adds_placeholder_for_untracked_position(self):
        from execution.trader import reconcile_positions, get_position_meta
        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("NVDA", 10)]
        reconcile_positions(client)
        meta = get_position_meta("NVDA")
        # Placeholder entry should exist with a valid entry_date
        self.assertIn("entry_date", meta)

    def test_keeps_existing_metadata_for_held_positions(self):
        from execution.trader import record_buy, reconcile_positions, get_position_meta
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
        from alpaca.trading.enums import OrderType, OrderSide
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
        from alpaca.trading.enums import OrderType
        client = MagicMock()
        client.get_all_positions.return_value = [_mock_position("AAPL", 10.0)]
        client.get_orders.return_value = []  # no existing stops

        with patch("execution.trader.place_trailing_stop") as mock_stop:
            ensure_stops_attached(client)
            mock_stop.assert_called_once()
            args = mock_stop.call_args
            self.assertEqual(args[0][1], "AAPL")

    def test_does_not_duplicate_stop_when_already_covered(self):
        from execution.trader import ensure_stops_attached
        from alpaca.trading.enums import OrderType, OrderSide
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
        from execution.trader import record_buy, get_position_signal
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
        result = place_partial_sell(client, "AAPL", 5.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["symbol"], "AAPL")
        self.assertAlmostEqual(result["qty"], 5.0)

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

    def test_returns_none_on_error(self):
        from execution.trader import place_partial_sell
        client = MagicMock()
        client.submit_order.side_effect = Exception("insufficient shares")
        result = place_partial_sell(client, "AAPL", 5.0)
        self.assertIsNone(result)


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
        from execution.trader import place_trailing_stop
        from alpaca.trading.requests import TrailingStopOrderRequest
        client = MagicMock()
        client.submit_order.return_value = self._make_order("stop-trail")
        result = place_trailing_stop(client, "AAPL", 10.0, current_price=150.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["symbol"], "AAPL")
        self.assertIn("trail_pct", result)
        # Verify TrailingStopOrderRequest was used (not StopOrderRequest)
        submitted = client.submit_order.call_args[0][0]
        self.assertIsInstance(submitted, TrailingStopOrderRequest)

    def test_fractional_shares_places_fixed_stop_order(self):
        from execution.trader import place_trailing_stop
        from alpaca.trading.requests import StopOrderRequest
        client = MagicMock()
        client.submit_order.return_value = self._make_order("stop-fixed")
        result = place_trailing_stop(client, "AAPL", 2.5, current_price=150.0)
        self.assertIsNotNone(result)
        self.assertIn("stop_price", result)
        submitted = client.submit_order.call_args[0][0]
        self.assertIsInstance(submitted, StopOrderRequest)

    def test_fractional_stop_price_below_current(self):
        from execution.trader import place_trailing_stop
        from config import TRAILING_STOP_PCT
        client = MagicMock()
        client.submit_order.return_value = self._make_order()
        result = place_trailing_stop(client, "AAPL", 2.5, current_price=200.0)
        expected_stop = round(200.0 * (1 - TRAILING_STOP_PCT / 100), 2)
        self.assertAlmostEqual(result["stop_price"], expected_stop, places=2)

    def test_fractional_without_current_price_returns_none(self):
        from execution.trader import place_trailing_stop
        result = place_trailing_stop(MagicMock(), "AAPL", 2.5, current_price=None)
        self.assertIsNone(result)

    def test_returns_none_on_api_error(self):
        from execution.trader import place_trailing_stop
        client = MagicMock()
        client.submit_order.side_effect = Exception("order rejected")
        result = place_trailing_stop(client, "AAPL", 10.0, current_price=150.0)
        self.assertIsNone(result)

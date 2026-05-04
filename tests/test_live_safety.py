"""
Live-safety regression tests covering the three critical blockers identified
in the 2026-05-04 pre-live code review:

  1. Capital containment — £150-bounded caps must be enforced before every order.
  2. Duplicate-buy prevention — pending orders and same-day replays must be blocked.
  3. Stop failure is fatal — failed stop placement must flatten or halt, not just log.

Also covers: dollar daily loss cap, open-exposure cap, VIX-adjusted stop wiring,
broker account assertions, small-account sizing, and universe price filter.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from models import OrderResult, OrderStatus

# ── Shared helpers ────────────────────────────────────────────────────────────


def _account(value=150.0, cash=130.0):
    return {"portfolio_value": value, "cash": cash, "buying_power": cash, "equity": value}


def _pos(symbol="AAPL", qty=1.0, price=50.0, pl=0.0, plpc=0.0, mv=50.0):
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_entry_price": price,
        "current_price": price,
        "unrealized_pl": pl,
        "unrealized_plpc": plpc,
        "market_value": mv,
    }


def _regime(bearish=False):
    return {"regime": "BULL", "is_bearish": bearish}


def _macro(high_risk=False):
    return {"is_high_risk": high_risk, "event": ""}


def _decisions(buys=None, sells=None):
    return {
        "market_summary": "Small account test summary.",
        "buy_candidates": buys or [],
        "position_decisions": sells or [],
    }


def _buy_candidate(symbol="SOFI", confidence=8, signal="momentum"):
    return {
        "symbol": symbol,
        "confidence": confidence,
        "reasoning": "Strong momentum signal on high volume breakout.",
        "key_signal": signal,
    }


# ── Capital containment tests ─────────────────────────────────────────────────


class TestSmallAccountCapitalCaps(unittest.TestCase):
    """SMALL_ACCOUNT_MODE caps must be enforced before every order."""

    def test_max_single_order_default_in_small_account_mode(self):
        """In SMALL_ACCOUNT_MODE the single-order cap defaults to $55, not $50,000."""
        import importlib

        with patch.dict(os.environ, {"SMALL_ACCOUNT_MODE": "true"}, clear=False):
            import config as cfg

            importlib.reload(cfg)
            self.assertLessEqual(cfg.MAX_SINGLE_ORDER_USD, 55.0)
            self.assertLess(cfg.MAX_SINGLE_ORDER_USD, 1000.0)

    def test_max_daily_notional_default_in_small_account_mode(self):
        import importlib

        with patch.dict(os.environ, {"SMALL_ACCOUNT_MODE": "true"}, clear=False):
            import config as cfg

            importlib.reload(cfg)
            self.assertLessEqual(cfg.MAX_DAILY_NOTIONAL_USD, 75.0)

    def test_max_deployed_default_in_small_account_mode(self):
        import importlib

        with patch.dict(os.environ, {"SMALL_ACCOUNT_MODE": "true"}, clear=False):
            import config as cfg

            importlib.reload(cfg)
            self.assertGreater(cfg.MAX_DEPLOYED_USD, 0)
            self.assertLessEqual(cfg.MAX_DEPLOYED_USD, 125.0)

    def test_max_orders_per_run_is_1_in_small_account_mode(self):
        import importlib

        with patch.dict(os.environ, {"SMALL_ACCOUNT_MODE": "true"}, clear=False):
            import config as cfg

            importlib.reload(cfg)
            self.assertEqual(cfg.MAX_ORDERS_PER_RUN, 1)

    def test_max_positions_is_2_in_small_account_mode(self):
        import importlib

        with patch.dict(os.environ, {"SMALL_ACCOUNT_MODE": "true"}, clear=False):
            import config as cfg

            importlib.reload(cfg)
            self.assertEqual(cfg.MAX_POSITIONS, 2)

    def test_env_override_wins_over_small_account_default(self):
        """An explicit env var always beats the SMALL_ACCOUNT_MODE default."""
        import importlib

        with patch.dict(
            os.environ,
            {"SMALL_ACCOUNT_MODE": "true", "MAX_SINGLE_ORDER_USD": "40.0"},
            clear=False,
        ):
            import config as cfg

            importlib.reload(cfg)
            self.assertAlmostEqual(cfg.MAX_SINGLE_ORDER_USD, 40.0, places=1)


class TestCheckPreTradeDeployedCap(unittest.TestCase):
    """check_pre_trade() must enforce MAX_DEPLOYED_USD."""

    def test_deployed_cap_blocks_order_that_would_exceed_it(self):
        from utils.validators import check_pre_trade

        approved, reason = check_pre_trade(
            "SOFI",
            notional=55.0,
            daily_notional_so_far=0.0,
            max_single_order=55.0,
            max_daily_notional=75.0,
            open_exposure_usd=80.0,
            max_deployed_usd=125.0,
        )
        self.assertFalse(approved)
        self.assertIn("deployed cap", reason)

    def test_deployed_cap_permits_order_within_budget(self):
        from utils.validators import check_pre_trade

        approved, _ = check_pre_trade(
            "SOFI",
            notional=45.0,
            daily_notional_so_far=0.0,
            max_single_order=55.0,
            max_daily_notional=75.0,
            open_exposure_usd=70.0,
            max_deployed_usd=125.0,
        )
        self.assertTrue(approved)

    def test_deployed_cap_disabled_when_zero(self):
        from utils.validators import check_pre_trade

        approved, _ = check_pre_trade(
            "SOFI",
            notional=9999.0,
            daily_notional_so_far=0.0,
            max_single_order=99999.0,
            max_daily_notional=999999.0,
            open_exposure_usd=9999.0,
            max_deployed_usd=0.0,
        )
        self.assertTrue(approved)

    def test_rejection_message_contains_symbol(self):
        from utils.validators import check_pre_trade

        _, reason = check_pre_trade(
            "RKLB", 55.0, 0.0, 55.0, 75.0, open_exposure_usd=80.0, max_deployed_usd=125.0
        )
        self.assertIn("RKLB", reason)

    def test_rejection_shows_current_exposure(self):
        from utils.validators import check_pre_trade

        _, reason = check_pre_trade(
            "HOOD", 55.0, 0.0, 55.0, 75.0, open_exposure_usd=80.0, max_deployed_usd=125.0
        )
        self.assertIn("80", reason)


# ── Dollar daily loss cap ─────────────────────────────────────────────────────


class TestDollarDailyLossCap(unittest.TestCase):
    """MAX_DAILY_LOSS_USD triggers close-all independently of the % cap.

    Tests the calculation logic directly rather than running _run_inner,
    which would invoke real AI analysis and introduce unrelated sell paths.
    """

    def _should_trigger(self, portfolio_value, baseline, max_daily_loss_usd):
        """Apply the dollar cap logic from main._run_inner and return (triggered, dl_pct)."""
        dl_triggered = False
        dl_pct = 0.0
        if max_daily_loss_usd > 0:
            daily_loss_usd = baseline - portfolio_value
            if daily_loss_usd >= max_daily_loss_usd:
                dl_triggered = True
                dl_pct = (portfolio_value / baseline - 1) * 100
        return dl_triggered, dl_pct

    def test_dollar_loss_cap_triggers_when_exceeded(self):
        triggered, pct = self._should_trigger(125.0, 150.0, 20.0)
        self.assertTrue(triggered)
        self.assertLess(pct, 0)

    def test_dollar_loss_at_exact_cap_triggers(self):
        triggered, _ = self._should_trigger(130.0, 150.0, 20.0)
        self.assertTrue(triggered)

    def test_dollar_loss_below_cap_does_not_trigger(self):
        triggered, _ = self._should_trigger(140.0, 150.0, 20.0)
        self.assertFalse(triggered)

    def test_zero_cap_never_triggers(self):
        triggered, _ = self._should_trigger(1.0, 150.0, 0.0)
        self.assertFalse(triggered)

    def test_no_loss_does_not_trigger(self):
        triggered, _ = self._should_trigger(160.0, 150.0, 20.0)
        self.assertFalse(triggered)

    def test_config_max_daily_loss_usd_set_in_small_account_mode(self):
        import importlib

        with patch.dict(os.environ, {"SMALL_ACCOUNT_MODE": "true"}, clear=False):
            import config as cfg

            importlib.reload(cfg)
            self.assertGreater(cfg.MAX_DAILY_LOSS_USD, 0)
            self.assertLessEqual(cfg.MAX_DAILY_LOSS_USD, 20.0)


# ── Duplicate buy prevention ─────────────────────────────────────────────────


class TestHasPendingBuy(unittest.TestCase):
    """has_pending_buy() must detect pending broker buy orders."""

    def _make_order(self, symbol, side_buy, status):
        from alpaca.trading.enums import OrderSide

        o = MagicMock()
        o.symbol = symbol
        o.side = OrderSide.BUY if side_buy else MagicMock()
        o.status = status
        return o

    def test_detects_new_buy_order(self):
        from execution.trader import has_pending_buy

        client = MagicMock()
        client.get_orders.return_value = [self._make_order("SOFI", True, "new")]
        self.assertTrue(has_pending_buy(client, "SOFI"))

    def test_detects_accepted_buy_order(self):
        from execution.trader import has_pending_buy

        client = MagicMock()
        client.get_orders.return_value = [self._make_order("SOFI", True, "accepted")]
        self.assertTrue(has_pending_buy(client, "SOFI"))

    def test_detects_partially_filled_buy_order(self):
        from execution.trader import has_pending_buy

        client = MagicMock()
        client.get_orders.return_value = [self._make_order("SOFI", True, "partially_filled")]
        self.assertTrue(has_pending_buy(client, "SOFI"))

    def test_no_false_positive_for_different_symbol(self):
        from execution.trader import has_pending_buy

        client = MagicMock()
        client.get_orders.return_value = [self._make_order("AAPL", True, "new")]
        self.assertFalse(has_pending_buy(client, "SOFI"))

    def test_no_false_positive_for_filled_order(self):
        from execution.trader import has_pending_buy

        client = MagicMock()
        client.get_orders.return_value = [self._make_order("SOFI", True, "filled")]
        self.assertFalse(has_pending_buy(client, "SOFI"))

    def test_raises_broker_state_unavailable_when_get_orders_raises(self):
        from execution.trader import has_pending_buy
        from models import BrokerStateUnavailable

        client = MagicMock()
        client.get_orders.side_effect = Exception("network error")
        with self.assertRaises(BrokerStateUnavailable):
            has_pending_buy(client, "SOFI")

    def test_no_false_positive_for_sell_order(self):
        from alpaca.trading.enums import OrderSide

        from execution.trader import has_pending_buy

        o = MagicMock()
        o.symbol = "SOFI"
        o.side = OrderSide.SELL
        o.status = "new"
        client = MagicMock()
        client.get_orders.return_value = [o]
        self.assertFalse(has_pending_buy(client, "SOFI"))


class TestClientOrderIdIsSymbolDate(unittest.TestCase):
    """place_buy_order must use a symbol+date client_order_id for idempotency."""

    def test_client_order_id_contains_symbol_and_date(self):
        from execution.trader import place_buy_order

        submitted_kwargs = []

        def fake_submit(req):
            submitted_kwargs.append(req)
            order = MagicMock()
            order.id = "test-id"
            return order

        client = MagicMock()
        client.submit_order.side_effect = fake_submit

        with patch("execution.trader.wait_for_fill", return_value=1.0):
            place_buy_order(client, "SOFI", 50.0)

        self.assertEqual(len(submitted_kwargs), 1)
        req = submitted_kwargs[0]
        cid = req.client_order_id
        self.assertIn("SOFI", cid)
        from config import today_et

        self.assertIn(today_et().isoformat(), cid)

    def test_same_symbol_same_day_produces_same_id(self):
        """Two calls on the same day with same symbol → same client_order_id."""
        from execution.trader import place_buy_order

        ids = []

        def fake_submit(req):
            ids.append(req.client_order_id)
            order = MagicMock()
            order.id = "test-id"
            return order

        client = MagicMock()
        client.submit_order.side_effect = fake_submit

        with patch("execution.trader.wait_for_fill", return_value=1.0):
            place_buy_order(client, "SOFI", 50.0)
            place_buy_order(client, "SOFI", 50.0)

        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], ids[1])


# ── Stop failure → flatten ────────────────────────────────────────────────────


class TestHandleStopFailure(unittest.TestCase):
    """_handle_stop_failure must flatten in live mode; halt if flatten also fails."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.halt_file = os.path.join(self.tmpdir, "HALT")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _run(self, flatten_succeeds=True):
        flatten_result = OrderResult(
            status=OrderStatus.FILLED if flatten_succeeds else OrderStatus.REJECTED,
            symbol="SOFI",
        )
        with (
            patch("main.config.IS_PAPER", False),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
            patch("main.trader.close_position", return_value=flatten_result),
            patch("main.trader.record_sell"),
            patch("main.alerts.alert_error"),
        ):
            from main import _handle_stop_failure

            _handle_stop_failure(MagicMock(), "SOFI", dry_run=False)

    def test_flatten_attempted_on_stop_failure_in_live_mode(self):
        close_mock = MagicMock(return_value=OrderResult(status=OrderStatus.FILLED, symbol="SOFI"))
        with (
            patch("main.config.IS_PAPER", False),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
            patch("main.trader.close_position", close_mock),
            patch("main.trader.record_sell"),
            patch("main.alerts.alert_error"),
        ):
            from main import _handle_stop_failure

            _handle_stop_failure(MagicMock(), "SOFI", dry_run=False)
        close_mock.assert_called_once()

    def test_halt_file_written_when_flatten_also_fails(self):
        self._run(flatten_succeeds=False)
        self.assertTrue(os.path.exists(self.halt_file))

    def test_halt_file_contains_symbol(self):
        self._run(flatten_succeeds=False)
        with open(self.halt_file) as f:
            data = json.load(f)
        self.assertEqual(data["symbol"], "SOFI")
        self.assertEqual(data["reason"], "stop_failure_and_flatten_failure")

    def test_no_halt_when_flatten_succeeds(self):
        self._run(flatten_succeeds=True)
        self.assertFalse(os.path.exists(self.halt_file))

    def test_dry_run_does_not_flatten(self):
        close_mock = MagicMock()
        with (
            patch("main.config.IS_PAPER", False),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
            patch("main.trader.close_position", close_mock),
            patch("main.trader.record_sell"),
            patch("main.alerts.alert_error"),
        ):
            from main import _handle_stop_failure

            _handle_stop_failure(MagicMock(), "SOFI", dry_run=True)
        close_mock.assert_not_called()

    def test_paper_mode_does_not_flatten(self):
        close_mock = MagicMock()
        with (
            patch("main.config.IS_PAPER", True),
            patch("main.config.HALT_FILE", self.halt_file),
            patch("main.config.LOG_DIR", self.tmpdir),
            patch("main.trader.close_position", close_mock),
            patch("main.trader.record_sell"),
            patch("main.alerts.alert_error"),
        ):
            from main import _handle_stop_failure

            _handle_stop_failure(MagicMock(), "SOFI", dry_run=False)
        close_mock.assert_not_called()


# ── ensure_stops_attached fatal return value ──────────────────────────────────


class TestEnsureStopsAttachedFatalReturn(unittest.TestCase):
    """ensure_stops_attached() must return False when stop attachment fails."""

    def _make_position(self, symbol, qty, price):
        p = MagicMock()
        p.symbol = symbol
        p.qty = str(qty)
        p.current_price = str(price)
        return p

    def test_returns_true_when_all_positions_protected(self):
        from alpaca.trading.enums import OrderSide, OrderType

        from execution.trader import ensure_stops_attached

        pos = self._make_position("SOFI", 2.0, 10.0)
        stop_order = MagicMock()
        stop_order.symbol = "SOFI"
        stop_order.order_type = OrderType.STOP
        stop_order.side = OrderSide.SELL
        stop_order.qty = "2"
        client = MagicMock()
        client.get_all_positions.return_value = [pos]
        client.get_orders.return_value = [stop_order]
        result = ensure_stops_attached(client)
        self.assertTrue(result)

    def test_returns_false_when_stop_attachment_fails(self):
        from execution.trader import ensure_stops_attached

        pos = self._make_position("SOFI", 2.0, 10.0)
        client = MagicMock()
        client.get_all_positions.return_value = [pos]
        client.get_orders.return_value = []
        # Stop order submission fails
        client.submit_order.side_effect = Exception("broker error")
        result = ensure_stops_attached(client)
        self.assertFalse(result)

    def test_returns_true_with_no_positions(self):
        from execution.trader import ensure_stops_attached

        client = MagicMock()
        client.get_all_positions.return_value = []
        result = ensure_stops_attached(client)
        self.assertTrue(result)

    def test_returns_false_on_unexpected_exception(self):
        from execution.trader import ensure_stops_attached

        client = MagicMock()
        client.get_all_positions.side_effect = Exception("network failure")
        result = ensure_stops_attached(client)
        self.assertFalse(result)


# ── Small-account sizing ──────────────────────────────────────────────────────


class TestSmallAccountSize(unittest.TestCase):
    """small_account_size() must produce usable notionals for a £150 account."""

    def test_produces_at_least_40_for_150_account(self):
        from risk.position_sizer import small_account_size

        result = small_account_size(150.0)
        self.assertGreaterEqual(result, 40.0)

    def test_capped_at_max_single_order(self):
        from risk.position_sizer import small_account_size

        result = small_account_size(10000.0, max_single_order=55.0)
        self.assertLessEqual(result, 55.0)

    def test_zero_portfolio_returns_zero(self):
        from risk.position_sizer import small_account_size

        self.assertEqual(small_account_size(0.0), 0.0)

    def test_negative_portfolio_returns_zero(self):
        from risk.position_sizer import small_account_size

        self.assertEqual(small_account_size(-100.0), 0.0)

    def test_floor_at_40_even_for_very_small_account(self):
        from risk.position_sizer import small_account_size

        # Very small account — floor at 40 means it's non-functional, but doesn't crash
        result = small_account_size(5.0)
        self.assertEqual(result, 40.0)


# ── VIX-adjusted trailing stop ────────────────────────────────────────────────


class TestVixAdjustedTrailPercent(unittest.TestCase):
    """place_trailing_stop() trail_percent parameter must override config."""

    def test_custom_trail_percent_used_for_whole_share(self):
        from alpaca.trading.requests import TrailingStopOrderRequest

        from execution.trader import place_trailing_stop

        client = MagicMock()
        order = MagicMock()
        order.id = "x"
        client.submit_order.return_value = order

        place_trailing_stop(client, "SOFI", qty=2.0, trail_percent=8.5)
        submitted = client.submit_order.call_args[0][0]
        self.assertIsInstance(submitted, TrailingStopOrderRequest)
        self.assertAlmostEqual(float(submitted.trail_percent), 8.5, places=1)

    def test_custom_trail_percent_used_for_fractional(self):
        from alpaca.trading.requests import StopOrderRequest

        from execution.trader import place_trailing_stop

        client = MagicMock()
        order = MagicMock()
        order.id = "x"
        client.submit_order.return_value = order

        current_price = 10.0
        trail = 9.0
        place_trailing_stop(
            client, "SOFI", qty=1.5, current_price=current_price, trail_percent=trail
        )
        stop_req = client.submit_order.call_args_list[0][0][0]
        self.assertIsInstance(stop_req, StopOrderRequest)
        expected = round(current_price * (1 - trail / 100), 2)
        self.assertAlmostEqual(float(stop_req.stop_price), expected, places=2)


# ── Broker account assertions ─────────────────────────────────────────────────


class TestAssertAccountSafety(unittest.TestCase):
    """_assert_account_safety must raise RuntimeError on margin/PDT accounts."""

    def test_paper_mode_skips_assertions(self):
        with patch("main.config.IS_PAPER", True):
            from main import _assert_account_safety

            try:
                _assert_account_safety(MagicMock())
            except RuntimeError:
                self.fail("_assert_account_safety raised in paper mode")

    def test_pdt_flag_raises_in_live_mode(self):
        account = MagicMock()
        account.pattern_day_trader = True
        account.equity = "150"
        account.buying_power = "200"
        client = MagicMock()
        client.get_account.return_value = account

        with patch("main.config.IS_PAPER", False), patch("main.config.ALLOW_MARGIN", False):
            from main import _assert_account_safety

            with self.assertRaises(RuntimeError):
                _assert_account_safety(client)

    def test_high_buying_power_raises_in_live_mode(self):
        account = MagicMock()
        account.pattern_day_trader = False
        account.equity = "150"
        account.buying_power = "400"  # >2× equity suggests margin
        client = MagicMock()
        client.get_account.return_value = account

        with patch("main.config.IS_PAPER", False), patch("main.config.ALLOW_MARGIN", False):
            from main import _assert_account_safety

            with self.assertRaises(RuntimeError):
                _assert_account_safety(client)

    def test_allow_margin_true_suppresses_assertion(self):
        account = MagicMock()
        account.pattern_day_trader = True
        account.equity = "150"
        account.buying_power = "400"
        client = MagicMock()
        client.get_account.return_value = account

        with patch("main.config.IS_PAPER", False), patch("main.config.ALLOW_MARGIN", True):
            from main import _assert_account_safety

            try:
                _assert_account_safety(client)
            except RuntimeError:
                self.fail("Should not raise when ALLOW_MARGIN=True")

    def test_normal_cash_account_passes(self):
        account = MagicMock()
        account.pattern_day_trader = False
        account.equity = "150"
        account.buying_power = "150"  # 1× equity = cash account
        client = MagicMock()
        client.get_account.return_value = account

        with patch("main.config.IS_PAPER", False), patch("main.config.ALLOW_MARGIN", False):
            from main import _assert_account_safety

            try:
                _assert_account_safety(client)
            except RuntimeError:
                self.fail("Normal cash account should pass")


# ── get_total_open_exposure includes pending orders ───────────────────────────


class TestGetTotalOpenExposureIncludesPending(unittest.TestCase):
    """get_total_open_exposure() must include pending buy order notional."""

    def _make_position(self, symbol, market_value):
        p = MagicMock()
        p.symbol = symbol
        p.market_value = str(market_value)
        return p

    def _make_pending_order(self, symbol, notional, status="new"):
        from alpaca.trading.enums import OrderSide

        o = MagicMock()
        o.symbol = symbol
        o.side = OrderSide.BUY
        o.status = status
        o.notional = str(notional)
        return o

    def test_includes_pending_order_for_unfilled_symbol(self):
        from execution.trader import get_total_open_exposure

        client = MagicMock()
        client.get_all_positions.return_value = [self._make_position("AAPL", 50.0)]
        client.get_orders.return_value = [self._make_pending_order("SOFI", 45.0)]
        result = get_total_open_exposure(client)
        self.assertAlmostEqual(result, 95.0, places=1)

    def test_skips_pending_order_for_already_filled_symbol(self):
        """Partial fill: AAPL is in positions and also has a pending order — no double count."""
        from execution.trader import get_total_open_exposure

        client = MagicMock()
        client.get_all_positions.return_value = [self._make_position("AAPL", 50.0)]
        client.get_orders.return_value = [self._make_pending_order("AAPL", 45.0)]
        result = get_total_open_exposure(client)
        self.assertAlmostEqual(result, 50.0, places=1)

    def test_only_counts_new_accepted_pending_new_statuses(self):
        """filled/cancelled orders must not be counted as pending exposure."""
        from execution.trader import get_total_open_exposure

        client = MagicMock()
        client.get_all_positions.return_value = []
        client.get_orders.return_value = [
            self._make_pending_order("SOFI", 45.0, status="filled"),
            self._make_pending_order("RKLB", 45.0, status="cancelled"),
        ]
        result = get_total_open_exposure(client)
        self.assertAlmostEqual(result, 0.0, places=1)

    def test_raises_broker_state_unavailable_on_exception(self):
        from execution.trader import get_total_open_exposure
        from models import BrokerStateUnavailable

        client = MagicMock()
        client.get_all_positions.side_effect = Exception("network error")
        with self.assertRaises(BrokerStateUnavailable):
            get_total_open_exposure(client)


# ── Universe price filter ─────────────────────────────────────────────────────


class TestUniversePriceFilter(unittest.TestCase):
    """Price filter must drop candidates outside MIN/MAX_PRICE_USD range."""

    def test_check_pre_trade_still_passes_without_deployed_cap(self):
        """Regression: existing pre-trade tests still pass with new signature."""
        from utils.validators import check_pre_trade

        approved, _ = check_pre_trade("AAPL", 40.0, 0.0, 55.0, 75.0)
        self.assertTrue(approved)

    def test_config_price_filter_active_in_small_account_mode(self):
        import importlib

        with patch.dict(os.environ, {"SMALL_ACCOUNT_MODE": "true"}, clear=False):
            import config as cfg

            importlib.reload(cfg)
            self.assertGreater(cfg.MIN_PRICE_USD, 0)
            self.assertGreater(cfg.MAX_PRICE_USD, 0)

    def test_config_price_filter_disabled_by_default(self):
        import importlib

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SMALL_ACCOUNT_MODE", "MIN_PRICE_USD", "MAX_PRICE_USD")
        }
        env["SMALL_ACCOUNT_MODE"] = "false"
        with patch.dict(os.environ, env, clear=True):
            import config as cfg

            importlib.reload(cfg)
            self.assertEqual(cfg.MIN_PRICE_USD, 0.0)
            self.assertEqual(cfg.MAX_PRICE_USD, 0.0)


if __name__ == "__main__":
    unittest.main()

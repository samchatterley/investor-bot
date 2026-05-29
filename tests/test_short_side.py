"""Tests for short-side functionality: trader short functions, scanner, and DB migration."""

import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from models import OrderResult, OrderStatus

# ── helpers ──────────────────────────────────────────────────────────────────


def _mock_order(order_id="short-123", status="new", filled_qty=None, filled_avg_price="0.0"):
    o = MagicMock()
    o.id = order_id
    o.status = status
    o.filled_qty = filled_qty
    o.filled_avg_price = filled_avg_price
    return o


def _mock_position(symbol, qty, current_price=50.0):
    p = MagicMock()
    p.symbol = symbol
    p.qty = str(qty)
    p.current_price = str(current_price)
    p.market_value = str(abs(qty) * current_price)
    return p


def _snap(**kwargs):
    defaults = {
        "symbol": "WEAK",
        "current_price": 50.0,
        "rs_rank_pct": 10.0,
        "price_vs_ema21_pct": -3.0,
        "ema9_above_ema21": False,
        "rel_strength_20d": -7.0,
        "avg_volume": 1_000_000,
    }
    defaults.update(kwargs)
    return defaults


@contextmanager
def _meta_patcher(tmpdir):
    import utils.db as db_module

    db_path = os.path.join(tmpdir, "test.db")
    patchers = [
        patch.object(db_module, "_DB_PATH", db_path),
        patch.object(db_module, "_initialized", False),
        patch.object(db_module, "_migrate_json_state", lambda: None),
    ]
    for p in patchers:
        p.start()
    try:
        db_module.init_db()
        yield
    finally:
        for p in patchers:
            p.stop()


# ── place_short_order ─────────────────────────────────────────────────────────


class TestPlaceShortOrder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _client(self, order_id="short-123", fill_status="filled", fill_qty="5.0"):
        client = MagicMock()
        submitted = _mock_order(order_id=order_id)
        client.submit_order.return_value = submitted
        filled = _mock_order(
            order_id=order_id, status=fill_status, filled_qty=fill_qty, filled_avg_price="50.0"
        )
        client.get_order_by_id.return_value = filled
        return client

    def test_returns_none_when_qty_zero(self):
        from execution.trader import place_short_order

        result = place_short_order(MagicMock(), "WEAK", 0)
        self.assertIsNone(result)

    def test_filled_returns_filled_result(self):
        from execution.trader import place_short_order

        client = self._client()
        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.wait_for_fill", return_value=(5.0, 50.0)),
        ):
            result = place_short_order(client, "WEAK", 5)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(result.symbol, "WEAK")
        self.assertAlmostEqual(result.filled_qty, 5.0)

    def test_rejected_on_broker_exception(self):
        from execution.trader import place_short_order

        client = MagicMock()
        client.submit_order.side_effect = RuntimeError("margin required")
        with _meta_patcher(self.tmpdir):
            result = place_short_order(client, "WEAK", 5)
        self.assertEqual(result.status, OrderStatus.REJECTED)

    def test_timeout_when_fill_not_confirmed(self):
        from execution.trader import place_short_order

        client = self._client()
        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.wait_for_fill", return_value=None),
            patch.object(client, "get_order_by_id", side_effect=Exception("unavail")),
        ):
            result = place_short_order(client, "WEAK", 3)
        self.assertEqual(result.status, OrderStatus.TIMEOUT)

    def test_uses_sell_side(self):
        from alpaca.trading.enums import OrderSide

        from execution.trader import place_short_order

        client = self._client()
        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.wait_for_fill", return_value=(3.0, 50.0)),
        ):
            place_short_order(client, "WEAK", 3)
        submitted_req = client.submit_order.call_args[0][0]
        self.assertEqual(submitted_req.side, OrderSide.SELL)

    def test_client_order_id_contains_short(self):
        from execution.trader import place_short_order

        client = self._client()
        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.wait_for_fill", return_value=(2.0, 50.0)),
        ):
            place_short_order(client, "WEAK", 2)
        cid = client.submit_order.call_args[0][0].client_order_id
        self.assertIn("SHORT", cid)
        self.assertIn("WEAK", cid)


# ── place_short_cover_stop ────────────────────────────────────────────────────


class TestPlaceShortCoverStop(unittest.TestCase):
    def test_returns_none_for_zero_qty(self):
        from execution.trader import place_short_cover_stop

        self.assertIsNone(place_short_cover_stop(MagicMock(), "WEAK", 0))

    def test_returns_unprotected_for_sub_share(self):
        from execution.trader import place_short_cover_stop

        result = place_short_cover_stop(MagicMock(), "WEAK", 0.4)
        self.assertEqual(result.status, OrderStatus.UNPROTECTED)

    def test_filled_on_success(self):
        from execution.trader import place_short_cover_stop

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "cov-stop-1"
        client.submit_order.return_value = submitted
        result = place_short_cover_stop(client, "WEAK", 5)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(result.stop_order_id, "cov-stop-1")

    def test_uses_buy_side(self):
        from alpaca.trading.enums import OrderSide

        from execution.trader import place_short_cover_stop

        client = MagicMock()
        client.submit_order.return_value = MagicMock(id="x")
        place_short_cover_stop(client, "WEAK", 5)
        submitted_req = client.submit_order.call_args[0][0]
        self.assertEqual(submitted_req.side, OrderSide.BUY)

    def test_stop_failed_on_broker_exception(self):
        from execution.trader import place_short_cover_stop

        client = MagicMock()
        client.submit_order.side_effect = RuntimeError("no borrow")
        result = place_short_cover_stop(client, "WEAK", 5)
        self.assertEqual(result.status, OrderStatus.STOP_FAILED)

    def test_floors_fractional_qty(self):
        from execution.trader import place_short_cover_stop

        client = MagicMock()
        client.submit_order.return_value = MagicMock(id="x")
        place_short_cover_stop(client, "WEAK", 5.7)
        submitted_req = client.submit_order.call_args[0][0]
        self.assertEqual(submitted_req.qty, 5)

    def test_custom_trail_percent(self):
        from execution.trader import place_short_cover_stop

        client = MagicMock()
        client.submit_order.return_value = MagicMock(id="x")
        place_short_cover_stop(client, "WEAK", 5, trail_percent=6.0)
        submitted_req = client.submit_order.call_args[0][0]
        self.assertAlmostEqual(submitted_req.trail_percent, 6.0)


# ── record_short / record_cover / get_open_longs / get_open_shorts ───────────


class TestShortRecords(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _patch(self):
        return _meta_patcher(self.tmpdir)

    def test_record_short_and_get_open_shorts(self):
        from execution.trader import get_open_shorts, record_short
        from utils.db import init_db

        with self._patch():
            init_db()
            record_short("WEAK", 50.0, signal="rs_short", regime="BULL_TREND")
            shorts = get_open_shorts()
        self.assertIn("WEAK", shorts)

    def test_record_short_does_not_appear_in_longs(self):
        from execution.trader import get_open_longs, record_short
        from utils.db import init_db

        with self._patch():
            init_db()
            record_short("WEAK", 50.0)
            longs = get_open_longs()
        self.assertNotIn("WEAK", longs)

    def test_record_cover_removes_short(self):
        from execution.trader import get_open_shorts, record_cover, record_short
        from utils.db import init_db

        with self._patch():
            init_db()
            record_short("WEAK", 50.0)
            record_cover("WEAK")
            shorts = get_open_shorts()
        self.assertNotIn("WEAK", shorts)

    def test_record_buy_appears_in_longs_not_shorts(self):
        from execution.trader import get_open_longs, get_open_shorts, record_buy
        from utils.db import init_db

        with self._patch():
            init_db()
            record_buy("AAPL", 180.0, signal="mean_reversion")
            longs = get_open_longs()
            shorts = get_open_shorts()
        self.assertIn("AAPL", longs)
        self.assertNotIn("AAPL", shorts)

    def test_empty_when_no_positions(self):
        from execution.trader import get_open_longs, get_open_shorts
        from utils.db import init_db

        with self._patch():
            init_db()
            self.assertEqual(get_open_longs(), set())
            self.assertEqual(get_open_shorts(), set())


# ── get_short_notional / get_long_notional ────────────────────────────────────


class TestBookNotional(unittest.TestCase):
    def _client_with_positions(self, positions):
        client = MagicMock()
        client.get_all_positions.return_value = positions
        return client

    def test_short_notional_sums_short_positions(self):
        from execution.trader import get_short_notional

        positions = [
            _mock_position("WEAK", -10, 50.0),  # short: abs(market_value) = 500
            _mock_position("WEAK2", -5, 40.0),  # short: 200
            _mock_position("LONG1", 3, 100.0),  # long — excluded
        ]
        client = self._client_with_positions(positions)
        result = get_short_notional(client)
        self.assertAlmostEqual(result, 700.0, places=1)

    def test_long_notional_sums_long_positions(self):
        from execution.trader import get_long_notional

        positions = [
            _mock_position("AAPL", 5, 180.0),  # long: 900
            _mock_position("WEAK", -3, 50.0),  # short — excluded
        ]
        client = self._client_with_positions(positions)
        result = get_long_notional(client)
        self.assertAlmostEqual(result, 900.0, places=1)

    def test_returns_zero_on_exception(self):
        from execution.trader import get_long_notional, get_short_notional

        client = MagicMock()
        client.get_all_positions.side_effect = Exception("broker down")
        self.assertEqual(get_short_notional(client), 0.0)
        self.assertEqual(get_long_notional(client), 0.0)


# ── ensure_stops_attached (short-aware) ──────────────────────────────────────


class TestEnsureStopsAttachedShorts(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_attaches_cover_stop_for_unprotected_short(self):
        from execution.trader import ensure_stops_attached

        short_pos = _mock_position("WEAK", -5, 50.0)
        client = MagicMock()
        client.get_all_positions.return_value = [short_pos]
        client.get_orders.return_value = []

        with patch("execution.trader.place_short_cover_stop") as mock_cover:
            mock_cover.return_value = OrderResult(status=OrderStatus.FILLED, symbol="WEAK")
            with _meta_patcher(self.tmpdir):
                result = ensure_stops_attached(client)
        mock_cover.assert_called_once_with(client, "WEAK", 5)
        self.assertTrue(result)

    def test_does_not_double_attach_when_cover_stop_exists(self):
        from alpaca.trading.enums import OrderSide

        from execution.trader import ensure_stops_attached

        short_pos = _mock_position("WEAK", -5, 50.0)
        cover_order = MagicMock()
        cover_order.symbol = "WEAK"
        cover_order.side = OrderSide.BUY
        cover_order.type = "trailing_stop"
        cover_order.qty = "5"

        client = MagicMock()
        client.get_all_positions.return_value = [short_pos]
        client.get_orders.return_value = [cover_order]

        with (
            patch("execution.trader.place_short_cover_stop") as mock_cover,
            _meta_patcher(self.tmpdir),
        ):
            ensure_stops_attached(client)
        mock_cover.assert_not_called()

    def test_long_position_still_gets_sell_stop(self):
        from execution.trader import ensure_stops_attached

        long_pos = _mock_position("AAPL", 5, 180.0)

        client = MagicMock()
        client.get_all_positions.return_value = [long_pos]
        client.get_orders.return_value = []

        with patch("execution.trader.place_trailing_stop") as mock_stop:
            mock_stop.return_value = OrderResult(status=OrderStatus.FILLED, symbol="AAPL")
            with _meta_patcher(self.tmpdir):
                ensure_stops_attached(client)
        mock_stop.assert_called_once()
        # Verify it called place_trailing_stop (not the short cover version)
        self.assertEqual(mock_stop.call_args[0][1], "AAPL")

    def test_returns_false_when_cover_stop_fails(self):
        from execution.trader import ensure_stops_attached

        short_pos = _mock_position("WEAK", -5, 50.0)
        client = MagicMock()
        client.get_all_positions.return_value = [short_pos]
        client.get_orders.return_value = []

        with patch("execution.trader.place_short_cover_stop") as mock_cover:
            mock_cover.return_value = OrderResult(status=OrderStatus.STOP_FAILED, symbol="WEAK")
            with _meta_patcher(self.tmpdir):
                result = ensure_stops_attached(client)
        self.assertFalse(result)

    def test_sub_share_short_skipped_silently(self):
        from execution.trader import ensure_stops_attached

        short_pos = _mock_position("WEAK", -0.3, 50.0)
        client = MagicMock()
        client.get_all_positions.return_value = [short_pos]
        client.get_orders.return_value = []

        with (
            patch("execution.trader.place_short_cover_stop") as mock_cover,
            _meta_patcher(self.tmpdir),
        ):
            result = ensure_stops_attached(client)
        mock_cover.assert_not_called()
        self.assertTrue(result)

    def test_zero_qty_position_skipped_silently(self):
        # pos_qty == 0 → neither long nor short branch taken (670->646)
        from execution.trader import ensure_stops_attached

        zero_pos = _mock_position("FLAT", 0, 50.0)
        client = MagicMock()
        client.get_all_positions.return_value = [zero_pos]
        client.get_orders.return_value = []

        with _meta_patcher(self.tmpdir):
            result = ensure_stops_attached(client)
        self.assertTrue(result)


# ── scan_short_candidates ─────────────────────────────────────────────────────


class TestScanShortCandidates(unittest.TestCase):
    def test_returns_empty_for_blocked_regime(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=5.0, price_vs_ema21_pct=-2.0)]
        self.assertEqual(scan_short_candidates(snaps, "DEFENSIVE_DOWNTREND", set()), [])
        self.assertEqual(scan_short_candidates(snaps, "HIGH_VOL_DOWNTREND", set()), [])
        self.assertEqual(scan_short_candidates(snaps, "STRESS_RISK_OFF", set()), [])
        self.assertEqual(scan_short_candidates(snaps, None, set()), [])

    def test_returns_candidate_in_allowed_regime(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=10.0, price_vs_ema21_pct=-2.0)]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        self.assertEqual(len(result), 1)
        result2 = scan_short_candidates(snaps, "NEUTRAL_CHOP", set())
        self.assertEqual(len(result2), 1)

    def test_excludes_held_symbols(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(symbol="WEAK", rs_rank_pct=5.0, price_vs_ema21_pct=-2.0)]
        result = scan_short_candidates(snaps, "BULL_TREND", {"WEAK"})
        self.assertEqual(result, [])

    def test_excludes_above_quartile_threshold(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(symbol="MID", rs_rank_pct=25.0, price_vs_ema21_pct=-2.0),
            _snap(symbol="HIGH", rs_rank_pct=60.0, price_vs_ema21_pct=-2.0),
        ]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        self.assertEqual(result, [])

    def test_excludes_price_above_ema21(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=5.0, price_vs_ema21_pct=1.5)]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        self.assertEqual(result, [])

    def test_excludes_etfs(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(symbol="SPY", rs_rank_pct=5.0, price_vs_ema21_pct=-2.0)]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        self.assertEqual(result, [])

    def test_excludes_low_volume(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=5.0, price_vs_ema21_pct=-2.0, avg_volume=100_000)]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        self.assertEqual(result, [])

    def test_sorted_by_rs_rank_ascending(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(symbol="B", rs_rank_pct=20.0, price_vs_ema21_pct=-1.0),
            _snap(symbol="A", rs_rank_pct=5.0, price_vs_ema21_pct=-1.0),
            _snap(symbol="C", rs_rank_pct=15.0, price_vs_ema21_pct=-1.0),
        ]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        ranks = [r["rs_rank_pct"] for r in result]
        self.assertEqual(ranks, sorted(ranks))

    def test_missing_rs_rank_excluded(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(price_vs_ema21_pct=-2.0)]
        del snaps[0]["rs_rank_pct"]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        self.assertEqual(result, [])

    def test_returns_multiple_candidates(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(symbol="W1", rs_rank_pct=3.0, price_vs_ema21_pct=-2.0),
            _snap(symbol="W2", rs_rank_pct=12.0, price_vs_ema21_pct=-1.0),
            _snap(symbol="W3", rs_rank_pct=24.9, price_vs_ema21_pct=-0.5),
        ]
        result = scan_short_candidates(snaps, "NEUTRAL_CHOP", set())
        self.assertEqual(len(result), 3)

    def test_requires_at_least_one_bearish_signal(self):
        """Stock with no signals is excluded even when it passes all hard gates."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(
                rs_rank_pct=10.0,
                price_vs_ema21_pct=-1.0,
                ema9_above_ema21=True,  # EMA slope up — no ema_breakdown
                rel_strength_20d=-2.0,  # not weak enough — no loser_momentum
                earnings_miss_candidate=False,
            )
        ]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        self.assertEqual(result, [])

    def test_candidate_carries_signal_metadata(self):
        """Result candidates include key_signal, matched_signals, and confidence."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap()]
        result = scan_short_candidates(snaps, "BULL_TREND", set())
        self.assertEqual(len(result), 1)
        c = result[0]
        self.assertIn("key_signal", c)
        self.assertIn("matched_signals", c)
        self.assertIn("confidence", c)
        self.assertIsInstance(c["matched_signals"], list)
        self.assertGreater(len(c["matched_signals"]), 0)
        self.assertGreaterEqual(c["confidence"], 0)

    def test_sorted_by_signal_count_then_rs(self):
        """More signals = higher priority; equal signals sorted by weakest RS."""
        from execution.stock_scanner import scan_short_candidates

        strong = _snap(
            symbol="STRONG",
            rs_rank_pct=5.0,
            price_vs_ema21_pct=-3.0,
            ema9_above_ema21=False,
            rel_strength_20d=-8.0,
        )
        weak_only = _snap(
            symbol="WEAK_RS",
            rs_rank_pct=15.0,
            price_vs_ema21_pct=-1.0,
            ema9_above_ema21=True,
            rel_strength_20d=-6.0,
        )
        result = scan_short_candidates([weak_only, strong], "BULL_TREND", set())
        self.assertEqual(result[0]["symbol"], "STRONG")


# ── evaluate_short_signals ────────────────────────────────────────────────────


class TestEvaluateShortSignals(unittest.TestCase):
    def test_loser_momentum_fires_on_weak_rel_strength(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"rel_strength_20d": -6.0}
        self.assertIn("loser_momentum", evaluate_short_signals(snap))

    def test_loser_momentum_not_fire_on_moderate_underperformance(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"rel_strength_20d": -3.0}
        self.assertNotIn("loser_momentum", evaluate_short_signals(snap))

    def test_loser_momentum_absent_when_rel_strength_missing(self):
        from signals.evaluator import evaluate_short_signals

        self.assertNotIn("loser_momentum", evaluate_short_signals({}))

    def test_ema_breakdown_fires_when_solidly_below_and_slope_down(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"price_vs_ema21_pct": -3.0, "ema9_above_ema21": False}
        self.assertIn("ema_breakdown", evaluate_short_signals(snap))

    def test_ema_breakdown_not_fire_when_barely_below(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"price_vs_ema21_pct": -1.0, "ema9_above_ema21": False}
        self.assertNotIn("ema_breakdown", evaluate_short_signals(snap))

    def test_ema_breakdown_not_fire_when_ema_slope_up(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"price_vs_ema21_pct": -3.0, "ema9_above_ema21": True}
        self.assertNotIn("ema_breakdown", evaluate_short_signals(snap))

    def test_earnings_miss_fires_on_candidate(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"earnings_miss_candidate": True}
        self.assertIn("earnings_miss", evaluate_short_signals(snap))

    def test_earnings_miss_not_fire_without_flag(self):
        from signals.evaluator import evaluate_short_signals

        self.assertNotIn("earnings_miss", evaluate_short_signals({}))

    def test_returns_empty_when_no_signals(self):
        from signals.evaluator import evaluate_short_signals

        snap = {
            "rel_strength_20d": -2.0,
            "price_vs_ema21_pct": -1.0,
            "ema9_above_ema21": True,
        }
        self.assertEqual(evaluate_short_signals(snap), [])

    def test_sorted_by_priority_earnings_miss_first(self):
        from signals.evaluator import evaluate_short_signals

        snap = {
            "earnings_miss_candidate": True,
            "rel_strength_20d": -8.0,
            "price_vs_ema21_pct": -3.0,
            "ema9_above_ema21": False,
        }
        result = evaluate_short_signals(snap)
        self.assertEqual(result[0], "earnings_miss")

    def test_blocked_signal_excluded(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"rel_strength_20d": -8.0}
        self.assertNotIn(
            "loser_momentum",
            evaluate_short_signals(snap, blocked=frozenset({"loser_momentum"})),
        )

    def test_custom_params_override_threshold(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"rel_strength_20d": -3.0}
        self.assertIn(
            "loser_momentum",
            evaluate_short_signals(snap, params={"loser_mom_threshold": -2.0}),
        )

    def test_high_short_interest_fires_when_flag_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"high_short_interest": True}
        self.assertIn("high_short_interest", evaluate_short_signals(snap))

    def test_high_short_interest_not_fire_without_flag(self):
        from signals.evaluator import evaluate_short_signals

        self.assertNotIn("high_short_interest", evaluate_short_signals({}))

    def test_high_short_interest_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"high_short_interest": True}
        self.assertNotIn(
            "high_short_interest",
            evaluate_short_signals(snap, blocked=frozenset({"high_short_interest"})),
        )

    def test_priority_order_earnings_miss_before_short_interest(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"earnings_miss_candidate": True, "high_short_interest": True}
        result = evaluate_short_signals(snap)
        self.assertEqual(result[0], "earnings_miss")
        self.assertEqual(result[1], "high_short_interest")

    def test_priority_order_short_interest_before_loser_momentum(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"high_short_interest": True, "rel_strength_20d": -8.0}
        result = evaluate_short_signals(snap)
        self.assertEqual(result[0], "high_short_interest")
        self.assertEqual(result[1], "loser_momentum")


# ── DB migration 5: side column ───────────────────────────────────────────────


class TestSideMigration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_existing_positions_default_to_long(self):
        from execution.trader import get_open_longs, record_buy
        from utils.db import init_db

        with _meta_patcher(self.tmpdir):
            init_db()
            record_buy("AAPL", 180.0)
            longs = get_open_longs()
        self.assertIn("AAPL", longs)

    def test_short_and_long_coexist_separately(self):
        from execution.trader import get_open_longs, get_open_shorts, record_buy, record_short
        from utils.db import init_db

        with _meta_patcher(self.tmpdir):
            init_db()
            record_buy("AAPL", 180.0)
            record_short("WEAK", 50.0)
            longs = get_open_longs()
            shorts = get_open_shorts()
        self.assertIn("AAPL", longs)
        self.assertNotIn("WEAK", longs)
        self.assertIn("WEAK", shorts)
        self.assertNotIn("AAPL", shorts)


# ── Error-path branches ───────────────────────────────────────────────────────


class TestPlaceShortOrderErrorPaths(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_ledger_import_failure_still_places_order(self):
        import sys

        from execution.trader import place_short_order

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "s1"
        client.submit_order.return_value = submitted
        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.wait_for_fill", return_value=(5.0, 50.0)),
            patch.dict(sys.modules, {"utils.order_ledger": None}),
        ):
            result = place_short_order(client, "WEAK", 5)
        self.assertEqual(result.status, OrderStatus.FILLED)

    def test_raises_ledger_unavailable_on_live_when_intent_fails(self):
        from execution.trader import place_short_order
        from models import OrderLedgerUnavailable

        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.IS_PAPER", False),
            patch("utils.order_ledger.create_intent", return_value=None),
            self.assertRaises(OrderLedgerUnavailable),
        ):
            place_short_order(MagicMock(), "WEAK", 5)

    def test_timeout_without_ledger(self):
        # trader.py 949->952: _ledger=False path when wait_for_fill times out
        import sys

        from execution.trader import place_short_order

        client = MagicMock()
        submitted = MagicMock()
        submitted.id = "s1"
        client.submit_order.return_value = submitted
        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.wait_for_fill", return_value=None),
            patch.dict(sys.modules, {"utils.order_ledger": None}),
        ):
            result = place_short_order(client, "WEAK", 5)
        self.assertEqual(result.status, OrderStatus.TIMEOUT)

    def test_exception_without_ledger(self):
        # trader.py 957->960: _ledger=False path when submit_order raises
        import sys

        from execution.trader import place_short_order

        client = MagicMock()
        client.submit_order.side_effect = RuntimeError("broker down")
        with (
            _meta_patcher(self.tmpdir),
            patch.dict(sys.modules, {"utils.order_ledger": None}),
        ):
            result = place_short_order(client, "WEAK", 5)
        self.assertEqual(result.status, OrderStatus.REJECTED)


class TestGetOpenPositionsDbErrors(unittest.TestCase):
    def test_get_open_longs_returns_empty_on_db_error(self):
        from execution.trader import get_open_longs

        with patch("execution.trader._db", side_effect=Exception("db down")):
            self.assertEqual(get_open_longs(), set())

    def test_get_open_shorts_returns_empty_on_db_error(self):
        from execution.trader import get_open_shorts

        with patch("execution.trader._db", side_effect=Exception("db down")):
            self.assertEqual(get_open_shorts(), set())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

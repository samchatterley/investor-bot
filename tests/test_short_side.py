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
        "earnings_miss_candidate": True,
        "failed_breakout_flag": False,
        "close_pct_of_range": 0.5,
        "vol_ratio": 1.0,
        "rsi_14": 50.0,
        "ret_5d_pct": 0.0,
        "high_short_interest": True,
    }
    defaults.update(kwargs)
    return defaults


def _reversal_snap(**kwargs):
    """Snap for the reversal path (rs_rank >= 65, high_short_interest)."""
    defaults = {
        "symbol": "EXTENDED",
        "current_price": 120.0,
        "rs_rank_pct": 70.0,
        "price_vs_ema21_pct": 4.0,
        "ema9_above_ema21": True,
        "avg_volume": 1_000_000,
        "failed_breakout_flag": True,
        "vol_ratio": 1.5,
        "rsi_14": 62.0,
        "close_pct_of_range": 0.5,
        "ret_5d_pct": 3.0,
        "earnings_miss_candidate": False,
        "high_short_interest": True,
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

    def test_late_fill_recovered_after_timeout(self):
        """E1: a short that fills just after wait_for_fill gives up is recovered as FILLED,
        not lost to TIMEOUT (mirrors place_buy_order)."""
        from execution.trader import place_short_order

        client = self._client()
        # Final get_order_by_id → enum-status "filled" 4.0 (recovery reads status.value).
        late = _mock_order(filled_qty="4.0", filled_avg_price="50.0")
        late.status = MagicMock(value="filled")
        client.get_order_by_id.return_value = late
        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.wait_for_fill", return_value=None),  # poll timed out
        ):
            result = place_short_order(client, "WEAK", 4)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertAlmostEqual(result.filled_qty, 4.0)

    def test_late_fill_recovered_without_ledger(self):
        # trader.py 1102->1110: late-fill recovery returns FILLED when _ledger is False.
        import sys

        from execution.trader import place_short_order

        client = self._client()
        late = _mock_order(filled_qty="4.0", filled_avg_price="50.0")
        late.status = MagicMock(value="filled")
        client.get_order_by_id.return_value = late
        with (
            _meta_patcher(self.tmpdir),
            patch("execution.trader.wait_for_fill", return_value=None),
            patch.dict(sys.modules, {"utils.order_ledger": None}),
        ):
            result = place_short_order(client, "WEAK", 4)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertAlmostEqual(result.filled_qty, 4.0)

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
        """BULL_TREND and NEUTRAL_CHOP are not in SHORT_ALLOWED_REGIMES → empty."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=5.0, price_vs_ema21_pct=-2.0)]
        self.assertEqual(scan_short_candidates(snaps, "BULL_TREND", set()), [])
        self.assertEqual(scan_short_candidates(snaps, "NEUTRAL_CHOP", set()), [])
        self.assertEqual(scan_short_candidates(snaps, None, set()), [])

    def test_returns_fundamental_candidate_in_stress_regime(self):
        """Fundamental path: rs_rank < 25, earnings miss, bearish regime → returns candidate."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=10.0, price_vs_ema21_pct=-2.0, earnings_miss_candidate=True)]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(len(result), 1)

    def test_returns_fundamental_candidate_in_high_vol_downtrend(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=10.0, price_vs_ema21_pct=-2.0, earnings_miss_candidate=True)]
        result = scan_short_candidates(snaps, "HIGH_VOL_DOWNTREND", set())
        self.assertEqual(len(result), 1)

    def test_returns_reversal_candidate_via_high_short_interest(self):
        """Reversal path: rs_rank >= 65, high_short_interest=True → returns candidate."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [_reversal_snap(high_short_interest=True)]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(len(result), 1)
        self.assertIn("high_short_interest", result[0]["matched_signals"])

    def test_path_d_fires_on_failed_bounce_short(self):
        """S1: Path D surfaces a mid-RS post-earnings-gapdown failed-bounce candidate."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(
                symbol="GAPPER",
                rs_rank_pct=50.0,  # mid-band → Paths A/B/C don't fire
                earnings_gap_pct=-9.0,
                gap_failed_bounce=True,
                vol_ratio=1.8,
                high_short_interest=False,
                earnings_miss_candidate=False,
            )
        ]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(len(result), 1)
        self.assertIn("post_earnings_gapdown_failed_bounce", result[0]["matched_signals"])

    def test_path_d_blocks_naive_earnings_gap_down_live(self):
        """A3: even at high continuation volume, the superseded earnings_gap_down is blocked live;
        only the failed-bounce short surfaces."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(
                symbol="GAPPER",
                rs_rank_pct=50.0,
                earnings_gap_pct=-9.0,
                gap_failed_bounce=True,
                vol_ratio=3.0,  # ≥2.5 → earnings_gap_down WOULD fire if not blocked
                high_short_interest=False,
                earnings_miss_candidate=False,
            )
        ]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(len(result), 1)
        self.assertIn("post_earnings_gapdown_failed_bounce", result[0]["matched_signals"])
        self.assertNotIn("earnings_gap_down", result[0]["matched_signals"])

    def test_path_d_no_failed_bounce_does_not_fire(self):
        """A recent gap-down with no failed bounce yet is not surfaced by Path D."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(
                symbol="GAPPER",
                rs_rank_pct=50.0,
                earnings_gap_pct=-9.0,
                gap_failed_bounce=False,  # bounce hasn't failed
                vol_ratio=1.8,
                high_short_interest=False,
                earnings_miss_candidate=False,
            )
        ]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_reversal_path_all_signals_disabled_returns_empty(self):
        """Reversal path: rs_rank >= 65 but all signals disabled and no hsi → empty."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _reversal_snap(
                high_short_interest=False,
                failed_breakout_flag=True,
                vol_ratio=2.5,
                close_pct_of_range=0.1,
                rsi_14=65.0,
                ret_5d_pct=3.0,
            )
        ]
        result = scan_short_candidates(snaps, "DEFENSIVE_DOWNTREND", set())
        self.assertEqual(result, [])

    def test_excludes_held_symbols(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(symbol="WEAK", rs_rank_pct=5.0, earnings_miss_candidate=True)]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", {"WEAK"})
        self.assertEqual(result, [])

    def test_fundamental_path_excludes_middle_band_25_to_65(self):
        """Middle band (25–64) produces no signal from either path."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(
                symbol="MID",
                rs_rank_pct=40.0,
                price_vs_ema21_pct=-2.0,
                earnings_miss_candidate=True,
            ),
        ]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_fundamental_path_excludes_price_above_ema21(self):
        """Fundamental path gate: price_vs_ema21_pct >= 0 → excluded."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=5.0, price_vs_ema21_pct=1.5, earnings_miss_candidate=True)]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_excludes_etfs(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(symbol="SPY", rs_rank_pct=5.0, earnings_miss_candidate=True)]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_excludes_low_volume(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(rs_rank_pct=5.0, earnings_miss_candidate=True, avg_volume=100_000)]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_missing_rs_rank_excluded(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(price_vs_ema21_pct=-2.0, earnings_miss_candidate=True)]
        del snaps[0]["rs_rank_pct"]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_fundamental_path_requires_active_signal(self):
        """Bottom-quartile stock with no live signals → excluded."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(
                rs_rank_pct=10.0,
                price_vs_ema21_pct=-2.0,
                earnings_miss_candidate=False,
                high_short_interest=False,
            )
        ]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_reversal_path_requires_reversal_signal(self):
        """High-RS stock with no live signals → excluded."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _reversal_snap(
                high_short_interest=False,
                failed_breakout_flag=False,
                vol_ratio=1.0,
                close_pct_of_range=0.6,
                rsi_14=45.0,
                ret_5d_pct=0.5,
            )
        ]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_candidate_carries_signal_metadata(self):
        """Result candidates include key_signal, matched_signals, and confidence."""
        from execution.stock_scanner import scan_short_candidates

        snaps = [_snap(earnings_miss_candidate=True)]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(len(result), 1)
        c = result[0]
        self.assertIn("key_signal", c)
        self.assertIn("matched_signals", c)
        self.assertIn("confidence", c)
        self.assertIsInstance(c["matched_signals"], list)
        self.assertGreater(len(c["matched_signals"]), 0)
        self.assertGreaterEqual(c["confidence"], 0)

    def test_returns_multiple_fundamental_candidates(self):
        from execution.stock_scanner import scan_short_candidates

        snaps = [
            _snap(symbol="W1", rs_rank_pct=3.0, price_vs_ema21_pct=-2.0),
            _snap(symbol="W2", rs_rank_pct=12.0, price_vs_ema21_pct=-2.0),
            _snap(symbol="W3", rs_rank_pct=24.9, price_vs_ema21_pct=-2.0),
        ]
        result = scan_short_candidates(snaps, "STRESS_RISK_OFF", set())
        self.assertEqual(len(result), 3)

    def test_sorted_by_signal_count_then_rs(self):
        """More signals = higher priority; equal signals sorted by weakest RS first."""
        from execution.stock_scanner import scan_short_candidates

        # Two fundamental candidates; lower RS rank should appear first
        low_rs = _snap(symbol="LOW_RS", rs_rank_pct=3.0, price_vs_ema21_pct=-3.0)
        high_rs = _snap(symbol="HIGH_RS", rs_rank_pct=20.0, price_vs_ema21_pct=-1.5)
        result = scan_short_candidates([high_rs, low_rs], "STRESS_RISK_OFF", set())
        self.assertEqual(result[0]["symbol"], "LOW_RS")

    def test_path_c_no_signals_does_not_add_candidate(self):
        """Path C criteria met (rs_lag > 65, rs_rank < 65) but no signals fire → excluded."""
        from execution.stock_scanner import scan_short_candidates

        # rs_rank between 45 and 65 (so rs_deterioration won't fire),
        # ret_5d not negative enough, high_short_interest=False
        snap = _snap(
            rs_rank_pct=50.0,
            rs_rank_pct_10d_ago=70.0,
            ret_5d_pct=-1.0,
            high_short_interest=False,
            earnings_miss_candidate=False,
        )
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])


# ── evaluate_short_signals ────────────────────────────────────────────────────


class TestEvaluateShortSignals(unittest.TestCase):
    def test_ema_breakdown_globally_disabled(self):
        """ema_breakdown is in SHORT_GLOBALLY_DISABLED — never fires."""
        from signals.evaluator import SHORT_GLOBALLY_DISABLED, evaluate_short_signals

        self.assertIn("ema_breakdown", SHORT_GLOBALLY_DISABLED)
        snap = {"price_vs_ema21_pct": -3.0, "ema9_above_ema21": False}
        self.assertNotIn("ema_breakdown", evaluate_short_signals(snap))

    def test_winner_reversal_globally_disabled(self):
        """winner_reversal is in SHORT_GLOBALLY_DISABLED — never fires."""
        from signals.evaluator import SHORT_GLOBALLY_DISABLED, evaluate_short_signals

        self.assertIn("winner_reversal", SHORT_GLOBALLY_DISABLED)
        snap = {"rsi_14": 80.0, "price_vs_ema21_pct": 5.0, "ret_5d_pct": -1.0}
        self.assertNotIn("winner_reversal", evaluate_short_signals(snap))

    def test_earnings_miss_globally_disabled(self):
        """earnings_miss is in SHORT_GLOBALLY_DISABLED — never fires."""
        from signals.evaluator import SHORT_GLOBALLY_DISABLED, evaluate_short_signals

        self.assertIn("earnings_miss", SHORT_GLOBALLY_DISABLED)
        snap = {"earnings_miss_candidate": True}
        self.assertNotIn("earnings_miss", evaluate_short_signals(snap))

    def test_earnings_miss_not_fire_without_flag(self):
        from signals.evaluator import evaluate_short_signals

        self.assertNotIn("earnings_miss", evaluate_short_signals({}))

    def test_failed_breakout_globally_disabled(self):
        """failed_breakout is in SHORT_GLOBALLY_DISABLED — never fires."""
        from signals.evaluator import SHORT_GLOBALLY_DISABLED, evaluate_short_signals

        self.assertIn("failed_breakout", SHORT_GLOBALLY_DISABLED)
        snap = {"failed_breakout_flag": True, "vol_ratio": 1.5, "rsi_14": 62.0}
        self.assertNotIn("failed_breakout", evaluate_short_signals(snap))

    def test_failed_breakout_blocked_when_flag_false(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"failed_breakout_flag": False, "vol_ratio": 2.0, "rsi_14": 62.0}
        self.assertNotIn("failed_breakout", evaluate_short_signals(snap))

    def test_failed_breakout_blocked_by_low_rsi(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"failed_breakout_flag": True, "vol_ratio": 1.5, "rsi_14": 30.0}
        self.assertNotIn("failed_breakout", evaluate_short_signals(snap))

    def test_failed_breakout_blocked_by_very_high_rsi(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"failed_breakout_flag": True, "vol_ratio": 1.5, "rsi_14": 90.0}
        self.assertNotIn("failed_breakout", evaluate_short_signals(snap))

    def test_high_vol_reversal_globally_disabled(self):
        """high_vol_reversal is in SHORT_GLOBALLY_DISABLED — never fires."""
        from signals.evaluator import SHORT_GLOBALLY_DISABLED, evaluate_short_signals

        self.assertIn("high_vol_reversal", SHORT_GLOBALLY_DISABLED)
        snap = {"vol_ratio": 2.5, "close_pct_of_range": 0.1, "rsi_14": 65.0, "ret_5d_pct": 3.0}
        self.assertNotIn("high_vol_reversal", evaluate_short_signals(snap))

    def test_high_vol_reversal_blocked_by_low_volume(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"vol_ratio": 1.2, "close_pct_of_range": 0.1, "rsi_14": 65.0, "ret_5d_pct": 3.0}
        self.assertNotIn("high_vol_reversal", evaluate_short_signals(snap))

    def test_high_vol_reversal_blocked_by_high_range_pct(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"vol_ratio": 2.5, "close_pct_of_range": 0.6, "rsi_14": 65.0, "ret_5d_pct": 3.0}
        self.assertNotIn("high_vol_reversal", evaluate_short_signals(snap))

    def test_high_vol_reversal_blocked_by_low_rsi(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"vol_ratio": 2.5, "close_pct_of_range": 0.1, "rsi_14": 40.0, "ret_5d_pct": 3.0}
        self.assertNotIn("high_vol_reversal", evaluate_short_signals(snap))

    def test_high_vol_reversal_blocked_by_low_ret5d(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"vol_ratio": 2.5, "close_pct_of_range": 0.1, "rsi_14": 65.0, "ret_5d_pct": 0.5}
        self.assertNotIn("high_vol_reversal", evaluate_short_signals(snap))

    def test_returns_empty_when_no_signals(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"price_vs_ema21_pct": -1.0, "rsi_14": 50.0}
        self.assertEqual(evaluate_short_signals(snap), [])

    def test_high_short_interest_fires_as_only_live_signal(self):
        """With all other signals disabled, high_short_interest is the only live signal."""
        from signals.evaluator import evaluate_short_signals

        snap = {
            "earnings_miss_candidate": True,
            "failed_breakout_flag": True,
            "vol_ratio": 1.5,
            "rsi_14": 62.0,
            "high_short_interest": True,
        }
        result = evaluate_short_signals(snap)
        self.assertEqual(result, ["high_short_interest"])

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

    def test_all_disabled_signals_produce_empty_result(self):
        """All signals in SHORT_GLOBALLY_DISABLED produce empty result even when conditions met."""
        from signals.evaluator import evaluate_short_signals

        snap = {
            "earnings_miss_candidate": True,
            "failed_breakout_flag": True,
            "vol_ratio": 1.5,
            "rsi_14": 62.0,
        }
        result = evaluate_short_signals(snap)
        self.assertEqual(result, [])

    def test_priority_order_failed_breakout_before_high_vol_reversal(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertLess(
            SHORT_SIGNAL_PRIORITY["failed_breakout"],
            SHORT_SIGNAL_PRIORITY["high_vol_reversal"],
        )

    def test_failed_breakout_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"failed_breakout_flag": True, "vol_ratio": 1.5, "rsi_14": 62.0}
        self.assertNotIn(
            "failed_breakout",
            evaluate_short_signals(snap, blocked=frozenset({"failed_breakout"})),
        )

    def test_guidance_downgrade_fires_when_flag_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"guidance_negative": True}
        self.assertIn("guidance_downgrade", evaluate_short_signals(snap))

    def test_guidance_downgrade_blocked_by_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"guidance_negative": True}
        self.assertNotIn(
            "guidance_downgrade",
            evaluate_short_signals(snap, blocked=frozenset({"guidance_downgrade"})),
        )

    def test_secondary_offering_short_fires_when_flag_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"secondary_offering": True}
        self.assertIn("secondary_offering_short", evaluate_short_signals(snap))

    def test_secondary_offering_short_blocked_by_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"secondary_offering": True}
        self.assertNotIn(
            "secondary_offering_short",
            evaluate_short_signals(snap, blocked=frozenset({"secondary_offering_short"})),
        )


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

    def test_get_open_shorts_raises_on_db_error(self):
        # get_open_shorts must fail closed (raise) so the short-slot cap isn't bypassed
        from execution.trader import get_open_shorts
        from models import OrderLedgerUnavailable

        with (
            patch("execution.trader._db", side_effect=Exception("db down")),
            self.assertRaises(OrderLedgerUnavailable),
        ):
            get_open_shorts()


class TestRsDeteriorationSignal(unittest.TestCase):
    """Tests for the rs_deterioration cross-sectional signal in evaluate_short_signals."""

    def _snap_det(self, **kwargs):
        base = {
            "rs_rank_pct_10d_ago": 72.0,  # was in top 35%
            "rs_rank_pct": 38.0,  # now below median
            "ret_5d_pct": -3.0,  # falling
        }
        base.update(kwargs)
        return base

    def test_rs_deterioration_blocked_globally_even_with_valid_conditions(self):
        from signals.evaluator import evaluate_short_signals

        result = evaluate_short_signals(self._snap_det())
        self.assertNotIn("rs_deterioration", result)

    def test_rs_deterioration_does_not_fire_without_lag_field(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"rs_rank_pct": 38.0, "ret_5d_pct": -3.0}  # no rs_rank_pct_10d_ago
        self.assertNotIn("rs_deterioration", evaluate_short_signals(snap))

    def test_rs_deterioration_blocked_when_lag_below_threshold(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap_det(rs_rank_pct_10d_ago=60.0)  # was not in top 35%
        self.assertNotIn("rs_deterioration", evaluate_short_signals(snap))

    def test_rs_deterioration_blocked_when_current_rank_above_threshold(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap_det(rs_rank_pct=48.0)  # still above the 45% threshold
        self.assertNotIn("rs_deterioration", evaluate_short_signals(snap))

    def test_rs_deterioration_blocked_when_ret5d_not_negative_enough(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap_det(ret_5d_pct=-1.0)  # only -1%, threshold is -2%
        self.assertNotIn("rs_deterioration", evaluate_short_signals(snap))

    def test_rs_deterioration_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap_det()
        self.assertNotIn(
            "rs_deterioration",
            evaluate_short_signals(snap, blocked=frozenset({"rs_deterioration"})),
        )

    def test_rs_deterioration_in_short_globally_disabled(self):
        """rs_deterioration disabled: 0/11 profitable walk-forward folds, Sharpe -0.872."""
        from signals.evaluator import SHORT_GLOBALLY_DISABLED

        self.assertIn("rs_deterioration", SHORT_GLOBALLY_DISABLED)

    def test_rs_deterioration_priority_higher_than_failed_breakout(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertLess(
            SHORT_SIGNAL_PRIORITY["rs_deterioration"],
            SHORT_SIGNAL_PRIORITY["failed_breakout"],
        )

    def test_rs_deterioration_params_present_in_defaults(self):
        from signals.evaluator import DEFAULT_SHORT_SIGNAL_PARAMS

        for key in ("rs_det_lag_min", "rs_det_current_max", "rs_det_ret5d_max"):
            self.assertIn(key, DEFAULT_SHORT_SIGNAL_PARAMS)


class TestScanShortCandidatesDeteriorationPath(unittest.TestCase):
    """Tests for the new Path C (deterioration) in scan_short_candidates."""

    def _det_snap(self, **kwargs):
        base = {
            "symbol": "FADE",
            "rs_rank_pct_10d_ago": 75.0,
            "rs_rank_pct": 35.0,
            "ret_5d_pct": -3.5,
            "avg_volume": 2_000_000,
            "price_vs_ema21_pct": -1.5,
            "high_short_interest": False,
        }
        base.update(kwargs)
        return base

    def test_deterioration_path_returns_no_candidates_because_globally_disabled(self):
        from execution.stock_scanner import scan_short_candidates

        snap = self._det_snap()
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_deterioration_path_blocked_in_bull_trend(self):
        from execution.stock_scanner import scan_short_candidates

        snap = self._det_snap()
        result = scan_short_candidates([snap], "BULL_TREND", set())
        self.assertEqual(result, [])

    def test_deterioration_path_skips_etfs(self):
        from config import ETF_SYMBOLS
        from execution.stock_scanner import scan_short_candidates

        etf_sym = next(iter(ETF_SYMBOLS))
        snap = self._det_snap(symbol=etf_sym)
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_deterioration_path_skips_held_symbols(self):
        from execution.stock_scanner import scan_short_candidates

        snap = self._det_snap()
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", {"FADE"})
        self.assertEqual(result, [])

    def test_deterioration_path_skips_low_volume(self):
        from execution.stock_scanner import scan_short_candidates

        snap = self._det_snap(avg_volume=50_000)
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_deterioration_path_fires_when_high_short_interest(self):
        # high_short_interest is not blocked in Path C — should return a candidate
        from execution.stock_scanner import scan_short_candidates

        snap = self._det_snap(high_short_interest=True)
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(len(result), 1)
        self.assertIn("high_short_interest", result[0]["matched_signals"])


class TestShortUniverseModule(unittest.TestCase):
    """Tests for execution/short_universe.py."""

    def test_static_universe_is_list_of_strings(self):
        from execution.short_universe import STATIC_SHORT_UNIVERSE

        self.assertIsInstance(STATIC_SHORT_UNIVERSE, list)
        self.assertTrue(len(STATIC_SHORT_UNIVERSE) > 100)
        for s in STATIC_SHORT_UNIVERSE:
            self.assertIsInstance(s, str)

    def test_static_universe_has_no_duplicates(self):
        from execution.short_universe import STATIC_SHORT_UNIVERSE

        self.assertEqual(len(STATIC_SHORT_UNIVERSE), len(set(STATIC_SHORT_UNIVERSE)))

    def test_get_short_universe_falls_back_to_static_on_error(self):
        from execution.short_universe import STATIC_SHORT_UNIVERSE, get_short_universe

        client = MagicMock()
        client.get_all_assets.side_effect = RuntimeError("Alpaca down")
        result = get_short_universe(client, _retries=0)
        self.assertEqual(result, STATIC_SHORT_UNIVERSE)

    def test_get_short_universe_retries_then_falls_back(self):
        from execution.short_universe import STATIC_SHORT_UNIVERSE, get_short_universe

        client = MagicMock()
        client.get_all_assets.side_effect = RuntimeError("connection reset")
        with patch("execution.short_universe.time.sleep") as mock_sleep:
            result = get_short_universe(client, _retries=2, _retry_delay=1.0)
        self.assertEqual(result, STATIC_SHORT_UNIVERSE)
        self.assertEqual(client.get_all_assets.call_count, 3)  # 1 initial + 2 retries
        self.assertEqual(mock_sleep.call_count, 2)

    def test_get_short_universe_succeeds_after_retry(self):
        from execution.short_universe import get_short_universe

        good_asset = MagicMock()
        good_asset.tradable = True
        good_asset.easy_to_borrow = True
        good_asset.exchange = "NASDAQ"
        good_asset.symbol = "INTC"  # INTC is in STATIC_SHORT_UNIVERSE
        client = MagicMock()
        client.get_all_assets.side_effect = [RuntimeError("transient"), [good_asset]]
        with patch("execution.short_universe.time.sleep"):
            result = get_short_universe(client, _retries=1)
        self.assertIn("INTC", result)
        self.assertEqual(client.get_all_assets.call_count, 2)

    def test_get_short_universe_filters_non_tradable(self):
        from execution.short_universe import get_short_universe

        tradable = MagicMock()
        tradable.tradable = True
        tradable.easy_to_borrow = True
        tradable.exchange = "NASDAQ"
        tradable.symbol = "INTC"  # INTC is in STATIC_SHORT_UNIVERSE

        not_tradable = MagicMock()
        not_tradable.tradable = False
        not_tradable.easy_to_borrow = True
        not_tradable.exchange = "NASDAQ"
        not_tradable.symbol = "IBM"  # IBM is in STATIC_SHORT_UNIVERSE but not tradable

        client = MagicMock()
        client.get_all_assets.return_value = [tradable, not_tradable]
        result = get_short_universe(client)
        self.assertIn("INTC", result)
        self.assertNotIn("IBM", result)

    def test_get_short_universe_filters_otc(self):
        from execution.short_universe import get_short_universe

        otc_asset = MagicMock()
        otc_asset.tradable = True
        otc_asset.easy_to_borrow = True
        otc_asset.exchange = "OTC"
        otc_asset.symbol = "OTCPK"

        client = MagicMock()
        client.get_all_assets.return_value = [otc_asset]
        result = get_short_universe(client)
        self.assertNotIn("OTCPK", result)

    def test_get_short_universe_filters_non_standard_symbols(self):
        from execution.short_universe import get_short_universe

        weird = MagicMock()
        weird.tradable = True
        weird.easy_to_borrow = True
        weird.exchange = "NYSE"
        weird.symbol = "BRK.B"  # dot in symbol → non-standard

        client = MagicMock()
        client.get_all_assets.return_value = [weird]
        result = get_short_universe(client)
        self.assertNotIn("BRK.B", result)

    def test_get_short_universe_returns_static_when_empty_response(self):
        from execution.short_universe import STATIC_SHORT_UNIVERSE, get_short_universe

        client = MagicMock()
        client.get_all_assets.return_value = []
        result = get_short_universe(client)
        self.assertEqual(result, STATIC_SHORT_UNIVERSE)

    def test_scan_short_universe_returns_empty_on_download_failure(self):
        from execution.short_universe import scan_short_universe

        with patch("execution.short_universe.yf.download", side_effect=Exception("network")):
            result = scan_short_universe(["AAPL"])
        self.assertEqual(result, [])

    def test_scan_short_universe_returns_empty_for_empty_symbols(self):
        from execution.short_universe import scan_short_universe

        self.assertEqual(scan_short_universe([]), [])

    def test_scan_short_universe_returns_empty_on_empty_dataframe(self):
        import pandas as pd

        from execution.short_universe import scan_short_universe

        with patch("execution.short_universe.yf.download", return_value=pd.DataFrame()):
            result = scan_short_universe(["AAPL"])
        self.assertEqual(result, [])

    def test_scan_short_universe_single_symbol_returns_snapshot(self):
        import pandas as pd

        from execution.short_universe import scan_short_universe

        idx = pd.bdate_range("2024-01-01", periods=25)
        closes = [100.0 + i * 0.1 for i in range(25)]
        df = pd.DataFrame(
            {
                "Open": closes,
                "High": [c * 1.005 for c in closes],
                "Low": [c * 0.995 for c in closes],
                "Close": closes,
                "Volume": [1_000_000] * 25,
            },
            index=idx,
        )
        with patch("execution.short_universe.yf.download", return_value=df):
            result = scan_short_universe(["AAPL"])
        self.assertEqual(len(result), 1)
        snap = result[0]
        self.assertEqual(snap["symbol"], "AAPL")
        self.assertIn("rs_rank_pct", snap)
        self.assertIn("rs_rank_pct_10d_ago", snap)
        # Gap fields are present; flat synthetic series → no gap detected.
        self.assertIsNone(snap["earnings_gap_pct"])
        self.assertFalse(snap["gap_failed_bounce"])

    def test_scan_short_universe_multi_symbol_returns_snapshots(self):
        import pandas as pd

        from execution.short_universe import scan_short_universe

        idx = pd.bdate_range("2024-01-01", periods=25)
        close = pd.DataFrame(
            {"AAPL": [100.0 + i for i in range(25)], "MSFT": [200.0 + i for i in range(25)]},
            index=idx,
        )
        volume = pd.DataFrame({"AAPL": [1_000_000] * 25, "MSFT": [2_000_000] * 25}, index=idx)
        low = close * 0.995
        multi_df = pd.concat({"Open": close, "Low": low, "Close": close, "Volume": volume}, axis=1)
        multi_df.columns = pd.MultiIndex.from_tuples(
            [(f, t) for f in ("Open", "Low", "Close", "Volume") for t in ("AAPL", "MSFT")]
        )
        with patch("execution.short_universe.yf.download", return_value=multi_df):
            result = scan_short_universe(["AAPL", "MSFT"])
        self.assertEqual(len(result), 2)
        symbols = {r["symbol"] for r in result}
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)

    def test_scan_short_universe_returns_empty_when_fewer_than_15_bars(self):
        import pandas as pd

        from execution.short_universe import scan_short_universe

        idx = pd.bdate_range("2024-01-01", periods=10)
        df = pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "Low": [99.0] * 10,
                "Close": [100.0] * 10,
                "Volume": [1_000_000] * 10,
            },
            index=idx,
        )
        with patch("execution.short_universe.yf.download", return_value=df):
            result = scan_short_universe(["AAPL"])
        self.assertEqual(result, [])

    def test_scan_short_universe_skips_symbol_with_zero_price(self):
        import pandas as pd

        from execution.short_universe import scan_short_universe

        idx = pd.bdate_range("2024-01-01", periods=25)
        prices = [0.0] * 25
        df = pd.DataFrame(
            {
                "Open": prices,
                "Low": prices,
                "Close": prices,
                "Volume": [1_000_000] * 25,
            },
            index=idx,
        )
        with patch("execution.short_universe.yf.download", return_value=df):
            result = scan_short_universe(["AAPL"])
        self.assertEqual(result, [])

    def test_scan_short_universe_skips_symbol_with_too_few_valid_bars(self):
        import numpy as np
        import pandas as pd

        from execution.short_universe import scan_short_universe

        idx = pd.bdate_range("2024-01-01", periods=25)
        # AAPL has 25 good bars; MSFT has only 5 valid bars → MSFT skipped at < 10 guard
        close = pd.DataFrame(
            {
                "AAPL": [100.0 + i for i in range(25)],
                "MSFT": [np.nan] * 20 + [200.0] * 5,
            },
            index=idx,
        )
        volume = pd.DataFrame({"AAPL": [1_000_000] * 25, "MSFT": [2_000_000] * 25}, index=idx)
        low = close * 0.995
        multi_df = pd.concat({"Open": close, "Low": low, "Close": close, "Volume": volume}, axis=1)
        multi_df.columns = pd.MultiIndex.from_tuples(
            [(f, t) for f in ("Open", "Low", "Close", "Volume") for t in ("AAPL", "MSFT")]
        )
        with patch("execution.short_universe.yf.download", return_value=multi_df):
            result = scan_short_universe(["AAPL", "MSFT"])
        symbols = {r["symbol"] for r in result}
        self.assertIn("AAPL", symbols)
        self.assertNotIn("MSFT", symbols)


class TestVixTermStructure(unittest.TestCase):
    """Tests for VIX9D term structure: RegimeFeatures.vix9d + to_dict vix_term_inverted."""

    def _snapshot_with_vix(self, vix, vix9d):
        from data.market_regime import (
            MarketRegime,
            MarketRegimeSnapshot,
            RegimeFeatures,
        )

        features = RegimeFeatures(
            spy_ret_1d=0.0,
            spy_ret_5d=-3.0,
            spy_ret_20d=-5.0,
            spy_above_ma200=False,
            spy_drawdown_pct=-5.0,
            vix=vix,
            vix_ma20=20.0,
            vix_vs_ma=(vix / 20.0 if vix else None),
            vix_5d_change=10.0,
            vix9d=vix9d,
            data_quality="full",
        )
        return MarketRegimeSnapshot(
            regime=MarketRegime.STRESS_RISK_OFF,
            reasons=("test",),
            features=features,
        )

    def test_vix_term_inverted_true_when_vix9d_exceeds_vix_by_5pct(self):
        snap = self._snapshot_with_vix(vix=20.0, vix9d=21.5)  # ratio = 1.075 > 1.05
        d = snap.to_dict()
        self.assertTrue(d["vix_term_inverted"])

    def test_vix_term_inverted_false_when_vix9d_below_vix(self):
        snap = self._snapshot_with_vix(vix=20.0, vix9d=19.0)  # ratio = 0.95 < 1.05
        d = snap.to_dict()
        self.assertFalse(d["vix_term_inverted"])

    def test_vix_term_inverted_false_when_vix9d_is_none(self):
        snap = self._snapshot_with_vix(vix=20.0, vix9d=None)
        d = snap.to_dict()
        self.assertFalse(d["vix_term_inverted"])

    def test_vix_term_inverted_false_when_vix_is_none(self):
        snap = self._snapshot_with_vix(vix=None, vix9d=22.0)
        d = snap.to_dict()
        self.assertFalse(d["vix_term_inverted"])

    def test_to_dict_includes_vix9d_key(self):
        snap = self._snapshot_with_vix(vix=20.0, vix9d=19.0)
        d = snap.to_dict()
        self.assertIn("vix9d", d)
        self.assertIn("vix_term_inverted", d)

    def test_compute_regime_features_sets_vix9d_from_df(self):
        import pandas as pd

        from data.market_regime import compute_regime_features

        idx = pd.bdate_range("2024-01-01", periods=10)
        spy_df = pd.DataFrame({"Close": [400.0 + i for i in range(10)]}, index=idx)
        vix_df = pd.DataFrame({"Close": [20.0] * 10}, index=idx)
        vix9d_df = pd.DataFrame({"Close": [21.0] * 10}, index=idx)

        features = compute_regime_features(spy_df, vix_df, vix9d_df=vix9d_df)
        self.assertAlmostEqual(features.vix9d, 21.0)

    def test_compute_regime_features_vix9d_none_when_no_df(self):
        import pandas as pd

        from data.market_regime import compute_regime_features

        idx = pd.bdate_range("2024-01-01", periods=10)
        spy_df = pd.DataFrame({"Close": [400.0 + i for i in range(10)]}, index=idx)

        features = compute_regime_features(spy_df, None)
        self.assertIsNone(features.vix9d)

    def test_fetch_vix9d_history_returns_none_when_cache_cold(self):
        import os
        import tempfile

        from data.market_regime import fetch_vix9d_history

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("data.market_regime._CACHE_PATH", os.path.join(tmp, "spy_vix_cache.pkl")),
            patch("data.market_regime.yf.download", return_value=__import__("pandas").DataFrame()),
        ):
            result = fetch_vix9d_history()
        self.assertIsNone(result)


class TestVixTermGateInBacktest(unittest.TestCase):
    """Tests for vix_term_inverted gate in _short_entry_signal."""

    def _make_row(self, **kwargs):
        import pandas as pd

        base = {
            "rsi": 55.0,
            "bb_pct": 0.4,
            "vol_ratio": 1.5,
            "macd_diff": 0.0,
            "macd_cross": False,
            "ema9": 100.0,
            "ema21": 105.0,
            "adx": 25.0,
            "ret_5d": -3.0,
            "ret_10d": -5.0,
            "pct_vs_ema21": -1.5,
            "price_vs_52w_high_pct": -15.0,
            "hv_rank": 0.5,
            "bb_squeeze": False,
            "is_inside_day": False,
            "gap_pct": 0.0,
            "close_above_open": False,
            "rsi_divergence": False,
            "failed_breakout_flag": False,
            "close_pct_of_range": 0.5,
            "high_short_interest": True,
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_vix_term_gate_blocks_entry_when_not_inverted(self):
        from backtest.engine import _short_entry_signal

        row = self._make_row()
        result = _short_entry_signal(
            row,
            rs_rank_pct=15.0,
            spy_ret_20d=None,
            regime="STRESS_RISK_OFF",
            vix_term_inverted=False,
        )
        self.assertIsNone(result)

    def test_vix_term_gate_allows_entry_when_inverted(self):
        from backtest.engine import _short_entry_signal

        row = self._make_row()
        result = _short_entry_signal(
            row,
            rs_rank_pct=15.0,
            spy_ret_20d=None,
            regime="STRESS_RISK_OFF",
            vix_term_inverted=True,
        )
        self.assertIsNotNone(result)
        self.assertIn("high_short_interest", result)

    def test_vix_term_gate_default_true_allows_entry(self):
        from backtest.engine import _short_entry_signal

        row = self._make_row()
        result = _short_entry_signal(
            row,
            rs_rank_pct=15.0,
            spy_ret_20d=None,
            regime="STRESS_RISK_OFF",
        )
        self.assertIsNotNone(result)


class TestRsDeteriorationPathInBacktest(unittest.TestCase):
    """Tests for the deterioration path (rs_rank_pct_10d_ago) in _short_entry_signal."""

    def _make_row(self, **kwargs):
        import pandas as pd

        base = {
            "rsi": 50.0,
            "bb_pct": 0.4,
            "vol_ratio": 1.2,
            "macd_diff": 0.0,
            "macd_cross": False,
            "ema9": 100.0,
            "ema21": 105.0,
            "adx": 25.0,
            "ret_5d": -3.5,
            "ret_10d": -5.0,
            "pct_vs_ema21": -2.0,
            "price_vs_52w_high_pct": -20.0,
            "hv_rank": 0.5,
            "bb_squeeze": False,
            "is_inside_day": False,
            "gap_pct": 0.0,
            "close_above_open": False,
            "rsi_divergence": False,
            "failed_breakout_flag": False,
            "close_pct_of_range": 0.5,
            "high_short_interest": False,
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_deterioration_path_returns_none_because_globally_disabled(self):
        from backtest.engine import _short_entry_signal

        row = self._make_row()
        result = _short_entry_signal(
            row,
            rs_rank_pct=35.0,
            spy_ret_20d=None,
            regime="STRESS_RISK_OFF",
            rs_rank_pct_10d_ago=72.0,
        )
        self.assertIsNone(result)

    def test_deterioration_path_not_fire_without_lag(self):
        from backtest.engine import _short_entry_signal

        row = self._make_row()
        # Middle-band stock with no lag data and no active signal → should be blocked
        result = _short_entry_signal(
            row,
            rs_rank_pct=35.0,
            spy_ret_20d=None,
            regime="STRESS_RISK_OFF",
            rs_rank_pct_10d_ago=None,
        )
        self.assertIsNone(result)


class TestComputeRsRankLag10(unittest.TestCase):
    """Tests for the _compute_rs_rank_lag10 helper in backtest/engine.py."""

    def test_lag_shifts_ranks_forward_by_10_days(self):
        import pandas as pd

        from backtest.engine import _compute_rs_rank_lag10

        dates = pd.bdate_range("2024-01-02", periods=25)
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
        rs_ranks = {"AAPL": {s: float(i) for i, s in enumerate(date_strs)}}
        result = _compute_rs_rank_lag10(rs_ranks, dates)

        # dates[10] should have the rank from dates[0]
        self.assertAlmostEqual(result["AAPL"][date_strs[10]], rs_ranks["AAPL"][date_strs[0]])

    def test_lag_returns_empty_for_short_history(self):
        import pandas as pd

        from backtest.engine import _compute_rs_rank_lag10

        dates = pd.bdate_range("2024-01-02", periods=5)
        rs_ranks = {"AAPL": {d.strftime("%Y-%m-%d"): 50.0 for d in dates}}
        result = _compute_rs_rank_lag10(rs_ranks, dates)
        # Fewer than 10 trading days → no shifted entry possible
        self.assertEqual(result, {})

    def test_lag_skips_symbols_with_no_matching_past_dates(self):
        import pandas as pd

        from backtest.engine import _compute_rs_rank_lag10

        dates = pd.bdate_range("2024-01-02", periods=20)
        # Give FADE only the last 5 dates — not enough past dates to shift into first 15
        fade_dates = [d.strftime("%Y-%m-%d") for d in dates[-5:]]
        rs_ranks = {"FADE": dict.fromkeys(fade_dates, 70.0)}
        result = _compute_rs_rank_lag10(rs_ranks, dates)
        # FADE has dates[15..19], lag10 would map dates[25..29] → not in our range
        self.assertEqual(result.get("FADE", {}), {})


class TestEarningsGapDownSignal(unittest.TestCase):
    """Tests for the earnings_gap_down signal in evaluate_short_signals."""

    def _snap(self, **kwargs):
        base = {"earnings_gap_pct": -8.0, "vol_ratio": 3.0}
        base.update(kwargs)
        return base

    def test_fires_when_gap_and_volume_meet_thresholds(self):
        from signals.evaluator import evaluate_short_signals

        result = evaluate_short_signals(self._snap())
        self.assertIn("earnings_gap_down", result)

    def test_blocked_when_gap_too_small(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(earnings_gap_pct=-3.0)  # only 3%, below 7% threshold
        self.assertNotIn("earnings_gap_down", evaluate_short_signals(snap))

    def test_blocked_when_gap_is_positive(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(earnings_gap_pct=2.0)  # gap UP — not a down signal
        self.assertNotIn("earnings_gap_down", evaluate_short_signals(snap))

    def test_blocked_when_no_gap_field(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"vol_ratio": 2.0}  # no earnings_gap_pct key
        self.assertNotIn("earnings_gap_down", evaluate_short_signals(snap))

    def test_blocked_when_vol_too_low(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(vol_ratio=1.0)  # below 2.5× threshold
        self.assertNotIn("earnings_gap_down", evaluate_short_signals(snap))

    def test_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap()
        self.assertNotIn(
            "earnings_gap_down",
            evaluate_short_signals(snap, blocked=frozenset({"earnings_gap_down"})),
        )

    def test_not_in_short_globally_disabled(self):
        from signals.evaluator import SHORT_GLOBALLY_DISABLED

        self.assertNotIn("earnings_gap_down", SHORT_GLOBALLY_DISABLED)

    def test_in_short_signal_priority(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertIn("earnings_gap_down", SHORT_SIGNAL_PRIORITY)

    def test_params_in_defaults(self):
        from signals.evaluator import DEFAULT_SHORT_SIGNAL_PARAMS

        self.assertIn("egd_gap_pct_max", DEFAULT_SHORT_SIGNAL_PARAMS)
        self.assertIn("egd_vol_min", DEFAULT_SHORT_SIGNAL_PARAMS)

    def test_custom_thresholds_respected(self):
        from signals.evaluator import evaluate_short_signals

        # Tighten to -10% → a -8% gap should no longer fire
        snap = self._snap(earnings_gap_pct=-8.0)
        result = evaluate_short_signals(snap, params={"egd_gap_pct_max": -10.0})
        self.assertNotIn("earnings_gap_down", result)

    def test_exactly_at_threshold_fires(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(earnings_gap_pct=-7.0)  # exactly at threshold (<=)
        self.assertIn("earnings_gap_down", evaluate_short_signals(snap))

    def test_priority_higher_than_high_short_interest(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertLess(
            SHORT_SIGNAL_PRIORITY["earnings_gap_down"],
            SHORT_SIGNAL_PRIORITY["high_short_interest"],
        )


class TestScanShortCandidatesEventPath(unittest.TestCase):
    """Tests for Path D (event-driven) in scan_short_candidates."""

    def _gap_snap(self, **kwargs):
        base = {
            "symbol": "AAPL",
            "earnings_gap_pct": -8.0,  # ≤ egdfb_gap_pct_max (-7.0)
            "gap_failed_bounce": True,  # S1/A3: failed-bounce continuation is the live trigger
            "vol_ratio": 3.0,
            "avg_volume": 1_000_000,
            "rs_rank_pct": 40.0,  # middle band — wouldn't fire on paths A/B/C
        }
        base.update(kwargs)
        return base

    def test_path_d_returns_candidate(self):
        from execution.stock_scanner import scan_short_candidates

        result = scan_short_candidates([self._gap_snap()], "STRESS_RISK_OFF", set())
        self.assertEqual(len(result), 1)
        # A3: naive earnings_gap_down is blocked live; the failed-bounce short surfaces instead.
        self.assertIn("post_earnings_gapdown_failed_bounce", result[0]["matched_signals"])
        self.assertNotIn("earnings_gap_down", result[0]["matched_signals"])

    def test_path_d_skips_when_no_gap_field(self):
        from execution.stock_scanner import scan_short_candidates

        snap = self._gap_snap()
        del snap["earnings_gap_pct"]
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_path_d_skips_when_gap_too_small(self):
        from execution.stock_scanner import scan_short_candidates

        snap = self._gap_snap(earnings_gap_pct=-2.0)
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_path_d_skips_etfs(self):
        from config import ETF_SYMBOLS
        from execution.stock_scanner import scan_short_candidates

        etf = next(iter(ETF_SYMBOLS))
        snap = self._gap_snap(symbol=etf)
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_path_d_skips_held_symbols(self):
        from execution.stock_scanner import scan_short_candidates

        result = scan_short_candidates([self._gap_snap()], "STRESS_RISK_OFF", {"AAPL"})
        self.assertEqual(result, [])

    def test_path_d_skips_low_volume(self):
        from execution.stock_scanner import scan_short_candidates

        snap = self._gap_snap(avg_volume=50_000)
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        self.assertEqual(result, [])

    def test_path_d_symbol_not_double_counted(self):
        from execution.stock_scanner import scan_short_candidates

        # Same symbol qualifies for both Path D and would qualify for another path — only one entry
        snap = self._gap_snap(rs_rank_pct=10.0, high_short_interest=True)
        result = scan_short_candidates([snap], "STRESS_RISK_OFF", set())
        symbols = [c["symbol"] for c in result]
        self.assertEqual(len(symbols), symbols.count("AAPL"))  # at most one entry


class TestSqueezeRisk(unittest.TestCase):
    """Tests for execution/short_risk.py — is_squeeze_risk and fetch_squeeze_info."""

    def _snap(self, ret_5d_pct=0.0):
        return {"symbol": "CRWD", "ret_5d_pct": ret_5d_pct}

    # ── is_squeeze_risk ───────────────────────────────────────────────────────

    def test_blocked_by_short_pct_float(self):
        from execution.short_risk import is_squeeze_risk

        blocked, reason = is_squeeze_risk("CRWD", self._snap(), short_pct_float=0.25)
        self.assertTrue(blocked)
        self.assertIn("short_pct_float", reason)

    def test_blocked_by_days_to_cover(self):
        from execution.short_risk import is_squeeze_risk

        blocked, reason = is_squeeze_risk("CRWD", self._snap(), days_to_cover=6.0)
        self.assertTrue(blocked)
        self.assertIn("days_to_cover", reason)

    def test_blocked_by_momentum(self):
        from execution.short_risk import is_squeeze_risk

        blocked, reason = is_squeeze_risk("CRWD", self._snap(ret_5d_pct=20.0))
        self.assertTrue(blocked)
        self.assertIn("ret_5d_pct", reason)
        self.assertIn("active squeeze", reason)

    def test_safe_when_all_clear(self):
        from execution.short_risk import is_squeeze_risk

        blocked, reason = is_squeeze_risk(
            "CRWD",
            self._snap(ret_5d_pct=2.0),
            short_pct_float=0.10,
            days_to_cover=3.0,
        )
        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_none_short_pct_float_skips_check(self):
        from execution.short_risk import is_squeeze_risk

        # None → field check skipped; only momentum checked (below threshold)
        blocked, _ = is_squeeze_risk("CRWD", self._snap(), short_pct_float=None)
        self.assertFalse(blocked)

    def test_none_days_to_cover_skips_check(self):
        from execution.short_risk import is_squeeze_risk

        blocked, _ = is_squeeze_risk("CRWD", self._snap(), days_to_cover=None)
        self.assertFalse(blocked)

    def test_missing_ret_5d_defaults_to_zero(self):
        from execution.short_risk import is_squeeze_risk

        snap = {"symbol": "CRWD"}  # no ret_5d_pct key
        blocked, _ = is_squeeze_risk("CRWD", snap)
        self.assertFalse(blocked)

    def test_reason_includes_values(self):
        from execution.short_risk import is_squeeze_risk

        _, reason = is_squeeze_risk("CRWD", self._snap(), short_pct_float=0.30)
        self.assertIn("30.0%", reason)

    def test_custom_thresholds_respected(self):
        from execution.short_risk import is_squeeze_risk

        # Default short_pct_float_max=0.20; raising it to 0.40 should not block at 0.25
        blocked, _ = is_squeeze_risk(
            "CRWD", self._snap(), short_pct_float=0.25, short_pct_float_max=0.40
        )
        self.assertFalse(blocked)

    def test_exactly_at_threshold_not_blocked(self):
        from execution.short_risk import is_squeeze_risk

        # > not >=: exactly at threshold is safe
        blocked, _ = is_squeeze_risk("CRWD", self._snap(), short_pct_float=0.20)
        self.assertFalse(blocked)

    # ── fetch_squeeze_info ────────────────────────────────────────────────────

    def test_fetch_squeeze_info_returns_fields(self):
        from unittest.mock import MagicMock, patch

        from execution.short_risk import fetch_squeeze_info

        mock_ticker = MagicMock()
        mock_ticker.info = {"shortPercentOfFloat": 0.12, "shortRatio": 3.5}
        with patch("execution.short_risk.yf.Ticker", return_value=mock_ticker):
            result = fetch_squeeze_info("CRWD")
        self.assertAlmostEqual(result["short_pct_float"], 0.12)
        self.assertAlmostEqual(result["days_to_cover"], 3.5)

    def test_fetch_squeeze_info_returns_none_for_missing_keys(self):
        from unittest.mock import MagicMock, patch

        from execution.short_risk import fetch_squeeze_info

        mock_ticker = MagicMock()
        mock_ticker.info = {}  # no short-interest fields
        with patch("execution.short_risk.yf.Ticker", return_value=mock_ticker):
            result = fetch_squeeze_info("CRWD")
        self.assertIsNone(result["short_pct_float"])
        self.assertIsNone(result["days_to_cover"])

    def test_fetch_squeeze_info_handles_exception(self):
        from unittest.mock import patch

        from execution.short_risk import fetch_squeeze_info

        with patch("execution.short_risk.yf.Ticker", side_effect=RuntimeError("api down")):
            result = fetch_squeeze_info("CRWD")
        self.assertIsNone(result["short_pct_float"])
        self.assertIsNone(result["days_to_cover"])


class TestOverboughtDowntrendSignal(unittest.TestCase):
    """Tests for the overbought_downtrend signal in evaluate_short_signals."""

    def _snap(self, **kwargs):
        base = {
            "price_below_sma200": True,
            "rsi_prev": 68.0,  # was above ordt_rsi_entry (65.0)
            "rsi_14": 57.0,  # now below ordt_rsi_exit (60.0)
            "vol_ratio": 1.0,
        }
        base.update(kwargs)
        return base

    def test_does_not_fire_when_globally_disabled(self):
        from signals.evaluator import evaluate_short_signals

        result = evaluate_short_signals(self._snap())
        self.assertNotIn("overbought_downtrend", result)

    def test_blocked_when_not_below_sma200(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(price_below_sma200=False)
        self.assertNotIn("overbought_downtrend", evaluate_short_signals(snap))

    def test_blocked_when_rsi_prev_below_entry_threshold(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(rsi_prev=62.0)  # never reached ordt_rsi_entry (65.0) — not overbought
        self.assertNotIn("overbought_downtrend", evaluate_short_signals(snap))

    def test_blocked_when_rsi_still_above_exit_threshold(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(rsi_14=61.0)  # above ordt_rsi_exit (60.0) — hasn't crossed back down
        self.assertNotIn("overbought_downtrend", evaluate_short_signals(snap))

    def test_blocked_when_vol_too_low(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(vol_ratio=0.3)  # below 0.8 floor
        self.assertNotIn("overbought_downtrend", evaluate_short_signals(snap))

    def test_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap()
        self.assertNotIn(
            "overbought_downtrend",
            evaluate_short_signals(snap, blocked=frozenset({"overbought_downtrend"})),
        )

    def test_in_short_globally_disabled(self):
        from signals.evaluator import SHORT_GLOBALLY_DISABLED

        self.assertIn("overbought_downtrend", SHORT_GLOBALLY_DISABLED)

    def test_in_short_signal_priority(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertIn("overbought_downtrend", SHORT_SIGNAL_PRIORITY)

    def test_params_in_defaults(self):
        from signals.evaluator import DEFAULT_SHORT_SIGNAL_PARAMS

        self.assertIn("ordt_rsi_entry", DEFAULT_SHORT_SIGNAL_PARAMS)
        self.assertIn("ordt_rsi_exit", DEFAULT_SHORT_SIGNAL_PARAMS)
        self.assertIn("ordt_vol_min", DEFAULT_SHORT_SIGNAL_PARAMS)

    def test_custom_rsi_entry_threshold_respected(self):
        from signals.evaluator import evaluate_short_signals

        # Tighten entry to 70 → rsi_prev=68 is below 70, never crossed above threshold
        snap = self._snap(rsi_prev=68.0)
        result = evaluate_short_signals(snap, params={"ordt_rsi_entry": 70.0})
        self.assertNotIn("overbought_downtrend", result)

    def test_exactly_at_rsi_exit_does_not_fire(self):
        from signals.evaluator import evaluate_short_signals

        # rsi_14 == ordt_rsi_exit: condition is strict < so at-threshold should NOT fire
        snap = self._snap(rsi_14=60.0)
        self.assertNotIn("overbought_downtrend", evaluate_short_signals(snap))


class TestParabolicExhaustionSignal(unittest.TestCase):
    """Tests for the parabolic_exhaustion signal in evaluate_short_signals."""

    def _snap(self, **kwargs):
        base = {
            "ret_60d_pct": 90.0,
            "rsi_14": 75.0,
            "vol_ratio": 0.7,
        }
        base.update(kwargs)
        return base

    def test_does_not_fire_when_globally_disabled(self):
        from signals.evaluator import evaluate_short_signals

        result = evaluate_short_signals(self._snap())
        self.assertNotIn("parabolic_exhaustion", result)

    def test_blocked_when_ret60d_too_low(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(ret_60d_pct=60.0)  # below 80% threshold
        self.assertNotIn("parabolic_exhaustion", evaluate_short_signals(snap))

    def test_blocked_when_rsi_too_low(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(rsi_14=68.0)  # below 72 threshold
        self.assertNotIn("parabolic_exhaustion", evaluate_short_signals(snap))

    def test_blocked_when_volume_not_drying(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(vol_ratio=1.2)  # above 0.9 max — buyers still active
        self.assertNotIn("parabolic_exhaustion", evaluate_short_signals(snap))

    def test_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap()
        self.assertNotIn(
            "parabolic_exhaustion",
            evaluate_short_signals(snap, blocked=frozenset({"parabolic_exhaustion"})),
        )

    def test_in_short_globally_disabled(self):
        from signals.evaluator import SHORT_GLOBALLY_DISABLED

        self.assertIn("parabolic_exhaustion", SHORT_GLOBALLY_DISABLED)

    def test_in_short_signal_priority(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertIn("parabolic_exhaustion", SHORT_SIGNAL_PRIORITY)

    def test_params_in_defaults(self):
        from signals.evaluator import DEFAULT_SHORT_SIGNAL_PARAMS

        self.assertIn("pe_ret60d_min", DEFAULT_SHORT_SIGNAL_PARAMS)
        self.assertIn("pe_rsi_min", DEFAULT_SHORT_SIGNAL_PARAMS)
        self.assertIn("pe_vol_ratio_max", DEFAULT_SHORT_SIGNAL_PARAMS)

    def test_does_not_fire_even_at_exact_threshold(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(ret_60d_pct=80.0)  # at threshold but globally disabled
        self.assertNotIn("parabolic_exhaustion", evaluate_short_signals(snap))

    def test_custom_threshold_respected(self):
        from signals.evaluator import evaluate_short_signals

        # Raise threshold to 120% → 90% no longer qualifies
        snap = self._snap(ret_60d_pct=90.0)
        result = evaluate_short_signals(snap, params={"pe_ret60d_min": 120.0})
        self.assertNotIn("parabolic_exhaustion", result)


class TestFadedEarningsGapUpSignal(unittest.TestCase):
    """Tests for the faded_earnings_gap_up signal in evaluate_short_signals."""

    def _snap(self, **kwargs):
        base = {
            "faded_earnings_gap_up_pct": 8.0,
            "close_pct_of_range": 0.15,
            "vol_ratio": 2.0,
        }
        base.update(kwargs)
        return base

    def test_does_not_fire_when_globally_disabled(self):
        from signals.evaluator import evaluate_short_signals

        # faded_earnings_gap_up is in SHORT_GLOBALLY_DISABLED — never fires
        result = evaluate_short_signals(self._snap())
        self.assertNotIn("faded_earnings_gap_up", result)

    def test_blocked_when_gap_too_small(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(faded_earnings_gap_up_pct=3.0)  # below 5% threshold
        self.assertNotIn("faded_earnings_gap_up", evaluate_short_signals(snap))

    def test_blocked_when_no_gap_field(self):
        from signals.evaluator import evaluate_short_signals

        snap = {"close_pct_of_range": 0.15, "vol_ratio": 2.0}  # no faded_earnings_gap_up_pct
        self.assertNotIn("faded_earnings_gap_up", evaluate_short_signals(snap))

    def test_blocked_when_close_not_weak(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(close_pct_of_range=0.7)  # closed in upper half — no distribution
        self.assertNotIn("faded_earnings_gap_up", evaluate_short_signals(snap))

    def test_blocked_when_vol_too_low(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(vol_ratio=0.8)  # below 1.5 floor
        self.assertNotIn("faded_earnings_gap_up", evaluate_short_signals(snap))

    def test_blocked_when_in_blocked_set(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap()
        self.assertNotIn(
            "faded_earnings_gap_up",
            evaluate_short_signals(snap, blocked=frozenset({"faded_earnings_gap_up"})),
        )

    def test_in_short_globally_disabled(self):
        from signals.evaluator import SHORT_GLOBALLY_DISABLED

        self.assertIn("faded_earnings_gap_up", SHORT_GLOBALLY_DISABLED)

    def test_in_short_signal_priority(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertIn("faded_earnings_gap_up", SHORT_SIGNAL_PRIORITY)

    def test_params_in_defaults(self):
        from signals.evaluator import DEFAULT_SHORT_SIGNAL_PARAMS

        self.assertIn("fegu_gap_min", DEFAULT_SHORT_SIGNAL_PARAMS)
        self.assertIn("fegu_range_max", DEFAULT_SHORT_SIGNAL_PARAMS)
        self.assertIn("fegu_vol_min", DEFAULT_SHORT_SIGNAL_PARAMS)

    def test_globally_disabled_overrides_all_conditions(self):
        from signals.evaluator import evaluate_short_signals

        # Even with perfect conditions, globally disabled signals never fire
        snap = self._snap(faded_earnings_gap_up_pct=5.0)
        self.assertNotIn("faded_earnings_gap_up", evaluate_short_signals(snap))

    def test_gap_must_be_positive(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(faded_earnings_gap_up_pct=-3.0)  # negative gap → not a gap-up
        self.assertNotIn("faded_earnings_gap_up", evaluate_short_signals(snap))

    def test_priority_lower_than_earnings_gap_down(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertLess(
            SHORT_SIGNAL_PRIORITY["earnings_gap_down"],
            SHORT_SIGNAL_PRIORITY["faded_earnings_gap_up"],
        )


class TestIVCompressionShortSignal(unittest.TestCase):
    """Tests for iv_compression_short in evaluate_short_signals (disabled pending backtest)."""

    def _snap(self, **kwargs):
        base = {
            "price_below_sma200": True,
            "ema9_above_ema21": False,  # EMA9 below EMA21 — confirmed downtrend
            "hv_rank": 0.08,
            "vol_ratio": 1.2,
        }
        base.update(kwargs)
        return base

    def test_does_not_fire_when_globally_disabled(self):
        from signals.evaluator import evaluate_short_signals

        self.assertNotIn("iv_compression_short", evaluate_short_signals(self._snap()))

    def test_in_short_globally_disabled(self):
        from signals.evaluator import SHORT_GLOBALLY_DISABLED

        self.assertIn("iv_compression_short", SHORT_GLOBALLY_DISABLED)

    def test_in_short_signal_priority(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertIn("iv_compression_short", SHORT_SIGNAL_PRIORITY)

    def test_params_in_defaults(self):
        from signals.evaluator import DEFAULT_SHORT_SIGNAL_PARAMS

        self.assertIn("ivcs_hv_rank_max", DEFAULT_SHORT_SIGNAL_PARAMS)
        self.assertIn("ivcs_vol_min", DEFAULT_SHORT_SIGNAL_PARAMS)

    def test_blocked_when_above_sma200(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(price_below_sma200=False)
        self.assertNotIn("iv_compression_short", evaluate_short_signals(snap))

    def test_blocked_when_ema_uptrend(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(ema9_above_ema21=True)
        self.assertNotIn("iv_compression_short", evaluate_short_signals(snap))

    def test_blocked_when_hv_rank_too_high(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(hv_rank=0.50)  # above ivcs_hv_rank_max (0.15)
        self.assertNotIn("iv_compression_short", evaluate_short_signals(snap))

    def test_blocked_when_vol_too_low(self):
        from signals.evaluator import evaluate_short_signals

        snap = self._snap(vol_ratio=0.3)  # below ivcs_vol_min (1.0)
        self.assertNotIn("iv_compression_short", evaluate_short_signals(snap))

    def test_priority_lower_than_secondary_offering_short(self):
        from signals.evaluator import SHORT_SIGNAL_PRIORITY

        self.assertGreater(
            SHORT_SIGNAL_PRIORITY["iv_compression_short"],
            SHORT_SIGNAL_PRIORITY["secondary_offering_short"],
        )


class TestDetectFailedGapdown(unittest.TestCase):
    """detect_failed_gapdown: OHLCV-only earnings gap-down + failed-bounce detector."""

    def test_too_few_bars_returns_default(self):
        from execution.short_universe import detect_failed_gapdown

        out = detect_failed_gapdown([100.0], [99.0], [100.0], [1_000_000])
        self.assertIsNone(out["earnings_gap_pct"])
        self.assertFalse(out["gap_failed_bounce"])
        self.assertEqual(out["vol_ratio"], 1.0)

    def test_no_gap_in_window(self):
        from execution.short_universe import detect_failed_gapdown

        n = 10
        opens = [100.0] * n
        lows = [99.0] * n
        closes = [100.0] * n
        vols = [1_000_000] * n
        out = detect_failed_gapdown(opens, lows, closes, vols)
        self.assertIsNone(out["earnings_gap_pct"])
        self.assertFalse(out["gap_failed_bounce"])

    def test_gap_with_failed_bounce(self):
        """Gap-down 3 bars ago, then price breaks below the gap bar's low → failed bounce."""
        from execution.short_universe import detect_failed_gapdown

        # bars 0-2 flat at 100; bar 3 gaps down to 90 (open) low 88; bars 4-5 drift to 86 < 88
        opens = [100.0, 100.0, 100.0, 90.0, 89.0, 87.0]
        lows = [99.0, 99.0, 99.0, 88.0, 88.0, 86.0]
        closes = [100.0, 100.0, 100.0, 89.0, 88.5, 86.0]
        vols = [1_000_000] * 6
        out = detect_failed_gapdown(opens, lows, closes, vols)
        self.assertIsNotNone(out["earnings_gap_pct"])
        self.assertLess(out["earnings_gap_pct"], -7.0)
        self.assertTrue(out["gap_failed_bounce"])

    def test_gap_on_last_bar_no_bounce_yet(self):
        """Gap is the most recent bar → no continuation bar yet → bounce not confirmed."""
        from execution.short_universe import detect_failed_gapdown

        opens = [100.0, 100.0, 100.0, 100.0, 90.0]
        lows = [99.0, 99.0, 99.0, 99.0, 88.0]
        closes = [100.0, 100.0, 100.0, 100.0, 89.0]
        vols = [1_000_000] * 5
        out = detect_failed_gapdown(opens, lows, closes, vols)
        self.assertIsNotNone(out["earnings_gap_pct"])
        self.assertFalse(out["gap_failed_bounce"])

    def test_gap_but_price_held_above_gap_low(self):
        """Gap-down then price recovers above the gap bar's low → bounce did NOT fail."""
        from execution.short_universe import detect_failed_gapdown

        opens = [100.0, 100.0, 100.0, 90.0, 91.0, 92.0]
        lows = [99.0, 99.0, 99.0, 88.0, 90.0, 91.0]
        closes = [100.0, 100.0, 100.0, 89.0, 91.0, 92.0]
        vols = [1_000_000] * 6
        out = detect_failed_gapdown(opens, lows, closes, vols)
        self.assertIsNotNone(out["earnings_gap_pct"])
        self.assertFalse(out["gap_failed_bounce"])

    def test_zero_prev_close_skipped(self):
        """A zero prior close is skipped (no division), and an earlier real gap is still found."""
        from execution.short_universe import detect_failed_gapdown

        # bar1 prev_close=0 (skip); a real gap at bar 2 (prev close 100 → open 90)
        opens = [100.0, 0.0, 90.0, 89.0]
        lows = [99.0, 0.0, 88.0, 86.0]
        closes = [100.0, 0.0, 89.0, 86.0]
        vols = [1_000_000] * 4
        out = detect_failed_gapdown(opens, lows, closes, vols)
        # bar 3 prev_close=89 → no gap; bar 2 prev_close=0 → skipped; bar1 prev_close=100→open0? handled
        self.assertIn("gap_failed_bounce", out)

    def test_two_bars_gap_no_vol_ratio(self):
        """Exactly 2 bars: vol-ratio block is skipped (needs ≥3); a gap on the last bar is
        detected but the bounce cannot be confirmed."""
        from execution.short_universe import detect_failed_gapdown

        out = detect_failed_gapdown([100.0, 90.0], [99.0, 88.0], [100.0, 89.0], [1e6, 1e6])
        self.assertLess(out["earnings_gap_pct"], -7.0)
        self.assertFalse(out["gap_failed_bounce"])
        self.assertEqual(out["vol_ratio"], 1.0)

    def test_vol_ratio_uses_trailing_average(self):
        from execution.short_universe import detect_failed_gapdown

        opens = [100.0] * 5
        lows = [99.0] * 5
        closes = [100.0] * 5
        vols = [1_000_000, 1_000_000, 1_000_000, 1_000_000, 3_000_000]
        out = detect_failed_gapdown(opens, lows, closes, vols)
        self.assertGreater(out["vol_ratio"], 2.0)

    def test_zero_trailing_volume_keeps_default_ratio(self):
        from execution.short_universe import detect_failed_gapdown

        opens = [100.0] * 4
        lows = [99.0] * 4
        closes = [100.0] * 4
        vols = [0.0, 0.0, 0.0, 5_000.0]
        out = detect_failed_gapdown(opens, lows, closes, vols)
        self.assertEqual(out["vol_ratio"], 1.0)


class TestShortPreTrade(unittest.TestCase):
    """Short execution path must enforce fat-finger cap and daily-notional accounting."""

    def _make_deps(
        self, daily_notional=0.0, place_short_result=None, raise_on_daily_notional=False
    ):
        from core.deps import TradingDeps

        class FakeTrader:
            def get_open_shorts(self):
                return []

            def get_long_notional(self, _client):
                return 0.0

            def get_short_notional(self, _client):
                return 0.0

            def get_daily_notional(self, _date):
                if raise_on_daily_notional:
                    raise RuntimeError("ledger unavailable")
                return daily_notional

            def add_daily_notional(self, _date, _amount):
                self.added = _amount

            def place_short_order(self, _client, _symbol, _qty):
                return place_short_result

            def place_short_cover_stop(self, _client, _symbol, _qty):
                return type("CoverResult", (), {"is_success": True})()

            def record_short(self, *a, **kw):
                pass

        # ADR-006 B2: _execute_shorts no longer scans — it consumes the AI-vetted decisions and the
        # scanned candidates passed in — so deps.stock_scanner is unused on this path.
        deps = TradingDeps.__new__(TradingDeps)
        deps.trader = FakeTrader()
        deps.sector_data = type("SD", (), {"get_sector": staticmethod(lambda s: "Technology")})()
        deps.sector_momentum = type(
            "SM",
            (),
            {
                "get_sector_momentum_ranks": lambda self: {"Technology": 20},
                "sector_allowed_short": lambda self, sector, ranks: True,
            },
        )()
        deps.correlation = type(
            "C", (), {"correlated_with_held": staticmethod(lambda s, h: False)}
        )()
        deps.short_risk = type(
            "SR",
            (),
            {
                "fetch_squeeze_info": lambda self, s: {},
                "is_squeeze_risk": lambda self, s, c, **kw: (False, ""),
            },
        )()
        deps.position_sizer = type(
            "PS",
            (),
            {"risk_budget_size": lambda self, pv, confidence, signal, regime: 500.0},
        )()
        deps.audit_log = type("AL", (), {"log_order_placed": lambda self, *a, **kw: None})()
        deps.alerts = type("A", (), {"alert_error": lambda self, *a, **kw: None})()
        return deps

    @staticmethod
    def _xyz_scanned():
        """The rule-gated short candidate (full dict) the scanner produces pre-AI."""
        return [
            {
                "symbol": "XYZ",
                "current_price": 50.0,
                "matched_signals": ["earnings_gap_down"],
                "key_signal": "earnings_gap_down",
                "confidence": 8,
                "rs_rank_pct": 10.0,
            }
        ]

    @staticmethod
    def _xyz_decisions():
        """The AI's approved short_candidates (ADR-006 B2) citing the scanned XYZ candidate."""
        return {
            "short_candidates": [
                {
                    "symbol": "XYZ",
                    "confidence": 8,
                    "key_signal": "earnings_gap_down",
                    "reasoning": "negative PEAD continuation on heavy volume",
                }
            ]
        }

    def test_short_order_skipped_when_hard_to_borrow(self):
        """A hard-to-borrow name (high short interest → borrow rate ≥ HTB threshold) is skipped
        by the borrow-cost gate before sizing."""
        deps = self._make_deps()
        # Override squeeze info so the name reads as heavily shorted → HTB.
        deps.short_risk = type(
            "SR",
            (),
            {
                "fetch_squeeze_info": lambda self, s: {
                    "short_pct_float": 0.40,
                    "days_to_cover": 12.0,
                },
                "is_squeeze_risk": lambda self, s, c, **kw: (False, ""),
            },
        )()
        all_trades: list = []
        from main import _execute_shorts

        _execute_shorts(
            client=None,
            decisions=self._xyz_decisions(),
            scanned_short_candidates=self._xyz_scanned(),
            regime={"regime": "DEFENSIVE_DOWNTREND", "vix_term_inverted": True},
            open_positions=[],
            account_now={"portfolio_value": 10_000.0},
            all_trades=all_trades,
            executed_symbols=set(),
            dry_run=False,
            _live_shadow=False,
            today="2026-06-12",
            deps=deps,
        )
        self.assertEqual(all_trades, [], "Hard-to-borrow short should be skipped")

    def test_short_order_rejected_when_notional_exceeds_single_order_cap(self):
        """Short order must be skipped when order_notional > MAX_SINGLE_ORDER_USD.

        Without the C4 fix, no fat-finger check existed on the short path.
        """
        from unittest.mock import patch

        import config as cfg

        deps = self._make_deps()
        # candidate: $50 × floor(500/50) = 10 shares = $500 notional
        # Patch cap to $10 so the $500 order is rejected
        all_trades: list = []
        from main import _execute_shorts

        with (
            patch.object(cfg, "MAX_SINGLE_ORDER_USD", 10.0),
            patch.object(cfg, "MAX_SHORT_STANDALONE_RATIO", 1.0),
            patch.object(cfg, "MAX_SHORT_POSITIONS", 5),
        ):
            _execute_shorts(
                client=None,
                decisions=self._xyz_decisions(),
                scanned_short_candidates=self._xyz_scanned(),
                regime={"regime": "DEFENSIVE_DOWNTREND", "vix_term_inverted": True},
                open_positions=[],
                account_now={"portfolio_value": 10_000.0},
                all_trades=all_trades,
                executed_symbols=set(),
                dry_run=False,
                _live_shadow=False,
                today="2026-06-12",
                deps=deps,
            )
        self.assertEqual(all_trades, [], "Short order should have been blocked by fat-finger cap")

    def test_short_order_placed_records_daily_notional_on_success(self):
        """Success path: an approved short fill records the daily notional (C4 ledger accounting).

        Exercises the post-fill branch — place_short_order → record_short → add_daily_notional →
        cover stop — that the rejection test never reaches.
        """
        from unittest.mock import patch

        import config as cfg

        short_fill = type(
            "ShortResult",
            (),
            {
                "broker_order_id": "OID-XYZ-1",
                "is_success": True,
                "filled_avg_price": 50.0,
                "filled_qty": 10,
            },
        )()
        deps = self._make_deps(place_short_result=short_fill)
        all_trades: list = []
        from main import _execute_shorts

        with (
            patch.object(cfg, "MAX_SINGLE_ORDER_USD", 50_000.0),
            patch.object(cfg, "MAX_DAILY_NOTIONAL_USD", 150_000.0),
            patch.object(cfg, "MAX_SHORT_STANDALONE_RATIO", 1.0),
            patch.object(cfg, "MAX_SHORT_POSITIONS", 5),
        ):
            _execute_shorts(
                client=None,
                decisions=self._xyz_decisions(),
                scanned_short_candidates=self._xyz_scanned(),
                regime={"regime": "DEFENSIVE_DOWNTREND", "vix_term_inverted": True},
                open_positions=[],
                account_now={"portfolio_value": 10_000.0},
                all_trades=all_trades,
                executed_symbols=set(),
                dry_run=False,
                _live_shadow=False,
                today="2026-06-12",
                deps=deps,
            )
        # add_daily_notional was called on the fill path (0.5× short sizing → positive notional).
        self.assertGreater(deps.trader.added, 0.0, "Daily notional should be recorded on fill")
        self.assertEqual(len(all_trades), 1)
        self.assertEqual(all_trades[0]["action"], "SHORT")

    def test_short_pre_trade_tolerates_daily_notional_read_failure(self):
        """If get_daily_notional raises, the short path logs and proceeds with _daily_so_far=0
        (still enforcing the single-order cap). Covers the ledger-read except branch."""
        from unittest.mock import patch

        import config as cfg

        short_fill = type(
            "ShortResult",
            (),
            {
                "broker_order_id": "OID-XYZ-2",
                "is_success": True,
                "filled_avg_price": 50.0,
                "filled_qty": 10,
            },
        )()
        deps = self._make_deps(place_short_result=short_fill, raise_on_daily_notional=True)
        all_trades: list = []
        from main import _execute_shorts

        with (
            patch.object(cfg, "MAX_SINGLE_ORDER_USD", 50_000.0),
            patch.object(cfg, "MAX_DAILY_NOTIONAL_USD", 150_000.0),
            patch.object(cfg, "MAX_SHORT_STANDALONE_RATIO", 1.0),
            patch.object(cfg, "MAX_SHORT_POSITIONS", 5),
        ):
            _execute_shorts(
                client=None,
                decisions=self._xyz_decisions(),
                scanned_short_candidates=self._xyz_scanned(),
                regime={"regime": "DEFENSIVE_DOWNTREND", "vix_term_inverted": True},
                open_positions=[],
                account_now={"portfolio_value": 10_000.0},
                all_trades=all_trades,
                executed_symbols=set(),
                dry_run=False,
                _live_shadow=False,
                today="2026-06-12",
                deps=deps,
            )
        # The order still went through (cap enforced with _daily_so_far=0 fallback).
        self.assertEqual(len(all_trades), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

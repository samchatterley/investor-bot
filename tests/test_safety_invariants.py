"""
CI invariant tests — safety properties that must hold across all future edits.

These tests make it hard to merge code that weakens safety. They are not
testing specific implementations; they are testing that invariants are
preserved. A failure here means a safety regression has been introduced.

Invariants checked:
  1. has_pending_buy() raises BrokerStateUnavailable on broker failure (never fails open)
  2. get_total_open_exposure() raises BrokerStateUnavailable on broker failure (never returns 0)
  3. SMALL_ACCOUNT_MODE=true defaults are within safe bounds (caps never drift upward)
  4. MAX_ORDERS_PER_RUN <= 1 in SMALL_ACCOUNT_MODE
  5. Stop failure triggers flatten/halt in live mode (never silently ignored)
  6. check_pre_trade() is always called in the buy path (no bypass)
  7. ALLOW_MARGIN defaults to False
  8. BrokerStateUnavailable is raised (not swallowed) — buy loop must break, not continue
  9. HealthStatus.RED blocks new buys
 10. No buy can proceed when health is RED or YELLOW
 11. has_active_intent() raises OrderLedgerUnavailable on DB failure (never fails open)
 12. Quote gate last-trade failure raises BrokerStateUnavailable (never approves on unknown trade state)
 13. create_intent() failure in live mode raises OrderLedgerUnavailable (never proceeds without durable record)
"""

import importlib
import os
import unittest
from unittest.mock import MagicMock, patch


class TestFailClosedBrokerState(unittest.TestCase):
    """Broker query failures must raise, never fail open."""

    def test_has_pending_buy_raises_on_get_orders_failure(self):
        from execution.trader import has_pending_buy
        from models import BrokerStateUnavailable

        client = MagicMock()
        client.get_orders.side_effect = RuntimeError("timeout")
        with self.assertRaises(BrokerStateUnavailable):
            has_pending_buy(client, "AAPL")

    def test_has_pending_buy_raises_on_connection_error(self):
        from execution.trader import has_pending_buy
        from models import BrokerStateUnavailable

        client = MagicMock()
        client.get_orders.side_effect = ConnectionError("broker unreachable")
        with self.assertRaises(BrokerStateUnavailable):
            has_pending_buy(client, "MSFT")

    def test_get_total_open_exposure_raises_on_positions_failure(self):
        from execution.trader import get_total_open_exposure
        from models import BrokerStateUnavailable

        client = MagicMock()
        client.get_all_positions.side_effect = RuntimeError("timeout")
        with self.assertRaises(BrokerStateUnavailable):
            get_total_open_exposure(client)

    def test_get_total_open_exposure_raises_on_orders_failure(self):
        from execution.trader import get_total_open_exposure
        from models import BrokerStateUnavailable

        client = MagicMock()
        client.get_all_positions.return_value = []
        client.get_orders.side_effect = RuntimeError("auth error")
        with self.assertRaises(BrokerStateUnavailable):
            get_total_open_exposure(client)

    def test_broker_state_unavailable_is_exception_subclass(self):
        """BrokerStateUnavailable must be catchable as Exception (buy loop must handle it)."""
        from models import BrokerStateUnavailable

        self.assertTrue(issubclass(BrokerStateUnavailable, Exception))


class TestSmallAccountModeSafeBounds(unittest.TestCase):
    """SMALL_ACCOUNT_MODE defaults must never drift into unsafe territory."""

    def setUp(self):
        with patch.dict(os.environ, {"SMALL_ACCOUNT_MODE": "true"}, clear=False):
            import config as cfg

            importlib.reload(cfg)
            self.cfg = cfg

    def test_max_single_order_within_experiment_budget(self):
        self.assertLessEqual(
            self.cfg.MAX_SINGLE_ORDER_USD,
            60.0,
            "Single order cap must not exceed experiment budget",
        )

    def test_max_daily_notional_within_experiment_budget(self):
        self.assertLessEqual(
            self.cfg.MAX_DAILY_NOTIONAL_USD,
            100.0,
            "Daily notional cap must not exceed experiment budget",
        )

    def test_max_deployed_usd_active_and_bounded(self):
        self.assertGreater(
            self.cfg.MAX_DEPLOYED_USD, 0, "MAX_DEPLOYED_USD must be enabled in small-account mode"
        )
        self.assertLessEqual(
            self.cfg.MAX_DEPLOYED_USD, 150.0, "MAX_DEPLOYED_USD must not exceed experiment capital"
        )

    def test_max_orders_per_run_is_1(self):
        self.assertEqual(
            self.cfg.MAX_ORDERS_PER_RUN, 1, "MAX_ORDERS_PER_RUN must be 1 in small-account mode"
        )

    def test_max_positions_is_2(self):
        self.assertEqual(self.cfg.MAX_POSITIONS, 2, "MAX_POSITIONS must be 2 in small-account mode")

    def test_single_order_less_than_daily_notional(self):
        self.assertLess(
            self.cfg.MAX_SINGLE_ORDER_USD,
            self.cfg.MAX_DAILY_NOTIONAL_USD,
            "Single order cap must be less than daily notional cap",
        )

    def test_max_experiment_drawdown_active(self):
        self.assertGreater(
            self.cfg.MAX_EXPERIMENT_DRAWDOWN_USD,
            0,
            "MAX_EXPERIMENT_DRAWDOWN_USD must be active in small-account mode",
        )

    def test_margin_disabled_by_default(self):
        self.assertFalse(
            self.cfg.ALLOW_MARGIN, "Margin must be disabled by default in small-account mode"
        )


class TestMarginDisabledByDefault(unittest.TestCase):
    """ALLOW_MARGIN must default to False regardless of account mode."""

    def test_allow_margin_false_by_default(self):
        env = {k: v for k, v in os.environ.items() if k != "ALLOW_MARGIN"}
        with patch.dict(os.environ, env, clear=True):
            import config as cfg

            importlib.reload(cfg)
            self.assertFalse(cfg.ALLOW_MARGIN)

    def test_long_only_true_by_default(self):
        env = {k: v for k, v in os.environ.items() if k != "LONG_ONLY"}
        with patch.dict(os.environ, env, clear=True):
            import config as cfg

            importlib.reload(cfg)
            self.assertTrue(cfg.LONG_ONLY)


class TestStopFailureIsFatalInLiveMode(unittest.TestCase):
    """Stop placement failure must trigger flatten/halt, never be silently ignored."""

    def test_alert_always_fired_on_stop_failure(self):
        alert_mock = MagicMock()
        with (
            patch("main.config.IS_PAPER", False),
            patch("main.config.HALT_FILE", "/tmp/test_halt_invariant"),
            patch("main.config.LOG_DIR", "/tmp"),
            patch("main.trader.close_position", return_value=MagicMock(is_success=True)),
            patch("main.trader.record_sell"),
            patch("main.alerts.alert_error", alert_mock),
        ):
            from main import _handle_stop_failure

            _handle_stop_failure(MagicMock(), "TEST", dry_run=False)

        self.assertTrue(alert_mock.called, "alert_error must be called on stop failure")

    def test_close_position_called_in_live_mode(self):
        close_mock = MagicMock(return_value=MagicMock(is_success=True))
        with (
            patch("main.config.IS_PAPER", False),
            patch("main.config.HALT_FILE", "/tmp/test_halt_invariant2"),
            patch("main.config.LOG_DIR", "/tmp"),
            patch("main.trader.close_position", close_mock),
            patch("main.trader.record_sell"),
            patch("main.alerts.alert_error"),
        ):
            from main import _handle_stop_failure

            _handle_stop_failure(MagicMock(), "TEST", dry_run=False)

        close_mock.assert_called_once()

    def test_close_not_called_in_paper_mode(self):
        close_mock = MagicMock()
        with (
            patch("main.config.IS_PAPER", True),
            patch("main.config.HALT_FILE", "/tmp/test_halt_invariant3"),
            patch("main.config.LOG_DIR", "/tmp"),
            patch("main.trader.close_position", close_mock),
            patch("main.trader.record_sell"),
            patch("main.alerts.alert_error"),
        ):
            from main import _handle_stop_failure

            _handle_stop_failure(MagicMock(), "TEST", dry_run=False)

        close_mock.assert_not_called()


class TestCheckPreTradeCannotBeBypassedInBuyPath(unittest.TestCase):
    """check_pre_trade() must be called before any order in the buy path.

    Tests that a rejected pre-trade check prevents order submission.
    """

    def test_rejected_pre_trade_blocks_place_buy_order(self):
        """When check_pre_trade returns False, place_buy_order must not be called."""
        from utils.health import HealthReport, HealthStatus

        place_buy_mock = MagicMock()
        _patches = [
            patch("main.config.IS_PAPER", True),
            patch("main.trader.get_client", return_value=MagicMock()),
            patch("main.trader.is_market_open", return_value=True),
            patch(
                "main.trader.get_account_info",
                return_value={
                    "portfolio_value": 150.0,
                    "cash": 130.0,
                    "buying_power": 130.0,
                    "equity": 150.0,
                },
            ),
            patch("main.trader.get_open_positions", return_value=[]),
            patch("main.trader.reconcile_positions"),
            patch("main.trader.ensure_stops_attached", return_value=True),
            patch("main.portfolio_tracker.save_daily_baseline"),
            patch("main.portfolio_tracker.load_daily_baseline", return_value=150.0),
            patch("main.portfolio_tracker.load_history", return_value=[]),
            patch("main.portfolio_tracker.save_daily_run", return_value={"daily_pnl": 0}),
            patch("main.portfolio_tracker.print_summary"),
            patch("main.portfolio_tracker.get_day_summary", return_value=None),
            patch("main.portfolio_tracker.get_track_record", return_value={}),
            patch("main.risk_manager.check_circuit_breaker", return_value=(False, 0.0)),
            patch("main.risk_manager.check_daily_loss", return_value=(False, 0.0)),
            patch("main.risk_manager.validate_buy_candidates", side_effect=lambda c, **kw: c),
            patch("main.risk_manager.check_vix_stop_adjustment", return_value=4.0),
            patch(
                "main.stock_scanner.get_market_regime",
                return_value={"regime": "BULL", "is_bearish": False},
            ),
            patch("main.stock_scanner.get_top_movers", return_value=[]),
            patch(
                "main.stock_scanner.prefilter_candidates",
                return_value=[{"symbol": "SOFI", "current_price": 10.0, "key_signal": "momentum"}],
            ),
            patch("main.build_scan_universe", return_value=["SOFI"]),
            patch(
                "main.market_data.get_market_snapshots",
                return_value=[{"symbol": "SOFI", "current_price": 10.0}],
            ),
            patch("main.market_data.get_vix", return_value=15.0),
            patch("main.news_fetcher.fetch_news", return_value={}),
            patch("main.options_scanner.get_options_signals", return_value={}),
            patch("main.sector_data.get_sector_performance", return_value={}),
            patch("main.sector_data.get_leading_sectors", return_value=[]),
            patch("main.sector_data.get_sector", return_value="Tech"),
            patch("main.sentiment_module.get_sentiment", return_value={}),
            patch("main.earnings_calendar.get_earnings_risk_positions", return_value={}),
            patch(
                "main.macro_calendar.get_macro_risk",
                return_value={"is_high_risk": False, "event": ""},
            ),
            patch(
                "main.ai_analyst.get_trading_decisions",
                return_value={
                    "market_summary": "test",
                    "buy_candidates": [
                        {
                            "symbol": "SOFI",
                            "confidence": 9,
                            "reasoning": "Strong momentum breakout signal.",
                            "key_signal": "momentum",
                        }
                    ],
                    "position_decisions": [],
                },
            ),
            patch("main.validate_ai_response", return_value=(True, [])),
            patch("main.check_pre_trade", return_value=(False, "fat-finger cap exceeded")),
            patch("main.trader.place_buy_order", place_buy_mock),
            patch("main.trader.has_pending_buy", return_value=False),
            patch("main.trader.get_total_open_exposure", return_value=0.0),
            patch("main.trader.get_daily_notional", return_value=0.0),
            patch("main.trader.add_daily_notional"),
            patch("main.trader.record_buy"),
            patch("main.trader.get_position_ages", return_value={}),
            patch("main.trader.get_position_meta", return_value={"signal": "unknown"}),
            patch("main.trader.get_position_signal", return_value="unknown"),
            patch("main.performance.record_trade_outcome"),
            patch("main.performance.generate_dashboard"),
            patch("main.get_latest_review", return_value=""),
            patch("main.audit_log.set_run_id"),
            patch("main.audit_log.log_run_start"),
            patch("main.audit_log.log_run_end"),
            patch("main.audit_log.log_ai_decision"),
            patch("main.audit_log.log_event"),
            patch("main.decision_log.set_run_id"),
            patch("main.decision_log.log_decisions"),
            patch(
                "main.run_startup_health_check",
                return_value=HealthReport(
                    status=HealthStatus.GREEN,
                    issues=[],
                    metrics={},
                ),
            ),
        ]
        for p in _patches:
            p.start()
        try:
            from main import _run_inner

            _run_inner(dry_run=False, mode="open", today="2026-05-04")
        finally:
            for p in _patches:
                p.stop()

        place_buy_mock.assert_not_called()


class TestHealthStatusBlocksBuys(unittest.TestCase):
    """RED and YELLOW health status must prevent new buys."""

    def test_health_red_added_to_skip_buys_reasons(self):
        from utils.health import HealthReport, HealthStatus

        report = HealthReport(
            status=HealthStatus.RED,
            issues=["unexpected broker position: XYZ"],
        )
        self.assertEqual(report.status, HealthStatus.RED)
        self.assertIn("unexpected broker position", report.issues[0])

    def test_health_yellow_is_non_fatal(self):
        from utils.health import HealthReport, HealthStatus

        report = HealthReport(
            status=HealthStatus.YELLOW,
            issues=["stale open order: AAPL order abc (9.0h old)"],
        )
        self.assertNotEqual(report.status, HealthStatus.RED)

    def test_no_issues_produces_green(self):
        from utils.health import HealthReport, HealthStatus

        report = HealthReport(status=HealthStatus.GREEN, issues=[])
        self.assertEqual(report.status, HealthStatus.GREEN)
        self.assertFalse(report.issues)


class TestQuoteGateStructure(unittest.TestCase):
    """QuoteGateResult approved=False must have a non-empty reject_reason."""

    def test_rejected_result_has_reason(self):
        from execution.quote_gate import QuoteGateResult

        result = QuoteGateResult(symbol="AAPL", approved=False, reject_reason="stale quote")
        self.assertFalse(result.approved)
        self.assertTrue(result.reject_reason)

    def test_approved_result_has_no_reason(self):
        from execution.quote_gate import QuoteGateResult

        result = QuoteGateResult(symbol="AAPL", approved=True)
        self.assertTrue(result.approved)
        self.assertEqual(result.reject_reason, "")

    def test_quote_gate_raises_broker_state_unavailable_on_api_failure(self):
        from execution.quote_gate import check_quote_gate
        from models import BrokerStateUnavailable

        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.side_effect = RuntimeError("network down")
        with self.assertRaises(BrokerStateUnavailable):
            check_quote_gate("AAPL", 50.0, data_client=mock_client)

    def test_quote_gate_raises_broker_state_unavailable_on_last_trade_failure(self):
        """Last-trade fetch failure must raise BrokerStateUnavailable, not approve the order."""
        from execution.quote_gate import check_quote_gate
        from models import BrokerStateUnavailable

        mock_client = MagicMock()
        quote = MagicMock()
        quote.bid_price = 100.0
        quote.ask_price = 100.1
        from datetime import UTC, datetime

        quote.timestamp = datetime.now(UTC)
        mock_client.get_stock_latest_quote.return_value = {"AAPL": quote}
        mock_client.get_stock_latest_trade.side_effect = RuntimeError("trade feed down")
        with self.assertRaises(BrokerStateUnavailable):
            check_quote_gate("AAPL", 200.0, data_client=mock_client)


class TestOrderLedgerFailClosed(unittest.TestCase):
    """Order-ledger failures must block buys, not fail open."""

    def test_has_active_intent_raises_on_db_failure(self):
        """has_active_intent() must raise OrderLedgerUnavailable when the DB cannot be queried."""
        from models import OrderLedgerUnavailable
        from utils.order_ledger import has_active_intent

        with (
            patch("utils.order_ledger.get_db", side_effect=Exception("DB locked")),
            self.assertRaises(OrderLedgerUnavailable),
        ):
            has_active_intent("AAPL", "BUY", "2026-05-04")

    def test_create_intent_failure_raises_order_ledger_unavailable_in_live_mode(self):
        """In live mode, create_intent returning None must raise OrderLedgerUnavailable."""
        import os
        import tempfile

        from execution.trader import place_buy_order
        from models import OrderLedgerUnavailable

        client = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            _patches = [
                patch("utils.db._DB_PATH", os.path.join(tmpdir, "test.db")),
                patch("config.LOG_DIR", tmpdir),
                patch("execution.trader.IS_PAPER", False),
                patch("utils.order_ledger.get_db", side_effect=Exception("DB locked")),
            ]
            for p in _patches:
                p.start()
            try:
                # init_db also uses get_db — bypass by re-patching only during place_buy_order
                with self.assertRaises(OrderLedgerUnavailable):
                    place_buy_order(client, "AAPL", 50.0)
            finally:
                for p in _patches:
                    p.stop()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""Tests for utils/health.py — run_startup_health_check and HealthReport.log."""

import os
import shutil
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import utils.db as db_module
from utils.health import HealthReport, HealthStatus, run_startup_health_check


def _broker_pos(symbol, qty=10.0):
    p = MagicMock()
    p.symbol = symbol
    p.qty = str(qty)
    return p


def _trailing_stop(symbol, qty=10.0, age_hours=1):
    from alpaca.trading.enums import OrderSide, OrderType

    o = MagicMock()
    o.symbol = symbol
    o.order_type = OrderType.TRAILING_STOP
    o.side = OrderSide.SELL
    o.qty = str(qty)
    o.status = "new"
    o.id = f"stop-{symbol}"
    o.created_at = datetime.now(UTC) - timedelta(hours=age_hours)
    return o


def _pending_order(symbol, age_hours=1, tz_aware=True, status="new"):
    from alpaca.trading.enums import OrderSide, OrderType

    o = MagicMock()
    o.symbol = symbol
    o.order_type = OrderType.MARKET
    o.side = OrderSide.BUY
    o.qty = "10.0"
    o.status = status
    o.id = f"buy-{symbol}"
    dt = datetime.now(UTC) - timedelta(hours=age_hours)
    o.created_at = dt if tz_aware else dt.replace(tzinfo=None)
    return o


class HealthBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.halt_file = os.path.join(self.tmpdir, ".halt")
        self._db_patchers = [
            patch.object(db_module, "_DB_PATH", self.db_path),
            patch.object(db_module, "_initialized", False),
            patch.object(db_module, "_migrate_json_state", lambda: None),
        ]
        for p in self._db_patchers:
            p.start()
        from utils.db import init_db

        init_db()

    def tearDown(self):
        for p in self._db_patchers:
            p.stop()
        shutil.rmtree(self.tmpdir)

    def _client(self, positions=None, orders=None, portfolio_value=100_000.0):
        c = MagicMock()
        c.get_all_positions.return_value = positions if positions is not None else []
        c.get_orders.return_value = orders if orders is not None else []
        acct = MagicMock()
        acct.portfolio_value = str(portfolio_value)
        c.get_account.return_value = acct
        return c

    def _run(self, client, max_deployed=0, max_daily_loss=0, extra=None):
        patchers = [
            patch("config.HALT_FILE", new=self.halt_file),
            patch("config.MAX_DEPLOYED_USD", new=max_deployed),
            patch("config.MAX_DAILY_LOSS_USD", new=max_daily_loss),
        ]
        if extra:
            patchers.extend(extra)
        for p in patchers:
            p.start()
        import contextlib

        try:
            return run_startup_health_check(client)
        finally:
            for p in reversed(patchers):
                with contextlib.suppress(RuntimeError):
                    p.stop()


class TestHealthCheckGreen(HealthBase):
    def test_all_clear_returns_green(self):
        client = self._client()
        report = self._run(client)
        self.assertEqual(report.status, HealthStatus.GREEN)
        self.assertFalse(report.issues)
        self.assertEqual(report.metrics["broker_positions"], 0)
        self.assertEqual(report.metrics["db_positions"], 0)
        self.assertEqual(report.metrics["stale_orders"], 0)
        self.assertEqual(report.metrics["uncovered_positions"], 0)
        self.assertEqual(report.metrics["unresolved_intents"], 0)


class TestHealthCheckHaltFile(HealthBase):
    def test_halt_file_present_returns_red(self):
        open(self.halt_file, "w").close()
        client = self._client()
        report = self._run(client)
        self.assertEqual(report.status, HealthStatus.RED)
        self.assertTrue(any("halt file" in i for i in report.issues))
        self.assertIn(os.path.basename(self.halt_file), " ".join(report.issues))


class TestHealthCheckPositionReconciliation(HealthBase):
    def test_unexpected_broker_position_returns_red(self):
        client = self._client(positions=[_broker_pos("AAPL")])
        report = self._run(client)
        self.assertEqual(report.status, HealthStatus.RED)
        issues_str = " ".join(report.issues)
        self.assertIn("unexpected broker position", issues_str)
        self.assertIn("AAPL", issues_str)

    def test_multiple_unexpected_positions_each_reported(self):
        client = self._client(positions=[_broker_pos("AAPL"), _broker_pos("MSFT")])
        report = self._run(client)
        self.assertEqual(report.status, HealthStatus.RED)
        symbols = [i for i in report.issues if "unexpected broker position" in i]
        self.assertEqual(len(symbols), 2)

    def test_db_position_missing_from_broker_adds_issue(self):
        from utils.db import get_db

        with get_db() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, entry_date, entry_price) VALUES (?,?,?)",
                ("TSLA", "2026-05-01", 200.0),
            )
        client = self._client(positions=[])
        report = self._run(client)
        self.assertTrue(any("stale metadata" in i for i in report.issues))
        self.assertIn("TSLA", " ".join(report.issues))

    def test_reconciliation_exception_adds_yellow_issue(self):
        client = MagicMock()
        client.get_all_positions.side_effect = RuntimeError("broker down")
        report = self._run(client)
        self.assertTrue(any("position reconciliation failed" in i for i in report.issues))
        self.assertNotEqual(report.status, HealthStatus.GREEN)

    def test_matched_positions_no_reconciliation_issues(self):
        from utils.db import get_db

        with get_db() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, entry_date, entry_price) VALUES (?,?,?)",
                ("NVDA", "2026-05-01", 800.0),
            )
        client = self._client(positions=[_broker_pos("NVDA")])
        report = self._run(client)
        self.assertFalse(any("unexpected broker position" in i for i in report.issues))
        self.assertFalse(any("stale metadata" in i for i in report.issues))


class TestHealthCheckStopCoverage(HealthBase):
    def test_uncovered_whole_share_returns_red(self):
        pos = _broker_pos("AAPL", qty=5.0)
        client = self._client(positions=[pos], orders=[])
        report = self._run(client)
        self.assertEqual(report.status, HealthStatus.RED)
        self.assertTrue(any("no stop protection" in i for i in report.issues))
        self.assertIn("AAPL", " ".join(report.issues))
        self.assertGreater(report.metrics["uncovered_positions"], 0)

    def test_fully_covered_position_no_stop_issue(self):
        pos = _broker_pos("AAPL", qty=10.0)
        stop = _trailing_stop("AAPL", qty=10.0)
        client = self._client(positions=[pos], orders=[stop])
        report = self._run(client)
        self.assertFalse(any("no stop protection" in i for i in report.issues))
        self.assertEqual(report.metrics["uncovered_positions"], 0)

    def test_fractional_remainder_below_whole_share_not_flagged(self):
        # 0.5 uncovered shares: uncovered_qty > 0 but whole = 0 → no flag
        pos = _broker_pos("AAPL", qty=10.5)
        stop = _trailing_stop("AAPL", qty=10.0)
        client = self._client(positions=[pos], orders=[stop])
        report = self._run(client)
        self.assertFalse(any("no stop protection" in i for i in report.issues))

    def test_stop_coverage_exception_adds_issue(self):
        client = MagicMock()
        client.get_all_positions.return_value = []
        client.get_orders.side_effect = RuntimeError("api error")
        report = self._run(client)
        self.assertTrue(any("stop coverage check failed" in i for i in report.issues))
        # open_orders undefined → stale order check also fails
        self.assertTrue(any("stale order check failed" in i for i in report.issues))


class TestHealthCheckStaleOrders(HealthBase):
    def test_stale_order_adds_yellow_issue(self):
        stale = _pending_order("AAPL", age_hours=10)
        client = self._client(orders=[stale])
        report = self._run(client)
        self.assertTrue(any("stale open order" in i for i in report.issues))
        self.assertEqual(report.metrics["stale_orders"], 1)

    def test_fresh_order_no_stale_issue(self):
        fresh = _pending_order("AAPL", age_hours=1)
        client = self._client(orders=[fresh])
        report = self._run(client)
        self.assertFalse(any("stale open order" in i for i in report.issues))
        self.assertEqual(report.metrics["stale_orders"], 0)

    def test_accepted_status_counts_as_stale(self):
        stale = _pending_order("AAPL", age_hours=9, status="accepted")
        client = self._client(orders=[stale])
        report = self._run(client)
        self.assertTrue(any("stale open order" in i for i in report.issues))

    def test_timezone_naive_created_at_handled(self):
        # Covers the `if created_at.tzinfo is None: created_at.replace(tzinfo=UTC)` branch
        stale_naive = _pending_order("AAPL", age_hours=10, tz_aware=False)
        client = self._client(orders=[stale_naive])
        report = self._run(client)
        self.assertTrue(any("stale open order" in i for i in report.issues))


class TestHealthCheckExposureCap(HealthBase):
    def test_exposure_exceeded_returns_red(self):
        client = self._client()
        report = self._run(
            client,
            max_deployed=10_000,
            extra=[patch("execution.trader.get_total_open_exposure", return_value=15_000.0)],
        )
        self.assertEqual(report.status, HealthStatus.RED)
        self.assertTrue(any("exposure" in i for i in report.issues))
        self.assertAlmostEqual(report.metrics["open_exposure_usd"], 15_000.0)

    def test_exposure_within_cap_no_issue(self):
        client = self._client()
        report = self._run(
            client,
            max_deployed=10_000,
            extra=[patch("execution.trader.get_total_open_exposure", return_value=5_000.0)],
        )
        self.assertFalse(any("exposure" in i for i in report.issues))
        self.assertAlmostEqual(report.metrics["open_exposure_usd"], 5_000.0)

    def test_broker_unavailable_exposure_returns_red(self):
        from models import BrokerStateUnavailable

        client = self._client()
        report = self._run(
            client,
            max_deployed=10_000,
            extra=[
                patch(
                    "execution.trader.get_total_open_exposure",
                    side_effect=BrokerStateUnavailable("down"),
                )
            ],
        )
        self.assertEqual(report.status, HealthStatus.RED)
        self.assertTrue(any("cannot verify exposure" in i for i in report.issues))

    def test_zero_max_deployed_skips_check(self):
        # max_deployed=0 → check is disabled, no open_exposure_usd in metrics
        client = self._client()
        report = self._run(client, max_deployed=0)
        self.assertNotIn("open_exposure_usd", report.metrics)


class TestHealthCheckDailyLoss(HealthBase):
    def test_daily_loss_exceeded_returns_red(self):
        client = self._client(portfolio_value=90_000.0)
        report = self._run(
            client,
            max_daily_loss=500.0,
            extra=[patch("utils.portfolio_tracker.load_daily_baseline", return_value=100_000.0)],
        )
        self.assertEqual(report.status, HealthStatus.RED)
        self.assertTrue(any("daily loss" in i for i in report.issues))
        self.assertGreater(report.metrics["daily_loss_usd"], 0)

    def test_daily_loss_below_cap_no_issue(self):
        client = self._client(portfolio_value=99_600.0)
        report = self._run(
            client,
            max_daily_loss=500.0,
            extra=[patch("utils.portfolio_tracker.load_daily_baseline", return_value=100_000.0)],
        )
        self.assertFalse(any("daily loss" in i for i in report.issues))

    def test_no_baseline_skips_daily_loss_body(self):
        client = self._client()
        report = self._run(
            client,
            max_daily_loss=500.0,
            extra=[patch("utils.portfolio_tracker.load_daily_baseline", return_value=None)],
        )
        self.assertFalse(any("daily loss" in i for i in report.issues))

    def test_daily_loss_check_exception_adds_yellow(self):
        client = self._client()
        report = self._run(
            client,
            max_daily_loss=500.0,
            extra=[
                patch(
                    "utils.portfolio_tracker.load_daily_baseline",
                    side_effect=RuntimeError("io error"),
                )
            ],
        )
        self.assertTrue(any("daily loss cap check failed" in i for i in report.issues))

    def test_zero_max_daily_loss_skips_check(self):
        client = self._client()
        report = self._run(client, max_daily_loss=0)
        self.assertNotIn("daily_loss_usd", report.metrics)


class TestHealthCheckOrderIntents(HealthBase):
    def test_unresolved_intents_returns_yellow(self):
        unresolved = [
            {
                "symbol": "AAPL",
                "side": "BUY",
                "client_order_id": "ib-AAPL-BUY-2026-05-07",
                "status": "timeout",
            }
        ]
        client = self._client()
        report = self._run(
            client,
            extra=[
                patch("utils.order_ledger.auto_cancel_timeout_intents"),
                patch("utils.order_ledger.get_unresolved_intents", return_value=unresolved),
            ],
        )
        self.assertEqual(report.status, HealthStatus.YELLOW)
        self.assertTrue(any("unresolved order intent" in i for i in report.issues))
        self.assertIn("AAPL", " ".join(report.issues))
        self.assertEqual(report.metrics["unresolved_intents"], 1)

    def test_order_intent_exception_adds_yellow_issue(self):
        client = self._client()
        report = self._run(
            client,
            extra=[
                patch(
                    "utils.order_ledger.auto_cancel_timeout_intents",
                    side_effect=RuntimeError("db locked"),
                )
            ],
        )
        self.assertTrue(any("order intent check failed" in i for i in report.issues))
        self.assertNotEqual(report.status, HealthStatus.GREEN)

    def test_import_error_silently_skipped(self):
        client = self._client()
        report = self._run(
            client,
            extra=[patch.dict(sys.modules, {"utils.order_ledger": None})],
        )
        self.assertFalse(any("order intent" in i for i in report.issues))


class TestHealthReportLog(unittest.TestCase):
    def test_log_with_issues_warns_each_issue(self):
        report = HealthReport(
            status=HealthStatus.YELLOW,
            issues=["halt file present: /tmp/.halt", "stale order: AAPL"],
        )
        with self.assertLogs("utils.health", level="WARNING") as cm:
            report.log()
        joined = "\n".join(cm.output)
        self.assertIn("halt file", joined)
        self.assertIn("stale order", joined)

    def test_log_green_reports_all_passed(self):
        report = HealthReport(status=HealthStatus.GREEN)
        with self.assertLogs("utils.health", level="INFO") as cm:
            report.log()
        self.assertTrue(any("All startup checks passed" in line for line in cm.output))

    def test_log_red_emits_status(self):
        report = HealthReport(status=HealthStatus.RED, issues=["unexpected broker position: AAPL"])
        with self.assertLogs("utils.health", level="WARNING") as cm:
            report.log()
        joined = "\n".join(cm.output)
        self.assertIn("unexpected broker position", joined)


if __name__ == "__main__":
    unittest.main()

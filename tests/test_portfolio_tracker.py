import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from utils.portfolio_tracker import (
    get_day_summary,
    get_track_record,
    load_history,
    save_daily_run,
)


def _account(value, cash=10_000):
    return {"portfolio_value": value, "cash": cash}


def _ai(summary="quiet"):
    return {"market_summary": summary, "position_decisions": [], "buy_candidates": []}


class PortfolioTrackerBase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_patcher = patch("utils.portfolio_tracker.LOG_DIR", self.tmpdir)
        self.log_patcher.start()

    def tearDown(self):
        self.log_patcher.stop()
        shutil.rmtree(self.tmpdir)


class TestSaveAndLoad(PortfolioTrackerBase):

    def test_save_creates_json_file(self):
        save_daily_run("2026-01-15", _account(100_000), _account(101_000), _ai(), [], [])
        # File is saved in a weekly subdirectory — search recursively
        all_files = [
            fname
            for _, _, files in os.walk(self.tmpdir)
            for fname in files
        ]
        self.assertTrue(any("2026-01-15" in f for f in all_files))

    def test_load_returns_saved_record(self):
        save_daily_run("2026-01-15", _account(100_000), _account(101_000), _ai(), [], [])
        records = load_history()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["date"], "2026-01-15")

    def test_load_ignores_non_run_files(self):
        with open(os.path.join(self.tmpdir, "positions_meta.json"), "w") as f:
            f.write('{"AAPL": {}}')
        save_daily_run("2026-01-15", _account(100_000), _account(100_000), _ai(), [], [])
        records = load_history()
        self.assertEqual(len(records), 1)

    def test_load_ignores_malformed_json(self):
        with open(os.path.join(self.tmpdir, "bad.json"), "w") as f:
            f.write("not json {{{")
        records = load_history()
        self.assertEqual(len(records), 0)

    def test_daily_pnl_stored_correctly(self):
        save_daily_run("2026-01-15", _account(100_000), _account(101_500), _ai(), [], [])
        records = load_history()
        self.assertAlmostEqual(records[0]["daily_pnl"], 1_500.0)


class TestGetDaySummary(PortfolioTrackerBase):

    def test_returns_none_when_no_records(self):
        self.assertIsNone(get_day_summary("2026-01-15"))

    def test_merges_open_and_close_runs(self):
        trade_a = {"symbol": "AAPL", "action": "BUY", "detail": "$5000"}
        trade_b = {"symbol": "AAPL", "action": "SELL", "detail": "earnings exit"}

        save_daily_run("2026-01-15", _account(100_000), _account(100_500), _ai("open"), [trade_a], [])
        save_daily_run("2026-01-15-close", _account(100_500), _account(101_200), _ai("close"), [trade_b], [])

        summary = get_day_summary("2026-01-15")
        self.assertIsNotNone(summary)
        self.assertEqual(len(summary["trades_executed"]), 2)

    def test_uses_first_account_before_and_last_account_after(self):
        save_daily_run("2026-01-15", _account(100_000), _account(100_500), _ai(), [], [])
        save_daily_run("2026-01-15-close", _account(100_500), _account(101_200), _ai(), [], [])

        summary = get_day_summary("2026-01-15")
        self.assertEqual(summary["account_before"]["portfolio_value"], 100_000)
        self.assertEqual(summary["account_after"]["portfolio_value"], 101_200)

    def test_daily_pnl_is_end_to_end(self):
        save_daily_run("2026-01-15", _account(100_000), _account(100_500), _ai(), [], [])
        save_daily_run("2026-01-15-close", _account(100_500), _account(101_000), _ai(), [], [])

        summary = get_day_summary("2026-01-15")
        self.assertAlmostEqual(summary["daily_pnl"], 1_000.0)


class TestPrintSummary(PortfolioTrackerBase):

    def _record(self, pnl=500.0, trades=None, stops=None):
        return {
            "date": "2026-01-15",
            "market_summary": "Quiet day",
            "account_after": {"portfolio_value": 100_500, "cash": 20_000},
            "daily_pnl": pnl,
            "trades_executed": trades or [],
            "stop_losses_triggered": stops or [],
        }

    def test_print_summary_does_not_raise(self):
        import io
        import sys

        from utils.portfolio_tracker import print_summary
        captured = io.StringIO()
        sys.stdout = captured
        try:
            print_summary(self._record())
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("2026-01-15", captured.getvalue())

    def test_print_summary_shows_pnl(self):
        import io
        import sys

        from utils.portfolio_tracker import print_summary
        captured = io.StringIO()
        sys.stdout = captured
        try:
            print_summary(self._record(pnl=250.0))
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("250", captured.getvalue())

    def test_print_summary_shows_trades(self):
        import io
        import sys

        from utils.portfolio_tracker import print_summary
        trades = [{"action": "BUY", "symbol": "NVDA", "detail": "$5000"}]
        captured = io.StringIO()
        sys.stdout = captured
        try:
            print_summary(self._record(trades=trades))
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("NVDA", captured.getvalue())

    def test_print_summary_no_trades_message(self):
        import io
        import sys

        from utils.portfolio_tracker import print_summary
        captured = io.StringIO()
        sys.stdout = captured
        try:
            print_summary(self._record())
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("No trades", captured.getvalue())


class TestGetTrackRecord(PortfolioTrackerBase):

    def test_returns_last_n_days(self):
        for i in range(1, 8):
            save_daily_run(f"2026-01-0{i}", _account(100_000), _account(100_000), _ai(), [], [])
        record = get_track_record(n_days=5)
        self.assertEqual(len(record), 5)

    def test_record_has_required_fields(self):
        save_daily_run("2026-01-01", _account(100_000), _account(101_000), _ai("up day"),
                       [{"symbol": "AAPL", "action": "BUY", "detail": "$5000"}], [])
        record = get_track_record(n_days=1)
        entry = record[0]
        self.assertIn("date", entry)
        self.assertIn("daily_pnl_usd", entry)
        self.assertIn("trades", entry)

    def test_stop_losses_appear_as_stop_loss_action(self):
        stop = {"symbol": "NVDA", "pl_pct": -8.5}
        save_daily_run("2026-01-01", _account(100_000), _account(91_500), _ai(), [], [stop])
        record = get_track_record(n_days=1)
        trade_actions = [t["action"] for t in record[0]["trades"]]
        self.assertIn("STOP_LOSS", trade_actions)


class TestTradeMergeGuard(PortfolioTrackerBase):

    def test_second_run_merges_new_trades(self):
        trade_a = {"symbol": "AAPL", "action": "BUY", "detail": "$5000", "order_id": "o1"}
        trade_b = {"symbol": "MSFT", "action": "BUY", "detail": "$3000", "order_id": "o2"}
        save_daily_run("2026-01-15", _account(100_000), _account(100_500), _ai(), [trade_a], [])
        save_daily_run("2026-01-15", _account(100_500), _account(101_200), _ai(), [trade_b], [])
        records = load_history()
        run = next(r for r in records if r["date"] == "2026-01-15")
        order_ids = {t.get("order_id") for t in run["trades_executed"]}
        self.assertIn("o1", order_ids)
        self.assertIn("o2", order_ids)

    def test_second_run_preserves_original_account_before(self):
        trade_a = {"symbol": "AAPL", "action": "BUY", "detail": "$5000", "order_id": "o1"}
        save_daily_run("2026-01-15", _account(100_000), _account(100_500), _ai(), [trade_a], [])
        save_daily_run("2026-01-15", _account(100_500), _account(101_200), _ai(), [], [])
        records = load_history()
        run = next(r for r in records if r["date"] == "2026-01-15")
        self.assertEqual(run["account_before"]["portfolio_value"], 100_000)

    def test_spurious_rerun_with_same_trades_does_not_duplicate(self):
        trade = {"symbol": "AAPL", "action": "BUY", "detail": "$5000", "order_id": "o1"}
        save_daily_run("2026-01-15", _account(100_000), _account(100_500), _ai(), [trade], [])
        save_daily_run("2026-01-15", _account(100_500), _account(101_200), _ai(), [trade], [])
        records = load_history()
        run = next(r for r in records if r["date"] == "2026-01-15")
        self.assertEqual(len(run["trades_executed"]), 1)


class TestSaveDailyBaselinePortfolio(PortfolioTrackerBase):

    def test_writes_baseline_file(self):
        from utils.portfolio_tracker import save_daily_baseline
        baseline_path = os.path.join(self.tmpdir, "daily_baseline.json")
        with patch("utils.portfolio_tracker._BASELINE_PATH", baseline_path):
            save_daily_baseline(100_000.0)
        self.assertTrue(os.path.exists(baseline_path))

    def test_baseline_file_contains_portfolio_value(self):
        import json as _json
        from utils.portfolio_tracker import save_daily_baseline
        baseline_path = os.path.join(self.tmpdir, "daily_baseline.json")
        with patch("utils.portfolio_tracker._BASELINE_PATH", baseline_path):
            save_daily_baseline(99_500.0)
        with open(baseline_path) as f:
            data = _json.load(f)
        self.assertAlmostEqual(data["portfolio_value"], 99_500.0)

    def test_invalid_date_string_uses_today(self):
        from utils.portfolio_tracker import _weekly_log_dir
        # Non-ISO date triggers ValueError → falls back to today()
        result = _weekly_log_dir("not-a-date-at-all!!")
        self.assertTrue(os.path.isdir(result))

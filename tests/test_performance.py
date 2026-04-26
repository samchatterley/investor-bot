import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from analysis.performance import (
    compute_metrics,
    _empty_bucket,
    _update_bucket,
    _bucket_summary,
    record_trade_outcome,
    get_win_rates,
    get_actionable_feedback,
)


def _run(date_str, before, after, pnl=None):
    if pnl is None:
        pnl = after - before
    return {
        "date": date_str,
        "account_before": {"portfolio_value": before},
        "account_after": {"portfolio_value": after},
        "daily_pnl": pnl,
        "trades_executed": [],
    }


class TestComputeMetrics(unittest.TestCase):

    def test_empty_returns_empty_dict(self):
        self.assertEqual(compute_metrics([]), {})

    def test_single_record_total_return_zero(self):
        records = [_run("2026-01-01", 100_000, 100_000)]
        m = compute_metrics(records)
        self.assertEqual(m["total_return_pct"], 0.0)

    def test_positive_total_return(self):
        records = [_run("2026-01-01", 100_000, 100_000),
                   _run("2026-01-02", 100_000, 110_000, pnl=10_000)]
        m = compute_metrics(records)
        self.assertAlmostEqual(m["total_return_pct"], 10.0, places=1)

    def test_max_drawdown_is_negative_or_zero(self):
        records = [_run("2026-01-01", 100_000, 100_000),
                   _run("2026-01-02", 100_000, 90_000, pnl=-10_000)]
        m = compute_metrics(records)
        self.assertLessEqual(m["max_drawdown_pct"], 0.0)

    def test_max_drawdown_magnitude(self):
        records = [
            _run("2026-01-01", 100_000, 100_000),
            _run("2026-01-02", 100_000, 100_000),
            _run("2026-01-03", 100_000, 80_000, pnl=-20_000),
        ]
        m = compute_metrics(records)
        self.assertAlmostEqual(m["max_drawdown_pct"], -20.0, places=1)

    def test_win_rate_all_positive_days(self):
        records = [
            _run("2026-01-01", 100_000, 101_000, pnl=1_000),
            _run("2026-01-02", 101_000, 102_000, pnl=1_000),
        ]
        m = compute_metrics(records)
        self.assertEqual(m["win_rate_pct"], 100.0)

    def test_win_rate_all_negative_days(self):
        records = [
            _run("2026-01-01", 100_000, 99_000, pnl=-1_000),
            _run("2026-01-02", 99_000, 98_000, pnl=-1_000),
        ]
        m = compute_metrics(records)
        self.assertEqual(m["win_rate_pct"], 0.0)

    def test_sharpe_is_zero_for_single_record(self):
        records = [_run("2026-01-01", 100_000, 101_000, pnl=1_000)]
        m = compute_metrics(records)
        self.assertEqual(m["sharpe"], 0.0)

    def test_days_traded_count(self):
        records = [_run(f"2026-01-0{i}", 100_000, 100_000) for i in range(1, 6)]
        m = compute_metrics(records)
        self.assertEqual(m["days_traded"], 5)


class TestBucketHelpers(unittest.TestCase):

    def test_empty_bucket_structure(self):
        b = _empty_bucket()
        self.assertEqual(b["trades"], 0)
        self.assertEqual(b["wins"], 0)
        self.assertEqual(b["losses"], 0)

    def test_update_bucket_win(self):
        b = _empty_bucket()
        _update_bucket(b, 5.0)
        self.assertEqual(b["trades"], 1)
        self.assertEqual(b["wins"], 1)
        self.assertEqual(b["losses"], 0)

    def test_update_bucket_loss(self):
        b = _empty_bucket()
        _update_bucket(b, -3.0)
        self.assertEqual(b["trades"], 1)
        self.assertEqual(b["wins"], 0)
        self.assertEqual(b["losses"], 1)

    def test_bucket_summary_empty(self):
        s = _bucket_summary(_empty_bucket())
        self.assertEqual(s["trades"], 0)
        self.assertEqual(s["win_rate"], 0.0)

    def test_bucket_summary_win_rate(self):
        b = _empty_bucket()
        _update_bucket(b, 5.0)
        _update_bucket(b, -2.0)
        s = _bucket_summary(b)
        self.assertEqual(s["win_rate"], 50.0)
        self.assertEqual(s["trades"], 2)


class TestSignalTracking(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.stats_patcher = patch(
            "analysis.performance._STATS_PATH",
            os.path.join(self.tmpdir, "signal_stats.json"),
        )
        self.stats_patcher.start()

    def tearDown(self):
        self.stats_patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_record_and_retrieve_win_rate(self):
        record_trade_outcome("momentum", 5.0, regime="BULL_TRENDING", confidence=8)
        record_trade_outcome("momentum", -2.0, regime="BULL_TRENDING", confidence=8)
        rates = get_win_rates()
        self.assertIn("momentum", rates)
        self.assertEqual(rates["momentum"]["trades"], 2)
        self.assertEqual(rates["momentum"]["win_rate"], 50.0)

    def test_regime_breakdown_populated(self):
        record_trade_outcome("momentum", 3.0, regime="BULL_TRENDING", confidence=8)
        rates = get_win_rates()
        self.assertIn("BULL_TRENDING", rates["momentum"]["by_regime"])

    def test_confidence_breakdown_populated(self):
        record_trade_outcome("momentum", 3.0, regime="BULL_TRENDING", confidence=8)
        rates = get_win_rates()
        self.assertIn("8", rates["momentum"]["by_confidence"])

    def test_get_actionable_feedback_empty_stats(self):
        result = get_actionable_feedback()
        self.assertEqual(result, "")

    def test_get_actionable_feedback_insufficient_trades(self):
        record_trade_outcome("momentum", 5.0)
        record_trade_outcome("momentum", 5.0)
        result = get_actionable_feedback()
        self.assertEqual(result, "")

    def test_get_actionable_feedback_enough_trades(self):
        for _ in range(4):
            record_trade_outcome("momentum", 5.0, confidence=8)
        result = get_actionable_feedback()
        self.assertIn("momentum", result)
        self.assertIn("win rate", result.lower())

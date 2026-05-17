import contextlib
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from analysis.performance import (
    _aggregate_dimension,
    _attribution_table_html,
    _bucket_summary,
    _empty_bucket,
    _hold_bucket,
    _update_bucket,
    compute_metrics,
    generate_dashboard,
    get_actionable_feedback,
    get_attribution,
    get_attribution_summary,
    get_win_rates,
    record_trade_outcome,
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
        records = [
            _run("2026-01-01", 100_000, 100_000),
            _run("2026-01-02", 100_000, 110_000, pnl=10_000),
        ]
        m = compute_metrics(records)
        self.assertAlmostEqual(m["total_return_pct"], 10.0, places=1)

    def test_max_drawdown_is_negative_or_zero(self):
        records = [
            _run("2026-01-01", 100_000, 100_000),
            _run("2026-01-02", 100_000, 90_000, pnl=-10_000),
        ]
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

    def test_actionable_feedback_verdict_working_well(self):
        # ≥60% win rate → "working well"
        for _ in range(4):
            record_trade_outcome("breakout", 3.0, confidence=8)
        result = get_actionable_feedback()
        self.assertIn("working well", result)

    def test_actionable_feedback_verdict_marginal(self):
        # 50% win rate → "marginal"
        for _ in range(2):
            record_trade_outcome("breakout", 3.0, confidence=7)
        for _ in range(2):
            record_trade_outcome("breakout", -2.0, confidence=7)
        result = get_actionable_feedback()
        self.assertIn("marginal", result)

    def test_actionable_feedback_verdict_underperforming(self):
        # 0% win rate → "underperforming"
        for _ in range(4):
            record_trade_outcome("breakout", -3.0, confidence=6)
        result = get_actionable_feedback()
        self.assertIn("underperforming", result)

    def test_actionable_feedback_includes_regime_note(self):
        # ≥2 trades in a regime with strong win rate adds a regime note
        for _ in range(3):
            record_trade_outcome("breakout", 5.0, regime="BULL_TRENDING", confidence=8)
        result = get_actionable_feedback()
        self.assertIn("BULL_TRENDING", result)

    def test_load_stats_handles_corrupt_json(self):
        corrupt_path = os.path.join(self.tmpdir, "signal_stats.json")
        with open(corrupt_path, "w") as f:
            f.write("{ not valid json }")
        from analysis.performance import _load_stats

        result = _load_stats()
        self.assertEqual(result, {})

    def test_get_win_rates_skips_signal_with_zero_trades(self):
        # Line 91: the `continue` when data.get("trades", 0) == 0
        zero_trades_stats = {
            "momentum": {"trades": 0, "wins": 0, "losses": 0, "total_return_pct": 0.0},
            "rsi_oversold": {"trades": 2, "wins": 1, "losses": 1, "total_return_pct": 3.0},
        }
        with patch("analysis.performance._load_stats", return_value=zero_trades_stats):
            rates = get_win_rates()
        self.assertNotIn("momentum", rates)
        self.assertIn("rsi_oversold", rates)


class TestGenerateDashboard(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dashboard_patcher = patch(
            "analysis.performance._DASHBOARD_PATH",
            os.path.join(self.tmpdir, "dashboard.html"),
        )
        self.stats_patcher = patch(
            "analysis.performance._STATS_PATH",
            os.path.join(self.tmpdir, "signal_stats.json"),
        )
        self.dashboard_patcher.start()
        self.stats_patcher.start()
        self.addCleanup(self.dashboard_patcher.stop)
        self.addCleanup(self.stats_patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def _run(self, date_str, before, after, pnl=None):
        if pnl is None:  # pragma: no cover
            pnl = after - before
        return {
            "date": date_str,
            "account_before": {"portfolio_value": before},
            "account_after": {"portfolio_value": after},
            "daily_pnl": pnl,
            "trades_executed": [],
        }

    def test_creates_html_file(self):
        records = [self._run("2026-01-01", 100_000, 101_000, pnl=1_000)]
        generate_dashboard(records)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "dashboard.html")))

    def test_does_nothing_for_empty_records(self):
        generate_dashboard([])
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "dashboard.html")))

    def test_html_contains_portfolio_value(self):
        records = [self._run("2026-01-01", 100_000, 105_000, pnl=5_000)]
        generate_dashboard(records)
        with open(os.path.join(self.tmpdir, "dashboard.html")) as f:
            html = f.read()
        self.assertIn("105000", html)

    def test_html_contains_trade_rows_when_trades_present(self):
        records = [self._run("2026-01-01", 100_000, 101_000, pnl=1_000)]
        records[0]["trades_executed"] = [{"action": "BUY", "symbol": "AAPL", "detail": "$5000"}]
        generate_dashboard(records)
        with open(os.path.join(self.tmpdir, "dashboard.html")) as f:
            html = f.read()
        self.assertIn("AAPL", html)


class TestHoldBucket(unittest.TestCase):
    def test_1d(self):
        self.assertEqual(_hold_bucket(1), "1d")

    def test_0_days_maps_to_1d(self):
        self.assertEqual(_hold_bucket(0), "1d")

    def test_2d(self):
        self.assertEqual(_hold_bucket(2), "2d")

    def test_3d(self):
        self.assertEqual(_hold_bucket(3), "3d")

    def test_4d_plus(self):
        self.assertEqual(_hold_bucket(4), "4d+")
        self.assertEqual(_hold_bucket(10), "4d+")


class TestAttributionDimensions(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patcher = patch(
            "analysis.performance._STATS_PATH",
            os.path.join(self.tmpdir, "signal_stats.json"),
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_sector_breakdown_populated(self):
        record_trade_outcome("momentum", 3.0, sector="Technology")
        rates = get_win_rates()
        self.assertIn("Technology", rates["momentum"]["by_sector"])

    def test_hold_days_breakdown_populated(self):
        record_trade_outcome("momentum", 3.0, hold_days=2)
        rates = get_win_rates()
        self.assertIn("2d", rates["momentum"]["by_hold_days"])

    def test_hold_days_zero_maps_to_unknown(self):
        record_trade_outcome("momentum", 3.0, hold_days=0)
        rates = get_win_rates()
        self.assertIn("unknown", rates["momentum"]["by_hold_days"])

    def test_multiple_dimensions_recorded_together(self):
        record_trade_outcome(
            "mean_reversion",
            2.5,
            regime="CHOPPY",
            confidence=7,
            sector="Financials",
            hold_days=3,
        )
        rates = get_win_rates()
        entry = rates["mean_reversion"]
        self.assertIn("CHOPPY", entry["by_regime"])
        self.assertIn("Financials", entry["by_sector"])
        self.assertIn("3d", entry["by_hold_days"])

    def test_attribution_summary_empty_stats(self):
        result = get_attribution_summary()
        self.assertEqual(result, {})

    def test_attribution_summary_by_signal_ranked_by_avg_return(self):
        record_trade_outcome("momentum", 5.0)
        record_trade_outcome("momentum", 3.0)
        record_trade_outcome("mean_reversion", -1.0)
        result = get_attribution_summary()
        signals = list(result["by_signal"].keys())
        self.assertEqual(signals[0], "momentum")
        self.assertEqual(signals[-1], "mean_reversion")

    def test_attribution_summary_by_sector_aggregated_across_signals(self):
        record_trade_outcome("momentum", 4.0, sector="Technology")
        record_trade_outcome("mean_reversion", 2.0, sector="Technology")
        result = get_attribution_summary()
        self.assertIn("Technology", result["by_sector"])
        self.assertEqual(result["by_sector"]["Technology"]["trades"], 2)

    def test_attribution_summary_by_hold_days_populated(self):
        record_trade_outcome("momentum", 3.0, hold_days=1)
        record_trade_outcome("momentum", 5.0, hold_days=3)
        result = get_attribution_summary()
        self.assertIn("1d", result["by_hold_days"])
        self.assertIn("3d", result["by_hold_days"])

    def test_attribution_summary_best_signal_is_highest_avg_return(self):
        record_trade_outcome("momentum", 5.0)
        record_trade_outcome("mean_reversion", 1.0)
        result = get_attribution_summary()
        self.assertEqual(result["best_signal"], "momentum")

    def test_attribution_summary_worst_signal_is_lowest_avg_return(self):
        record_trade_outcome("momentum", 5.0)
        record_trade_outcome("mean_reversion", -2.0)
        result = get_attribution_summary()
        self.assertEqual(result["worst_signal"], "mean_reversion")

    def test_attribution_summary_optimal_hold_is_best_bucket(self):
        record_trade_outcome("momentum", 5.0, hold_days=2)
        record_trade_outcome("momentum", 1.0, hold_days=4)
        result = get_attribution_summary()
        self.assertEqual(result["optimal_hold"], "2d")

    def test_actionable_feedback_includes_hold_duration(self):
        for _ in range(4):
            record_trade_outcome("momentum", 5.0, hold_days=2)
        result = get_actionable_feedback()
        self.assertIn("2d", result)

    def test_actionable_feedback_sector_strong_edge(self):
        for _ in range(4):
            record_trade_outcome("momentum", 4.0, sector="Technology")
        result = get_actionable_feedback()
        self.assertIn("Technology", result)


class TestActionableFeedbackBranchGaps(unittest.TestCase):
    """Partial branch coverage gaps in get_actionable_feedback."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patcher = patch(
            "analysis.performance._STATS_PATH",
            os.path.join(self.tmpdir, "signal_stats.json"),
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_regime_with_only_one_trade_skipped_for_notes(self):
        """Line 246->245: regime with < 2 trades → inner if skipped, loop continues."""
        for _ in range(4):
            record_trade_outcome("momentum", 3.0, regime="BULL_TRENDING", confidence=8)
        record_trade_outcome("momentum", 2.0, regime="CHOPPY", confidence=8)
        result = get_actionable_feedback()
        self.assertIn("momentum", result)
        self.assertNotIn("CHOPPY strong", result)
        self.assertNotIn("CHOPPY avoid", result)

    def test_hold_bucket_with_fewer_than_3_trades_excluded(self):
        """Line 263->262: hold bucket with < 3 trades is skipped, others may still show."""
        for _ in range(4):
            record_trade_outcome("momentum", 3.0, hold_days=3, confidence=8)
        record_trade_outcome("momentum", 2.0, hold_days=1)
        result = get_actionable_feedback()
        self.assertIn("momentum", result)
        self.assertNotIn("1d", result)

    def test_no_hold_duration_block_when_all_buckets_below_threshold(self):
        """Lines 268->273: hold_lines empty (each hold bucket has <3 trades) → breakdown absent."""
        for hd in [1, 2, 3, 4]:
            record_trade_outcome("breakout", 3.0, hold_days=hd, confidence=8)
        result = get_actionable_feedback()
        self.assertIn("breakout", result)
        self.assertNotIn("Hold duration breakdown", result)

    def test_sector_bucket_with_fewer_than_3_trades_excluded(self):
        """Line 274->273: sector with < 3 trades is skipped."""
        for _ in range(4):
            record_trade_outcome("momentum", 3.0, confidence=8)
        record_trade_outcome("momentum", 2.0, sector="Technology")
        result = get_actionable_feedback()
        self.assertIn("momentum", result)
        self.assertNotIn("Technology: strong edge", result)
        self.assertNotIn("Technology: drag", result)


class TestGenerateDashboardRunHelperNoPnl(unittest.TestCase):
    """Line 248: _run helper with no explicit pnl uses computed pnl = after - before."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dashboard_patcher = patch(
            "analysis.performance._DASHBOARD_PATH",
            os.path.join(self.tmpdir, "dashboard.html"),
        )
        self.stats_patcher = patch(
            "analysis.performance._STATS_PATH",
            os.path.join(self.tmpdir, "signal_stats.json"),
        )
        self.dashboard_patcher.start()
        self.stats_patcher.start()
        self.addCleanup(self.dashboard_patcher.stop)
        self.addCleanup(self.stats_patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def _run(self, date_str, before, after, pnl=None):
        if pnl is None:  # pragma: no branch
            pnl = after - before
        return {
            "date": date_str,
            "account_before": {"portfolio_value": before},
            "account_after": {"portfolio_value": after},
            "daily_pnl": pnl,
            "trades_executed": [],
        }

    def test_run_without_pnl_computes_difference(self):
        record = self._run("2026-01-02", 100_000, 101_500)
        self.assertEqual(record["daily_pnl"], 1_500)
        generate_dashboard([record])
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "dashboard.html")))


class TestAggregateDimension(unittest.TestCase):
    def test_skips_bucket_with_zero_trades(self):
        # Line 157: `continue` when bucket.get("trades", 0) == 0
        stats = {
            "momentum": {
                "by_sector": {
                    "Technology": {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "total_return_pct": 0.0,
                    },
                    "Financials": {
                        "trades": 2,
                        "wins": 1,
                        "losses": 1,
                        "total_return_pct": 3.0,
                    },
                }
            }
        }
        result = _aggregate_dimension(stats, "by_sector")
        self.assertNotIn("Technology", result)
        self.assertIn("Financials", result)


# ── Helpers for trades-table tests ───────────────────────────────────────────


def _make_trades_table(rows: list[tuple]):
    """Return an in-memory SQLite connection seeded with test trade rows."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE trades ("
        "id INTEGER PRIMARY KEY, date_closed TEXT, symbol TEXT, signal TEXT, "
        "entry_regime TEXT, entry_date TEXT, entry_price REAL, exit_price REAL, "
        "pnl_pct REAL, days_held INTEGER, confidence INTEGER, sector TEXT, "
        "exit_reason TEXT, source TEXT DEFAULT 'live')"
    )
    conn.executemany(
        "INSERT INTO trades (date_closed, symbol, signal, entry_regime, sector, days_held, pnl_pct, source) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return conn


@contextlib.contextmanager
def _fake_db(rows: list[tuple]):
    conn = _make_trades_table(rows)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _fake_get_db(rows: list[tuple]):
    return lambda: _fake_db(rows)


# ── record_trade_outcome — trades table write ─────────────────────────────────


class TestRecordTradeOutcomeWithSymbol(unittest.TestCase):
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

    def test_with_symbol_inserts_to_trades_table(self):
        """When symbol is provided, a row is inserted into the trades table."""
        mock_conn = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_cm.__exit__ = MagicMock(return_value=False)
        with patch("utils.db.get_db", return_value=mock_cm):
            record_trade_outcome(
                "momentum",
                3.5,
                regime="BULL_TRENDING",
                confidence=8,
                sector="Technology",
                hold_days=2,
                symbol="AAPL",
                entry_date="2026-05-10",
                entry_price=180.0,
                exit_reason="ai_sell",
            )
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        self.assertIn("INSERT INTO trades", call_args[0])
        self.assertIn("AAPL", call_args[1])

    def test_with_date_closed_uses_provided_date(self):
        """date_closed param overrides today's date in the insert."""
        mock_conn = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_cm.__exit__ = MagicMock(return_value=False)
        with patch("utils.db.get_db", return_value=mock_cm):
            record_trade_outcome("momentum", 2.0, symbol="MSFT", date_closed="2026-05-01")
        row = mock_conn.execute.call_args[0][1]
        self.assertEqual(row[0], "2026-05-01")

    def test_db_exception_is_swallowed(self):
        """DB failure logs a warning but does not crash record_trade_outcome."""
        with patch("utils.db.get_db", side_effect=RuntimeError("db down")):
            try:
                record_trade_outcome("momentum", 2.0, symbol="TSLA")
            except Exception as exc:  # pragma: no cover
                self.fail(f"record_trade_outcome raised unexpectedly: {exc}")

    def test_without_symbol_skips_db_write(self):
        """No symbol → trades table insert is never attempted."""
        with patch("utils.db.get_db") as mock_get_db:
            record_trade_outcome("momentum", 2.0)
        mock_get_db.assert_not_called()


# ── get_attribution ───────────────────────────────────────────────────────────


class TestGetAttribution(unittest.TestCase):
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

    def test_aggregates_by_signal(self):
        """Rows with two signals → by_signal has both keys."""
        rows = [
            ("2026-05-10", "AAPL", "momentum", "BULL_TRENDING", "Technology", 2, 3.5, "live"),
            ("2026-05-11", "MSFT", "momentum", "BULL_TRENDING", "Technology", 1, -1.0, "live"),
            ("2026-05-11", "NVDA", "bb_squeeze", "CHOPPY", "Technology", 3, 2.0, "live"),
        ]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=None)
        self.assertIn("momentum", result["by_signal"])
        self.assertIn("bb_squeeze", result["by_signal"])
        self.assertEqual(result["by_signal"]["momentum"]["trades"], 2)

    def test_win_rate_computed_correctly(self):
        """1 win + 1 loss = 50% win rate."""
        rows = [
            ("2026-05-10", "AAPL", "momentum", "BULL_TRENDING", "Technology", 2, 5.0, "live"),
            ("2026-05-11", "MSFT", "momentum", "BULL_TRENDING", "Technology", 2, -2.0, "live"),
        ]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=None)
        self.assertEqual(result["by_signal"]["momentum"]["win_rate"], 50.0)

    def test_ranked_best_signal_first(self):
        """Signal with higher avg_return_pct ranks first."""
        rows = [
            ("2026-05-10", "AAPL", "momentum", "BULL", "Tech", 2, 10.0, "live"),
            ("2026-05-10", "MSFT", "bb_squeeze", "BULL", "Tech", 2, 1.0, "live"),
        ]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=None)
        signals = list(result["by_signal"].keys())
        self.assertEqual(signals[0], "momentum")

    def test_days_none_does_not_add_date_filter(self):
        """days=None → all rows in the live source are returned."""
        rows = [
            ("2020-01-01", "AAPL", "momentum", "BULL", "Tech", 2, 3.0, "live"),
        ]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=None)
        self.assertEqual(result["total_trades"], 1)

    def test_days_filter_excludes_old_rows(self):
        """days=1 → only rows from today are included (old rows excluded)."""
        rows = [
            ("2020-01-01", "AAPL", "momentum", "BULL", "Tech", 2, 3.0, "live"),
        ]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=1)
        # Old row excluded → empty → fallback to signal_stats.json (also empty)
        self.assertEqual(result, {})

    def test_empty_rows_falls_back_to_attribution_summary(self):
        """Empty trades table falls back to get_attribution_summary() (signal_stats.json)."""
        with patch("utils.db.get_db", _fake_get_db([])):
            result = get_attribution(days=90)
        # signal_stats.json is empty in tmpdir → returns {}
        self.assertEqual(result, {})

    def test_db_exception_falls_back_to_attribution_summary(self):
        """DB failure falls back to get_attribution_summary()."""
        with patch("utils.db.get_db", side_effect=RuntimeError("db error")):
            result = get_attribution(days=90)
        # signal_stats.json empty → returns {}
        self.assertEqual(result, {})

    def test_total_trades_matches_row_count(self):
        rows = [
            ("2026-05-10", "AAPL", "momentum", "BULL", "Tech", 2, 3.0, "live"),
            ("2026-05-11", "MSFT", "momentum", "BULL", "Tech", 1, 1.0, "live"),
        ]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=None)
        self.assertEqual(result["total_trades"], 2)

    def test_best_and_worst_signal_populated(self):
        rows = [
            ("2026-05-10", "AAPL", "momentum", "BULL", "Tech", 2, 10.0, "live"),
            ("2026-05-10", "MSFT", "bb_squeeze", "BULL", "Tech", 2, -2.0, "live"),
        ]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=None)
        self.assertEqual(result["best_signal"], "momentum")
        self.assertEqual(result["worst_signal"], "bb_squeeze")

    def test_null_signal_maps_to_unknown(self):
        """Rows with NULL signal are bucketed as 'unknown'."""
        rows = [("2026-05-10", "AAPL", None, "BULL", "Tech", 2, 3.0, "live")]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=None)
        self.assertIn("unknown", result["by_signal"])

    def test_period_days_included_in_result(self):
        rows = [("2026-05-10", "AAPL", "momentum", "BULL", "Tech", 2, 3.0, "live")]
        with patch("utils.db.get_db", _fake_get_db(rows)):
            result = get_attribution(days=30)
        self.assertEqual(result["period_days"], 30)


# ── _attribution_table_html ───────────────────────────────────────────────────


class TestAttributionTableHtml(unittest.TestCase):
    def test_empty_rows_returns_empty_string(self):
        self.assertEqual(_attribution_table_html("By Signal", {}), "")

    def test_populated_returns_html(self):
        rows = {"momentum": {"trades": 3, "win_rate": 66.7, "avg_return_pct": 2.5}}
        html = _attribution_table_html("By Signal", rows)
        self.assertIn("momentum", html)
        self.assertIn("By Signal", html)
        self.assertIn("+2.50%", html)

    def test_positive_avg_uses_green_colour(self):
        rows = {"momentum": {"trades": 1, "win_rate": 100.0, "avg_return_pct": 1.0}}
        html = _attribution_table_html("By Signal", rows)
        self.assertIn("#2e7d32", html)

    def test_negative_avg_uses_red_colour(self):
        rows = {"bb_squeeze": {"trades": 1, "win_rate": 0.0, "avg_return_pct": -1.5}}
        html = _attribution_table_html("By Signal", rows)
        self.assertIn("#c62828", html)


# ── generate_dashboard attribution section ────────────────────────────────────


class TestGenerateDashboardAttribution(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dashboard_patcher = patch(
            "analysis.performance._DASHBOARD_PATH",
            os.path.join(self.tmpdir, "dashboard.html"),
        )
        self.stats_patcher = patch(
            "analysis.performance._STATS_PATH",
            os.path.join(self.tmpdir, "signal_stats.json"),
        )
        self.dashboard_patcher.start()
        self.stats_patcher.start()
        self.addCleanup(self.dashboard_patcher.stop)
        self.addCleanup(self.stats_patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def _run(self, date_str, before, after, pnl=None):
        if pnl is None:  # pragma: no cover
            pnl = after - before
        return {
            "date": date_str,
            "account_before": {"portfolio_value": before},
            "account_after": {"portfolio_value": after},
            "daily_pnl": pnl,
            "trades_executed": [],
        }

    def test_dashboard_with_attribution_data_renders_section(self):
        """When get_attribution returns data, the HTML includes the attribution heading."""
        records = [self._run("2026-05-10", 100_000, 101_000, pnl=1_000)]
        attribution_data = {
            "by_signal": {"momentum": {"trades": 2, "win_rate": 100.0, "avg_return_pct": 3.0}},
            "by_regime": {},
            "by_sector": {},
            "by_hold_days": {},
            "total_trades": 2,
            "period_days": 90,
        }
        with patch("analysis.performance.get_attribution", return_value=attribution_data):
            generate_dashboard(records)
        with open(os.path.join(self.tmpdir, "dashboard.html")) as f:
            html = f.read()
        self.assertIn("Performance Attribution", html)
        self.assertIn("momentum", html)

    def test_dashboard_with_empty_attribution_no_section(self):
        """When get_attribution returns {}, no attribution section is added."""
        records = [self._run("2026-05-10", 100_000, 101_000, pnl=1_000)]
        with patch("analysis.performance.get_attribution", return_value={}):
            generate_dashboard(records)
        with open(os.path.join(self.tmpdir, "dashboard.html")) as f:
            html = f.read()
        self.assertNotIn("Performance Attribution", html)

    def test_dashboard_all_empty_breakdown_dicts_skips_section(self):
        """When attribution has total_trades>0 but all breakdown dicts are empty, no section."""
        records = [self._run("2026-05-10", 100_000, 101_000, pnl=1_000)]
        attribution_data = {
            "by_signal": {},
            "by_regime": {},
            "by_sector": {},
            "by_hold_days": {},
            "total_trades": 5,
            "period_days": 90,
        }
        with patch("analysis.performance.get_attribution", return_value=attribution_data):
            generate_dashboard(records)
        with open(os.path.join(self.tmpdir, "dashboard.html")) as f:
            html = f.read()
        self.assertNotIn("Performance Attribution", html)

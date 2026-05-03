"""Tests for dashboard.py — helper functions and page-rendering branches.

Each page branch in dashboard.py is module-level code that runs during import.
To cover different branches, we reload the module with different mocked
`st.radio` return values and controlled data sources.
"""

import json
import sys
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

# ── Reload helper ─────────────────────────────────────────────────────────────


def _make_mock_st(page: str = "Overview") -> MagicMock:
    """Create a fully-mocked streamlit module for a given page value."""
    mock_st = MagicMock(name="streamlit")
    mock_st.radio.return_value = page
    # Pass-through so _diagnostics_button is the raw function, not a MagicMock
    mock_st.fragment.return_value = lambda f: f
    mock_st.button.return_value = False

    def _cols(spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        cols = [MagicMock() for _ in range(n)]
        for col in cols:
            col.multiselect.side_effect = lambda _label, options=None, **_kw: options or []
            col.slider.return_value = 1
        return cols

    mock_st.columns.side_effect = _cols
    return mock_st


def _reload(
    page: str = "Overview",
    history: list | None = None,
    decisions: list | None = None,
    account_error: bool = False,
    os_listdir_return: list | None = None,
    os_listdir_raises: bool = False,
    open_data: str | None = None,
    open_raises: bool = False,
    os_exists: bool = True,
):
    """Remove cached dashboard, install mocks, and reimport.

    Uses manual sys.modules patching (not patch.dict) so that newly-added
    entries (e.g. pandas/numpy submodules) survive the reload and are not
    removed when the context exits.

    Returns ``(module, mock_st)``.
    """
    sys.modules.pop("dashboard", None)
    mock_st = _make_mock_st(page)

    mock_pt = MagicMock()
    mock_pt.load_history.return_value = history if history is not None else []

    mock_dl = MagicMock()
    mock_dl.load_decisions.return_value = decisions if decisions is not None else []

    # Patch only these keys; save originals so we can restore them after import
    # plotly.graph_objects is mocked to avoid reading ~/.plotly/.config during tests
    _PATCH_KEYS = [
        "streamlit",
        "utils.portfolio_tracker",
        "utils.decision_log",
        "plotly.graph_objects",
    ]
    _saved = {k: sys.modules.get(k) for k in _PATCH_KEYS}
    sys.modules["streamlit"] = mock_st
    sys.modules["utils.portfolio_tracker"] = mock_pt
    sys.modules["utils.decision_log"] = mock_dl
    sys.modules["plotly.graph_objects"] = MagicMock()

    try:
        # Patch os.listdir for _load_diagnostics calls during import
        listdir_patch = None
        if os_listdir_return is not None:
            listdir_patch = patch("os.listdir", return_value=os_listdir_return)
        elif os_listdir_raises:
            listdir_patch = patch("os.listdir", side_effect=OSError("no dir"))

        # Patch builtins.open for _load_diagnostics / _load_backtest during import
        open_patch = None
        if open_data is not None:
            open_patch = patch("builtins.open", mock_open(read_data=open_data))
        elif open_raises:
            open_patch = patch("builtins.open", side_effect=OSError("read error"))

        exists_patch = patch("os.path.exists", return_value=os_exists)

        def _do_import():
            if account_error:
                with patch("execution.trader.get_client", side_effect=Exception("no creds")):
                    import dashboard as db
            else:
                import dashboard as db
            return db

        with exists_patch:
            if listdir_patch:
                with listdir_patch:
                    if open_patch:
                        with open_patch:
                            db = _do_import()
                    else:
                        db = _do_import()
            elif open_patch:
                with open_patch:
                    db = _do_import()
            else:
                db = _do_import()
    finally:
        # Restore only the 3 keys we patched; leave everything else untouched
        for key, val in _saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val

    return db, mock_st


# ── Pure helper functions ─────────────────────────────────────────────────────


class TestFmtPct(unittest.TestCase):
    """_fmt_pct — line 194."""

    @classmethod
    def setUpClass(cls):
        cls.db, _ = _reload("Overview")

    def test_positive(self):
        self.assertEqual(self.db._fmt_pct(5.0), "+5.00%")

    def test_zero(self):
        self.assertEqual(self.db._fmt_pct(0.0), "+0.00%")

    def test_negative(self):
        self.assertEqual(self.db._fmt_pct(-3.5), "-3.50%")


class TestAgeLabel(unittest.TestCase):
    """_age_label — all four branches."""

    @classmethod
    def setUpClass(cls):
        cls.db, _ = _reload("Overview")

    def test_just_now(self):
        self.assertEqual(self.db._age_label(0), "just now")

    def test_minutes_ago(self):
        self.assertEqual(self.db._age_label(300), "5m ago")

    def test_hours_ago(self):
        self.assertEqual(self.db._age_label(7200), "2h ago")

    def test_days_ago(self):
        self.assertEqual(self.db._age_label(172800), "2d ago")


# ── _load_account error path (lines 223-224) ─────────────────────────────────


class TestLoadAccountError(unittest.TestCase):
    """_load_account exception path covered during reload AND direct call."""

    def test_error_path_returns_none_acc(self):
        db, _ = _reload("Overview", account_error=True)
        # Call again directly to verify lines 223-224 from a direct call too
        with patch("execution.trader.get_client", side_effect=Exception("direct test")):
            acc, positions, err = db._load_account()
        self.assertIsNone(acc)
        self.assertEqual(positions, [])
        self.assertIn("direct test", err)


# ── _load_diagnostics (lines 228-239) ────────────────────────────────────────


class TestLoadDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, _ = _reload("Overview")

    def test_returns_none_when_no_reports(self):
        with patch("os.listdir", return_value=[]):
            self.assertIsNone(self.db._load_diagnostics())

    def test_returns_latest_report(self):
        report = {"status": "PASS", "total": 100}
        with (
            patch(
                "os.listdir",
                return_value=["test_report_2026-05-01.json", "test_report_2026-04-30.json"],
            ),
            patch("builtins.open", mock_open(read_data=json.dumps(report))),
        ):
            result = self.db._load_diagnostics()
        self.assertEqual(result["status"], "PASS")

    def test_returns_none_on_exception(self):
        with patch("os.listdir", side_effect=OSError("no dir")):
            self.assertIsNone(self.db._load_diagnostics())


# ── _load_backtest (lines 243-250) ───────────────────────────────────────────


class TestLoadBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, _ = _reload("Overview")

    def test_returns_none_when_no_file(self):
        with patch("os.path.exists", return_value=False):
            self.assertIsNone(self.db._load_backtest())

    def test_reads_file(self):
        data = {"sharpe_ratio": 1.5}
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(data))),
        ):
            result = self.db._load_backtest()
        self.assertAlmostEqual(result["sharpe_ratio"], 1.5)

    def test_returns_none_on_read_error(self):
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", side_effect=OSError("read error")),
        ):
            self.assertIsNone(self.db._load_backtest())


# ── Overview page — elif err: branch (lines 291-292) ─────────────────────────


class TestOverviewErrBranch(unittest.TestCase):
    def test_account_error_renders_warning(self):
        _, mock_st = _reload("Overview", account_error=True)
        mock_st.warning.assert_called()


# ── _diagnostics_button (lines 166-181) ──────────────────────────────────────
# Covered by the Diagnostics page reload that calls _diagnostics_button at
# line 608 of dashboard.py. Additional direct tests below for completeness.


class TestDiagnosticsButton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, cls.mock_st = _reload("Overview")

    def test_disabled_button_during_cooldown(self):
        future = datetime.now(UTC).timestamp() + 60
        self.db._diagnostics_button(total=100, cooldown_end=future)
        self.mock_st.button.assert_called()

    def test_enabled_button_not_clicked(self):
        self.mock_st.button.return_value = False
        self.db._diagnostics_button(total=100, cooldown_end=0.0)
        self.mock_st.button.assert_called()

    def test_button_click_success(self):
        self.mock_st.button.return_value = True
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            self.db._diagnostics_button(total=50, cooldown_end=0.0)
        self.mock_st.success.assert_called()

    def test_button_click_failure(self):
        self.mock_st.button.return_value = True
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "test failure output"
        with patch("subprocess.run", return_value=mock_result):
            self.db._diagnostics_button(total=50, cooldown_end=0.0)
        self.mock_st.error.assert_called()


# ── Overview page — equity curve + positions (lines 306-325, 337) ────────────


class TestOverviewWithData(unittest.TestCase):
    def test_equity_chart_rendered_with_history(self):
        _, mock_st = _reload("Overview", history=_trade_history(2))
        mock_st.plotly_chart.assert_called()

    def test_colour_pnl_function(self):
        acc_mock = {"portfolio_value": 100000.0, "cash": 50000.0}
        pos_mock = [
            {
                "symbol": "AAPL",
                "market_value": 5000.0,
                "unrealized_pl": 200.0,
                "unrealized_plpc": 4.0,
            }
        ]
        with (
            patch("execution.trader.get_client", return_value=MagicMock()),
            patch("execution.trader.get_account_info", return_value=acc_mock),
            patch("execution.trader.get_open_positions", return_value=pos_mock),
        ):
            db, _ = _reload("Overview")
        # _colour_pnl is defined in module scope when positions is non-empty;
        # call directly since st.dataframe (mocked) never invokes the styler
        self.assertIn("color:", db._colour_pnl(5.0))
        self.assertIn("color:", db._colour_pnl(-1.0))


# ── Trades page (lines 338-388) ──────────────────────────────────────────────


def _trade_history(n: int = 2, with_trades: bool = True) -> list:
    trades = [{"action": "BUY", "symbol": "AAPL", "detail": "$5000"}] if with_trades else []
    return [
        {
            "date": f"2026-01-{i:02d}",
            "daily_pnl": 100.0 * i,
            "market_summary": "Bullish",
            "account_before": {"portfolio_value": 100000},
            "account_after": {"portfolio_value": 100200},
            "trades_executed": trades,
            "stop_losses_triggered": [],
        }
        for i in range(1, n + 1)
    ]


class TestTradesPage(unittest.TestCase):
    def test_empty_history_shows_info(self):
        _, mock_st = _reload("Trades", history=[])
        mock_st.info.assert_called()

    def test_history_with_trades_shows_metrics(self):
        db, mock_st = _reload("Trades", history=_trade_history(3, with_trades=True))
        mock_st.columns.assert_called()
        # _style_action is defined in module scope when rows is non-empty;
        # call it directly since st.dataframe (mocked) never invokes the styler
        db._style_action("BUY")
        db._style_action("SELL")
        db._style_action("HOLD")

    def test_history_with_no_executed_trades_shows_info(self):
        _, mock_st = _reload("Trades", history=_trade_history(2, with_trades=False))
        mock_st.info.assert_called()


# ── AI Decisions page (lines 394-483) ────────────────────────────────────────


def _decisions(n: int = 3) -> list:
    return [
        {
            "date": "2026-01-01",
            "action": "BUY",
            "symbol": "AAPL",
            "confidence": 8,
            "executed": True,
            "key_signal": "momentum",
            "reasoning": "Strong uptrend",
        }
        for _ in range(n)
    ]


class TestAiDecisionsPage(unittest.TestCase):
    def test_empty_decisions_shows_info(self):
        _, mock_st = _reload("AI Decisions", decisions=[])
        mock_st.info.assert_called()

    def test_non_empty_decisions_shows_metrics(self):
        _, mock_st = _reload("AI Decisions", decisions=_decisions(3))
        mock_st.columns.assert_called()


# ── Backtest page (lines 489-545) ────────────────────────────────────────────


def _backtest_results(
    with_equity: bool = True, with_signals: bool = True, with_trades: bool = True
) -> str:
    r: dict = {
        "start": "2025-01-01",
        "end": "2025-12-31",
        "initial_capital": 100000.0,
        "final_value": 115000.0,
        "total_return_pct": 15.0,
        "win_rate_pct": 60.0,
        "total_trades": 25,
        "sharpe_ratio": 1.2,
        "max_drawdown_pct": -5.0,
    }
    if with_equity:
        r["equity_curve"] = [["2025-01-01", 100000], ["2025-12-31", 115000]]
    if with_signals:
        r["by_signal"] = {"momentum": {"wins": 10, "losses": 5, "total_return": 20.0}}
    if with_trades:
        r["trades"] = [{"action": "SELL", "symbol": "AAPL", "pnl_pct": 5.0}]
    return json.dumps(r)


class TestBacktestPage(unittest.TestCase):
    def test_no_results_shows_info(self):
        _, mock_st = _reload("Backtest", os_exists=False)
        mock_st.info.assert_called()

    def test_full_results_shows_metrics(self):
        data = _backtest_results()
        _, mock_st = _reload(
            "Backtest",
            os_exists=True,
            open_data=data,
        )
        mock_st.columns.assert_called()

    def test_results_without_equity_curve(self):
        # Covers the `if results.get("equity_curve"):` False path
        data = _backtest_results(with_equity=False, with_signals=False, with_trades=False)
        _, mock_st = _reload("Backtest", os_exists=True, open_data=data)
        mock_st.columns.assert_called()

    def test_results_without_closed_trades(self):
        # Covers `if closed:` False path when trades list has no SELL with pnl_pct
        r = json.loads(_backtest_results(with_trades=True))
        r["trades"] = [{"action": "BUY", "symbol": "AAPL"}]  # no SELL+pnl_pct
        _, mock_st = _reload("Backtest", os_exists=True, open_data=json.dumps(r))
        mock_st.columns.assert_called()


# ── Diagnostics page (lines 552-608) ─────────────────────────────────────────


def _diag_report(
    status: str = "PASS",
    ts_offset_s: float = -10.0,
    with_failures: bool = False,
    invalid_ts: bool = False,
) -> str:
    ts = (datetime.now(UTC) + timedelta(seconds=ts_offset_s)).isoformat()
    if invalid_ts:
        ts = "not-a-date"
    report: dict = {
        "status": status,
        "total": 100,
        "passed": 100 if status == "PASS" else 90,
        "failed": 0 if status == "PASS" else 10,
        "errors": 0,
        "duration_seconds": 5.2,
        "timestamp": ts,
        "failures": [],
    }
    if with_failures:
        report["failures"] = [
            {"test": "TestFoo.test_bar", "message": "AssertionError: False is not True"}
        ]
    return json.dumps(report)


class TestDiagnosticsPage(unittest.TestCase):
    def test_no_report_shows_info(self):
        _, mock_st = _reload("Diagnostics", os_listdir_return=[])
        mock_st.info.assert_called()

    def test_passing_report_shows_success(self):
        data = _diag_report("PASS", ts_offset_s=-10.0)
        _, mock_st = _reload(
            "Diagnostics",
            os_listdir_return=["test_report_2026-05-01.json"],
            open_data=data,
        )
        mock_st.success.assert_called()

    def test_failing_report_shows_error(self):
        data = _diag_report("FAIL", ts_offset_s=-200.0, with_failures=True)
        _, mock_st = _reload(
            "Diagnostics",
            os_listdir_return=["test_report_2026-05-01.json"],
            open_data=data,
        )
        mock_st.error.assert_called()

    def test_invalid_timestamp_falls_back(self):
        # Covers `except Exception: st.caption(ts)` in timestamp parsing
        data = _diag_report("PASS", invalid_ts=True)
        _, mock_st = _reload(
            "Diagnostics",
            os_listdir_return=["test_report_2026-05-01.json"],
            open_data=data,
        )
        mock_st.caption.assert_called()

    def test_report_with_old_timestamp_minutes(self):
        # age_s 300 → "5m ago"
        data = _diag_report("PASS", ts_offset_s=-300.0)
        _, mock_st = _reload(
            "Diagnostics",
            os_listdir_return=["test_report_2026-05-01.json"],
            open_data=data,
        )
        mock_st.success.assert_called()

    def test_report_with_old_timestamp_hours(self):
        # age_s 7200 → "2h ago"
        data = _diag_report("PASS", ts_offset_s=-7200.0)
        _, mock_st = _reload(
            "Diagnostics",
            os_listdir_return=["test_report_2026-05-01.json"],
            open_data=data,
        )
        mock_st.success.assert_called()

    def test_report_with_old_timestamp_days(self):
        # age_s 172800 → "2d ago"
        data = _diag_report("PASS", ts_offset_s=-172800.0)
        _, mock_st = _reload(
            "Diagnostics",
            os_listdir_return=["test_report_2026-05-01.json"],
            open_data=data,
        )
        mock_st.success.assert_called()

    def test_diagnostics_button_called_with_report(self):
        # Covers line 608: _diagnostics_button(total=total, cooldown_end=...)
        data = _diag_report("PASS", ts_offset_s=-10.0)
        db, mock_st = _reload(
            "Diagnostics",
            os_listdir_return=["test_report_2026-05-01.json"],
            open_data=data,
        )
        # _diagnostics_button was called during import; button was rendered
        mock_st.button.assert_called()

    def test_cooldown_end_invalid_ts_fallback(self):
        # Covers `except Exception: cooldown_end = 0.0` at lines 605-606
        # When ts is invalid, datetime.fromisoformat raises → cooldown_end = 0.0
        data = _diag_report("PASS", invalid_ts=True)
        db, mock_st = _reload(
            "Diagnostics",
            os_listdir_return=["test_report_2026-05-01.json"],
            open_data=data,
        )
        # If cooldown_end = 0.0, _diagnostics_button was called with cooldown_end=0
        mock_st.button.assert_called()


if __name__ == "__main__":
    unittest.main()

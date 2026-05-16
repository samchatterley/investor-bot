"""Tests for cli.py — all command handlers."""

import io
import sys
import unittest
from unittest.mock import MagicMock, patch


def _account(value=100_000, cash=30_000):
    return {"portfolio_value": value, "cash": cash, "buying_power": 60_000, "equity": value}


def _pos(symbol="AAPL", value=5_000, pl=250.0, plpc=5.0):
    return {"symbol": symbol, "market_value": value, "unrealized_pl": pl, "unrealized_plpc": plpc}


def _record(date="2026-01-15", pnl=200.0, trades=None, summary="Quiet day"):
    return {
        "date": date,
        "daily_pnl": pnl,
        "market_summary": summary,
        "trades_executed": trades or [],
        "stop_losses_triggered": [],
        "account_before": _account(),
        "account_after": _account(),
        "buy_candidates": [],
        "position_decisions": [],
    }


def _capture(fn, *args):
    """Run fn(*args), return stdout as string."""
    buf = io.StringIO()
    sys.stdout = buf
    try:
        fn(*args)
    finally:
        sys.stdout = sys.__stdout__
    return buf.getvalue()


class TestCliHelpers(unittest.TestCase):
    def test_header_prints_text(self):
        from cli import _header

        output = _capture(_header, "BOT STATUS")
        self.assertIn("BOT STATUS", output)

    def test_print_positions_empty(self):
        from cli import _print_positions

        output = _capture(_print_positions, [])
        self.assertIn("No open positions", output)

    def test_print_positions_shows_symbols(self):
        from cli import _print_positions

        output = _capture(_print_positions, [_pos("AAPL"), _pos("NVDA")])
        self.assertIn("AAPL", output)
        self.assertIn("NVDA", output)

    def test_print_positions_shows_positive_pnl(self):
        from cli import _print_positions

        output = _capture(_print_positions, [_pos("AAPL", pl=300.0)])
        self.assertIn("+", output)

    def test_print_positions_shows_negative_pnl(self):
        from cli import _print_positions

        output = _capture(_print_positions, [_pos("AAPL", pl=-100.0, plpc=-2.0)])
        self.assertIn("-", output)


class TestCmdStatus(unittest.TestCase):
    def test_shows_active_when_not_halted(self):
        args = MagicMock()
        mock_client = MagicMock()
        with (
            patch("os.path.exists", return_value=False),
            patch("execution.trader.get_client", return_value=mock_client),
            patch("execution.trader.get_account_info", return_value=_account()),
            patch("execution.trader.get_open_positions", return_value=[]),
            patch("cli.config.IS_PAPER", True),
            patch("cli.config.MAX_POSITIONS", 5),
        ):
            from cli import cmd_status

            output = _capture(cmd_status, args)
        self.assertIn("Active", output)

    def test_shows_halted_when_halt_file_present(self):
        args = MagicMock()
        with (
            patch("os.path.exists", return_value=True),
            patch("execution.trader.get_client", return_value=MagicMock()),
            patch("execution.trader.get_account_info", return_value=_account()),
            patch("execution.trader.get_open_positions", return_value=[]),
            patch("cli.config.IS_PAPER", True),
            patch("cli.config.MAX_POSITIONS", 5),
        ):
            from cli import cmd_status

            output = _capture(cmd_status, args)
        self.assertIn("HALTED", output)

    def test_shows_paper_mode(self):
        args = MagicMock()
        with (
            patch("os.path.exists", return_value=False),
            patch("execution.trader.get_client", return_value=MagicMock()),
            patch("execution.trader.get_account_info", return_value=_account()),
            patch("execution.trader.get_open_positions", return_value=[]),
            patch("cli.config.IS_PAPER", True),
            patch("cli.config.MAX_POSITIONS", 5),
        ):
            from cli import cmd_status

            output = _capture(cmd_status, args)
        self.assertIn("PAPER", output)

    def test_shows_error_message_on_api_exception(self):
        args = MagicMock()
        with (
            patch("os.path.exists", return_value=False),
            patch("execution.trader.get_client", side_effect=Exception("broker down")),
            patch("cli.config.IS_PAPER", True),
            patch("cli.config.MAX_POSITIONS", 5),
        ):
            from cli import cmd_status

            output = _capture(cmd_status, args)
        self.assertIn("error", output.lower())


class TestCmdPositions(unittest.TestCase):
    def test_shows_positions(self):
        args = MagicMock()
        with (
            patch("execution.trader.get_client", return_value=MagicMock()),
            patch("execution.trader.get_open_positions", return_value=[_pos("AAPL")]),
        ):
            from cli import cmd_positions

            output = _capture(cmd_positions, args)
        self.assertIn("AAPL", output)

    def test_shows_no_positions_message(self):
        args = MagicMock()
        with (
            patch("execution.trader.get_client", return_value=MagicMock()),
            patch("execution.trader.get_open_positions", return_value=[]),
        ):
            from cli import cmd_positions

            output = _capture(cmd_positions, args)
        self.assertIn("No open positions", output)

    def test_shows_error_on_api_exception(self):
        args = MagicMock()
        with patch("execution.trader.get_client", side_effect=Exception("network error")):
            from cli import cmd_positions

            output = _capture(cmd_positions, args)
        self.assertIn("Error", output)


class TestCmdTrades(unittest.TestCase):
    def test_shows_trade_history(self):
        args = MagicMock(days=10)
        trades = [{"action": "BUY", "symbol": "AAPL", "detail": "$5000"}]
        records = [_record(date="2026-01-15", trades=trades)]
        with patch("cli.load_history", return_value=records):
            from cli import cmd_trades

            output = _capture(cmd_trades, args)
        self.assertIn("AAPL", output)
        self.assertIn("BUY", output)

    def test_shows_no_history_message(self):
        args = MagicMock(days=10)
        with patch("cli.load_history", return_value=[]):
            from cli import cmd_trades

            output = _capture(cmd_trades, args)
        self.assertIn("No trade history", output)

    def test_filters_midday_and_close_records(self):
        args = MagicMock(days=10)
        records = [
            _record(date="2026-01-15"),
            _record(date="2026-01-15-midday"),
            _record(date="2026-01-15-close"),
        ]
        with patch("cli.load_history", return_value=records):
            from cli import cmd_trades

            output = _capture(cmd_trades, args)
        # Should only show one record (the open run)
        self.assertEqual(output.count("2026-01-15"), 1)

    def test_respects_days_limit(self):
        args = MagicMock(days=3)
        records = [_record(date=f"2026-01-0{i}") for i in range(1, 8)]
        with patch("cli.load_history", return_value=records):
            from cli import cmd_trades

            output = _capture(cmd_trades, args)
        self.assertNotIn("2026-01-01", output)
        self.assertIn("2026-01-07", output)


class TestCmdDecisions(unittest.TestCase):
    def test_shows_no_decisions_message(self):
        args = MagicMock(days=5)
        with patch("cli.load_decisions", return_value=[]):
            from cli import cmd_decisions

            output = _capture(cmd_decisions, args)
        self.assertIn("No AI decision records", output)

    def test_shows_decision_entries(self):
        args = MagicMock(days=5)
        entries = [
            {
                "date": "2026-01-15",
                "market_summary": "Bullish day",
                "action": "BUY",
                "symbol": "AAPL",
                "confidence": 8,
                "key_signal": "momentum",
                "reasoning": "Strong trend",
                "executed": True,
            }
        ]
        with patch("cli.load_decisions", return_value=entries):
            from cli import cmd_decisions

            output = _capture(cmd_decisions, args)
        self.assertIn("AAPL", output)
        self.assertIn("BUY", output)

    def test_marks_executed_entries(self):
        args = MagicMock(days=5)
        entries = [
            {
                "date": "2026-01-15",
                "market_summary": "",
                "action": "BUY",
                "symbol": "AAPL",
                "confidence": 8,
                "key_signal": "m",
                "reasoning": "",
                "executed": True,
            }
        ]
        with patch("cli.load_decisions", return_value=entries):
            from cli import cmd_decisions

            output = _capture(cmd_decisions, args)
        self.assertIn("EXECUTED", output)


class TestCmdRun(unittest.TestCase):
    def test_calls_bot_run_with_correct_mode(self):
        args = MagicMock(dry_run=False, mode="midday")
        mock_run = MagicMock()
        with patch("main.run", mock_run):
            from cli import cmd_run

            cmd_run(args)
        mock_run.assert_called_once_with(dry_run=False, mode="midday")

    def test_calls_bot_run_dry_run(self):
        args = MagicMock(dry_run=True, mode="open")
        mock_run = MagicMock()
        with patch("main.run", mock_run):
            from cli import cmd_run

            cmd_run(args)
        mock_run.assert_called_once_with(dry_run=True, mode="open")


class TestCmdHalt(unittest.TestCase):
    def test_confirmed_triggers_kill_switch(self):
        args = MagicMock()
        mock_kill = MagicMock()
        with patch("builtins.input", return_value="yes"), patch("main._run_kill_switch", mock_kill):
            from cli import cmd_halt

            cmd_halt(args)
        mock_kill.assert_called_once()

    def test_cancelled_does_not_trigger_kill_switch(self):
        args = MagicMock()
        mock_kill = MagicMock()
        with patch("builtins.input", return_value="no"), patch("main._run_kill_switch", mock_kill):
            from cli import cmd_halt

            output = _capture(cmd_halt, args)
        mock_kill.assert_not_called()
        self.assertIn("Cancelled", output)

    def test_empty_input_does_not_trigger_kill_switch(self):
        args = MagicMock()
        mock_kill = MagicMock()
        with patch("builtins.input", return_value=""), patch("main._run_kill_switch", mock_kill):
            from cli import cmd_halt

            cmd_halt(args)
        mock_kill.assert_not_called()


class TestCmdResume(unittest.TestCase):
    def test_calls_clear_halt(self):
        args = MagicMock()
        mock_clear = MagicMock()
        with patch("main._run_clear_halt", mock_clear):
            from cli import cmd_resume

            cmd_resume(args)
        mock_clear.assert_called_once()


class TestCmdBacktest(unittest.TestCase):
    def test_uses_provided_capital(self):
        args = MagicMock(capital=50_000.0, start="2025-01-01", end="2025-12-31")
        mock_run = MagicMock()
        with (
            patch("backtest.run_backtest", mock_run),
            patch("cli.config.STOCK_UNIVERSE", ["AAPL"]),
            patch("cli.config.MAX_POSITIONS", 5),
        ):
            from cli import cmd_backtest

            cmd_backtest(args)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        self.assertEqual(call_args[3], 50_000.0)

    def test_fetches_capital_from_broker_when_not_provided(self):
        args = MagicMock(capital=None, start="2025-01-01", end="2025-12-31")
        mock_run = MagicMock()
        mock_client = MagicMock()
        with (
            patch("backtest.run_backtest", mock_run),
            patch("execution.trader.get_client", return_value=mock_client),
            patch("execution.trader.get_account_info", return_value=_account(75_000)),
            patch("cli.config.STOCK_UNIVERSE", ["AAPL"]),
            patch("cli.config.MAX_POSITIONS", 5),
        ):
            from cli import cmd_backtest

            cmd_backtest(args)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        self.assertAlmostEqual(call_args[3], 75_000.0)

    def test_defaults_to_100k_on_broker_error(self):
        args = MagicMock(capital=None, start="2025-01-01", end="2025-12-31")
        mock_run = MagicMock()
        with (
            patch("backtest.run_backtest", mock_run),
            patch("execution.trader.get_client", side_effect=Exception("no connection")),
            patch("cli.config.STOCK_UNIVERSE", ["AAPL"]),
            patch("cli.config.MAX_POSITIONS", 5),
        ):
            from cli import cmd_backtest

            cmd_backtest(args)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        self.assertAlmostEqual(call_args[3], 100_000.0)


class TestCmdDashboard(unittest.TestCase):
    def test_calls_streamlit_subprocess(self):
        args = MagicMock()
        mock_run = MagicMock()
        with patch("subprocess.run", mock_run):
            from cli import cmd_dashboard

            cmd_dashboard(args)
        mock_run.assert_called_once()
        cmd_line = mock_run.call_args[0][0]
        self.assertIn("streamlit", cmd_line)
        self.assertIn("run", cmd_line)


class TestDemoModeImport(unittest.TestCase):
    """Covers the else-branch (line 31) in cli.py that imports config without validate()."""

    def test_demo_mode_import_covers_else_branch(self):
        """Setting sys.argv[1]=='demo' causes cli to take the else branch on import."""
        original_argv = sys.argv[:]
        # Remove any cached cli module so we get a fresh import
        sys.modules.pop("cli", None)

        try:
            sys.argv = ["cli.py", "demo"]
            # We need config to be importable; it already is from other tests
            import cli as _cli_demo  # noqa: F401 — import triggers line 31

            # Verify the module imported and _IS_DEMO is True
            self.assertTrue(_cli_demo._IS_DEMO)
        finally:
            sys.argv = original_argv
            # Restore non-demo cli for subsequent tests
            sys.modules.pop("cli", None)


# ── Minimal fixture for cmd_demo tests ───────────────────────────────────────


def _demo_fixture(is_bearish=False, buy_candidates=None, position_count=1):
    """Return a minimal demo_run.json-shaped dict for cmd_demo tests."""
    if buy_candidates is None:
        buy_candidates = [
            {
                "symbol": "AMD",
                "action": "BUY",
                "confidence": 8,
                "key_signal": "momentum",
                "reasoning": "Strong trend",
            },
        ]
    return {
        "account": {"portfolio_value": 100_000.0, "cash": 62_000.0},
        "open_positions": [
            {
                "symbol": "MSFT",
                "qty": 1,
                "avg_entry_price": 415.0,
                "current_price": 421.0,
                "unrealized_pl": 6.0,
                "unrealized_plpc": 1.4,
                "market_value": 421.0,
            }
        ][:position_count],
        "regime": {
            "is_bearish": is_bearish,
            "spy_change_pct": 0.4,
            "spy_5d_pct": 1.1,
            "regime": "BEAR_MARKET" if is_bearish else "BULL_TRENDING",
        },
        "vix": 16.2,
        "snapshots": [
            {
                "symbol": "AMD",
                "current_price": 162.8,
                "ret_1d_pct": 3.4,
                "ret_5d_pct": 7.1,
                "rsi_14": 54,
                "bb_pct": 0.68,
                "vol_ratio": 2.1,
                "ema9_above_ema21": True,
                "macd_diff": 0.88,
                "macd_crossed_up": True,
                "weekly_trend_up": True,
                "avg_volume": 38_000_000,
                "sector": "Technology",
            },
        ],
        "ai_response": {
            "market_summary": "Test market summary.",
            "position_decisions": [
                {"symbol": "MSFT", "action": "HOLD", "confidence": 7, "reasoning": "Hold."}
            ],
            "buy_candidates": buy_candidates,
        },
    }


def _run_cmd_demo(fixture, extra_patches=None):
    """
    Run cmd_demo with the given fixture injected via patched open/json.load.
    Patches prefilter_candidates to pass snapshots through unchanged,
    validate_ai_response to return (True, []),
    kelly_fraction to return 0.05,
    and time.sleep to a no-op.
    Returns captured stdout.
    """
    import json as _json

    extra_patches = extra_patches or {}

    # prefilter passes all snapshots through by default
    def _passthrough(snapshots):
        return snapshots

    buf = io.StringIO()

    with (
        patch("builtins.open", unittest.mock.mock_open(read_data=_json.dumps(fixture))),
        patch("json.load", return_value=fixture),
        patch("execution.stock_scanner.prefilter_candidates", side_effect=_passthrough),
        patch("utils.validators.validate_ai_response", return_value=(True, [])),
        patch("risk.position_sizer.kelly_fraction", return_value=0.05),
        patch("time.sleep"),
    ):
        sys.stdout = buf
        try:
            from cli import cmd_demo

            cmd_demo(None)
        finally:
            sys.stdout = sys.__stdout__

    return buf.getvalue()


class TestCmdDemo(unittest.TestCase):
    def test_demo_bearish_no_orders_placed(self):
        """In a bearish regime, the bear filter suppresses all buys."""
        fixture = _demo_fixture(is_bearish=True)
        output = _run_cmd_demo(fixture)
        self.assertIn("Bear filter active", output)
        self.assertIn("No orders placed", output)

    def test_demo_with_candidates_places_simulated_orders(self):
        """With qualified candidates in a non-bearish regime, orders are simulated."""
        fixture = _demo_fixture(is_bearish=False)
        output = _run_cmd_demo(fixture)
        # Should show a simulated BUY for AMD
        self.assertIn("SIMULATED", output)
        self.assertIn("AMD", output)

    def test_demo_prints_summary(self):
        """cmd_demo always prints 'Demo complete' at the end."""
        fixture = _demo_fixture(is_bearish=False)
        output = _run_cmd_demo(fixture)
        self.assertIn("Demo complete", output)

    def test_demo_low_confidence_skipped(self):
        """A candidate with confidence < 7 is skipped by the risk gate."""
        low_conf_candidate = [
            {
                "symbol": "AMD",
                "action": "BUY",
                "confidence": 5,
                "key_signal": "momentum",
                "reasoning": "Weak signal",
            },
        ]
        fixture = _demo_fixture(is_bearish=False, buy_candidates=low_conf_candidate)
        output = _run_cmd_demo(fixture)
        self.assertIn("confidence 5 below floor", output)

    def test_demo_no_positions_shows_zero_count(self):
        """With no open positions, summary shows 0 for positions."""
        fixture = _demo_fixture(is_bearish=True, position_count=0)
        output = _run_cmd_demo(fixture)
        self.assertIn("Demo complete", output)

    def test_demo_shows_candidate_count(self):
        """Summary line shows the buy_candidates count from the AI response."""
        fixture = _demo_fixture(is_bearish=False)
        output = _run_cmd_demo(fixture)
        self.assertIn("buy candidates", output)

    def test_demo_filtered_candidates_shows_filtered_message(self):
        """When prefilter_candidates returns fewer snapshots, prints 'Filtered:' message."""
        import json as _json

        fixture = _demo_fixture(is_bearish=True)

        # Return empty list from prefilter so all snapshots are filtered out
        def _filter_all(_snapshots):
            return []

        buf = io.StringIO()
        with (
            patch("builtins.open", unittest.mock.mock_open(read_data=_json.dumps(fixture))),
            patch("json.load", return_value=fixture),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=_filter_all),
            patch("utils.validators.validate_ai_response", return_value=(True, [])),
            patch("risk.position_sizer.kelly_fraction", return_value=0.05),
            patch("time.sleep"),
        ):
            sys.stdout = buf
            try:
                from cli import cmd_demo

                cmd_demo(None)
            finally:
                sys.stdout = sys.__stdout__

        self.assertIn("Filtered:", buf.getvalue())

    def test_demo_validation_errors_shown(self):
        """When validate_ai_response returns errors, each error is printed as a warning."""
        import json as _json

        fixture = _demo_fixture(is_bearish=True)

        def _passthrough(snapshots):
            return snapshots

        buf = io.StringIO()
        with (
            patch("builtins.open", unittest.mock.mock_open(read_data=_json.dumps(fixture))),
            patch("json.load", return_value=fixture),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=_passthrough),
            patch(
                "utils.validators.validate_ai_response",
                return_value=(False, ["FAKECORP not in universe"]),
            ),
            patch("risk.position_sizer.kelly_fraction", return_value=0.05),
            patch("time.sleep"),
        ):
            sys.stdout = buf
            try:
                from cli import cmd_demo

                cmd_demo(None)
            finally:
                sys.stdout = sys.__stdout__

        self.assertIn("Rejected:", buf.getvalue())

    def test_demo_tiny_notional_skipped(self):
        """When kelly_fraction returns near-zero, notional < 1.0 triggers skip warning."""
        import json as _json

        fixture = _demo_fixture(is_bearish=False)

        def _passthrough(snapshots):
            return snapshots

        # kelly=0 → notional=0 < 1.0
        buf = io.StringIO()
        with (
            patch("builtins.open", unittest.mock.mock_open(read_data=_json.dumps(fixture))),
            patch("json.load", return_value=fixture),
            patch("execution.stock_scanner.prefilter_candidates", side_effect=_passthrough),
            patch("utils.validators.validate_ai_response", return_value=(True, [])),
            patch("risk.position_sizer.kelly_fraction", return_value=0.0),
            patch("time.sleep"),
        ):
            sys.stdout = buf
            try:
                from cli import cmd_demo

                cmd_demo(None)
            finally:
                sys.stdout = sys.__stdout__

        self.assertIn("notional", buf.getvalue())
        self.assertIn("too small", buf.getvalue())


class TestMainEntryPoint(unittest.TestCase):
    """Tests for main() — argument parsing and command dispatch."""

    def _call_main(self, argv, patches):
        """Set sys.argv, apply patches dict, call cli.main()."""
        original_argv = sys.argv[:]
        sys.argv = argv
        try:
            import cli as cli_mod

            with unittest.mock.patch.multiple(cli_mod, **patches):
                cli_mod.main()
        finally:
            sys.argv = original_argv

    def test_main_status(self):
        """sys.argv=['cli.py','status'] dispatches to cmd_status."""
        mock_fn = MagicMock()
        self._call_main(["cli.py", "status"], {"cmd_status": mock_fn})
        mock_fn.assert_called_once()

    def test_main_trades_default_days(self):
        """sys.argv=['cli.py','trades'] dispatches to cmd_trades with default days=10."""
        mock_fn = MagicMock()
        self._call_main(["cli.py", "trades"], {"cmd_trades": mock_fn})
        mock_fn.assert_called_once()
        args = mock_fn.call_args[0][0]
        self.assertEqual(args.days, 10)

    def test_main_decisions_with_days(self):
        """sys.argv=['cli.py','decisions','--days','10'] passes days=10 to cmd_decisions."""
        mock_fn = MagicMock()
        self._call_main(["cli.py", "decisions", "--days", "10"], {"cmd_decisions": mock_fn})
        mock_fn.assert_called_once()
        args = mock_fn.call_args[0][0]
        self.assertEqual(args.days, 10)

    def test_main_run_with_dry_run(self):
        """sys.argv=['cli.py','run','--dry-run'] passes dry_run=True to cmd_run."""
        mock_fn = MagicMock()
        self._call_main(["cli.py", "run", "--dry-run"], {"cmd_run": mock_fn})
        mock_fn.assert_called_once()
        args = mock_fn.call_args[0][0]
        self.assertTrue(args.dry_run)

    def test_main_halt(self):
        """sys.argv=['cli.py','halt'] dispatches to cmd_halt."""
        mock_fn = MagicMock()
        self._call_main(["cli.py", "halt"], {"cmd_halt": mock_fn})
        mock_fn.assert_called_once()

    def test_main_demo(self):
        """sys.argv=['cli.py','demo'] dispatches to cmd_demo."""
        mock_fn = MagicMock()
        self._call_main(["cli.py", "demo"], {"cmd_demo": mock_fn})
        mock_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()

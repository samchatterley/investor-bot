"""Tests for cli.py — all command handlers."""
import io
import sys
import unittest
from unittest.mock import MagicMock, patch


def _account(value=100_000, cash=30_000):
    return {"portfolio_value": value, "cash": cash, "buying_power": 60_000, "equity": value}


def _pos(symbol="AAPL", value=5_000, pl=250.0, plpc=5.0):
    return {"symbol": symbol, "market_value": value,
            "unrealized_pl": pl, "unrealized_plpc": plpc}


def _record(date="2026-01-15", pnl=200.0, trades=None, summary="Quiet day"):
    return {
        "date": date, "daily_pnl": pnl,
        "market_summary": summary,
        "trades_executed": trades or [],
        "stop_losses_triggered": [],
        "account_before": _account(), "account_after": _account(),
        "buy_candidates": [], "position_decisions": [],
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

    def _run(self, halted=False, positions=None, api_error=False):
        args = MagicMock()
        mock_client = MagicMock()
        mock_acc = _account()
        mock_pos = positions or []

        with patch("cli.config.HALT_FILE", "/tmp/halt_test_file_not_real"), \
             patch("cli.config.IS_PAPER", True), \
             patch("cli.config.MAX_POSITIONS", 5):
            if halted:
                with patch("os.path.exists", return_value=True), \
                     patch("cli.trader.get_client", return_value=mock_client), \
                     patch("cli.trader.get_account_info", return_value=mock_acc), \
                     patch("cli.trader.get_open_positions", return_value=mock_pos):
                    from cli import cmd_status
                    output = _capture(cmd_status, args)
            elif api_error:
                with (
                    patch("os.path.exists", return_value=False),
                    patch("cli.trader", side_effect=Exception("API down")),
                ):
                    from cli import cmd_status
                    output = _capture(cmd_status, args)
            else:
                with patch("os.path.exists", return_value=False):
                    import cli as cli_mod
                    with patch.object(cli_mod, "_print_positions"), \
                         patch("execution.trader.get_client", return_value=mock_client), \
                         patch("execution.trader.get_account_info", return_value=mock_acc), \
                         patch("execution.trader.get_open_positions", return_value=mock_pos):
                        output = _capture(cmd_status, args)
        return output

    def test_shows_active_when_not_halted(self):
        args = MagicMock()
        mock_client = MagicMock()
        with patch("os.path.exists", return_value=False), \
             patch("execution.trader.get_client", return_value=mock_client), \
             patch("execution.trader.get_account_info", return_value=_account()), \
             patch("execution.trader.get_open_positions", return_value=[]), \
             patch("cli.config.IS_PAPER", True), \
             patch("cli.config.MAX_POSITIONS", 5):
            from cli import cmd_status
            output = _capture(cmd_status, args)
        self.assertIn("Active", output)

    def test_shows_halted_when_halt_file_present(self):
        args = MagicMock()
        with patch("os.path.exists", return_value=True), \
             patch("execution.trader.get_client", return_value=MagicMock()), \
             patch("execution.trader.get_account_info", return_value=_account()), \
             patch("execution.trader.get_open_positions", return_value=[]), \
             patch("cli.config.IS_PAPER", True), \
             patch("cli.config.MAX_POSITIONS", 5):
            from cli import cmd_status
            output = _capture(cmd_status, args)
        self.assertIn("HALTED", output)

    def test_shows_paper_mode(self):
        args = MagicMock()
        with patch("os.path.exists", return_value=False), \
             patch("execution.trader.get_client", return_value=MagicMock()), \
             patch("execution.trader.get_account_info", return_value=_account()), \
             patch("execution.trader.get_open_positions", return_value=[]), \
             patch("cli.config.IS_PAPER", True), \
             patch("cli.config.MAX_POSITIONS", 5):
            from cli import cmd_status
            output = _capture(cmd_status, args)
        self.assertIn("PAPER", output)

    def test_shows_error_message_on_api_exception(self):
        args = MagicMock()
        with patch("os.path.exists", return_value=False), \
             patch("execution.trader.get_client", side_effect=Exception("broker down")), \
             patch("cli.config.IS_PAPER", True), \
             patch("cli.config.MAX_POSITIONS", 5):
            from cli import cmd_status
            output = _capture(cmd_status, args)
        self.assertIn("error", output.lower())


class TestCmdPositions(unittest.TestCase):

    def test_shows_positions(self):
        args = MagicMock()
        with patch("execution.trader.get_client", return_value=MagicMock()), \
             patch("execution.trader.get_open_positions", return_value=[_pos("AAPL")]):
            from cli import cmd_positions
            output = _capture(cmd_positions, args)
        self.assertIn("AAPL", output)

    def test_shows_no_positions_message(self):
        args = MagicMock()
        with patch("execution.trader.get_client", return_value=MagicMock()), \
             patch("execution.trader.get_open_positions", return_value=[]):
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
        entries = [{
            "date": "2026-01-15", "market_summary": "Bullish day",
            "action": "BUY", "symbol": "AAPL", "confidence": 8,
            "key_signal": "momentum", "reasoning": "Strong trend", "executed": True,
        }]
        with patch("cli.load_decisions", return_value=entries):
            from cli import cmd_decisions
            output = _capture(cmd_decisions, args)
        self.assertIn("AAPL", output)
        self.assertIn("BUY", output)

    def test_marks_executed_entries(self):
        args = MagicMock(days=5)
        entries = [{
            "date": "2026-01-15", "market_summary": "",
            "action": "BUY", "symbol": "AAPL", "confidence": 8,
            "key_signal": "m", "reasoning": "", "executed": True,
        }]
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
        with patch("builtins.input", return_value="yes"), \
             patch("main._run_kill_switch", mock_kill):
            from cli import cmd_halt
            cmd_halt(args)
        mock_kill.assert_called_once()

    def test_cancelled_does_not_trigger_kill_switch(self):
        args = MagicMock()
        mock_kill = MagicMock()
        with patch("builtins.input", return_value="no"), \
             patch("main._run_kill_switch", mock_kill):
            from cli import cmd_halt
            output = _capture(cmd_halt, args)
        mock_kill.assert_not_called()
        self.assertIn("Cancelled", output)

    def test_empty_input_does_not_trigger_kill_switch(self):
        args = MagicMock()
        mock_kill = MagicMock()
        with patch("builtins.input", return_value=""), \
             patch("main._run_kill_switch", mock_kill):
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
        with patch("backtest.run_backtest", mock_run), \
             patch("cli.config.STOCK_UNIVERSE", ["AAPL"]), \
             patch("cli.config.MAX_POSITIONS", 5):
            from cli import cmd_backtest
            cmd_backtest(args)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        self.assertEqual(call_args[3], 50_000.0)

    def test_fetches_capital_from_broker_when_not_provided(self):
        args = MagicMock(capital=None, start="2025-01-01", end="2025-12-31")
        mock_run = MagicMock()
        mock_client = MagicMock()
        with patch("backtest.run_backtest", mock_run), \
             patch("execution.trader.get_client", return_value=mock_client), \
             patch("execution.trader.get_account_info", return_value=_account(75_000)), \
             patch("cli.config.STOCK_UNIVERSE", ["AAPL"]), \
             patch("cli.config.MAX_POSITIONS", 5):
            from cli import cmd_backtest
            cmd_backtest(args)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        self.assertAlmostEqual(call_args[3], 75_000.0)

    def test_defaults_to_100k_on_broker_error(self):
        args = MagicMock(capital=None, start="2025-01-01", end="2025-12-31")
        mock_run = MagicMock()
        with patch("backtest.run_backtest", mock_run), \
             patch("execution.trader.get_client", side_effect=Exception("no connection")), \
             patch("cli.config.STOCK_UNIVERSE", ["AAPL"]), \
             patch("cli.config.MAX_POSITIONS", 5):
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


if __name__ == "__main__":
    unittest.main()

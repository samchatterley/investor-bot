"""
Tests for the HTML-building logic in notifications/emailer.py.
SMTP is mocked — connections are never established.
"""
import unittest
from unittest.mock import patch, MagicMock
from notifications.emailer import (
    _humanise_detail,
    _build_html,
    _build_weekly_html,
    _build_diagnostics_section,
    _all_recipients,
    _named_recipients,
    _build_trade_cards,
)


def _record(pnl=1000.0, trades=None, stops=None):
    return {
        "date": "2026-04-26",
        "daily_pnl": pnl,
        "account_before": {"portfolio_value": 100_000.0, "cash": 20_000.0},
        "account_after":  {"portfolio_value": 100_000.0 + pnl, "cash": 18_000.0},
        "market_summary": "Steady upward day.",
        "trades_executed": trades or [],
        "stop_losses_triggered": stops or [],
        "buy_candidates": [],
        "position_decisions": [],
        "mode": "paper",
    }


class TestHumaniseDetail(unittest.TestCase):

    def test_empty_string_returns_simulated(self):
        self.assertEqual(_humanise_detail(""), "Simulated — no real order placed")

    def test_dry_run_returns_simulated(self):
        self.assertEqual(_humanise_detail("dry run"), "Simulated — no real order placed")

    def test_dollar_amount_bolded(self):
        result = _humanise_detail("$5000.00")
        self.assertIn("<b>", result)
        self.assertIn("$5000.00", result)

    def test_kelly_translated(self):
        result = _humanise_detail("35% Kelly")
        self.assertIn("35%", result)
        self.assertIn("cash", result)

    def test_confidence_translated(self):
        result = _humanise_detail("confidence=8")
        self.assertIn("8/10", result)

    def test_signal_translated_to_label(self):
        result = _humanise_detail("momentum")
        self.assertIn("Upward momentum", result)

    def test_full_detail_string_all_parts(self):
        detail = "$5000.00 | 35% Kelly | momentum | confidence=8"
        result = _humanise_detail(detail)
        self.assertIn("$5000.00", result)
        self.assertIn("35%", result)
        self.assertIn("Upward momentum", result)
        self.assertIn("8/10", result)


class TestBuildHtml(unittest.TestCase):

    def test_returns_string(self):
        html = _build_html(_record())
        self.assertIsInstance(html, str)

    def test_contains_date(self):
        html = _build_html(_record())
        self.assertIn("2026-04-26", html)

    def test_positive_pnl_shown(self):
        html = _build_html(_record(pnl=1234.56))
        self.assertIn("1,234.56", html)

    def test_negative_pnl_shown(self):
        html = _build_html(_record(pnl=-500.0))
        self.assertIn("500.00", html)

    def test_trade_symbol_appears(self):
        trades = [{"symbol": "NVDA", "action": "BUY", "detail": "$5000"}]
        html = _build_html(_record(trades=trades))
        self.assertIn("NVDA", html)

    def test_no_trades_message_shown(self):
        html = _build_html(_record(trades=[]))
        self.assertIn("No trades", html)

    def test_paper_trading_label(self):
        html = _build_html(_record())
        self.assertIn("Paper trading", html)

    def test_is_valid_html_structure(self):
        html = _build_html(_record())
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("</html>", html)


class TestBuildWeeklyHtml(unittest.TestCase):

    def _review(self, **kwargs):
        base = {
            "week_summary": "Good week overall.",
            "what_worked": ["momentum signals in bull markets"],
            "what_didnt": ["mean reversion in choppy conditions"],
            "lessons": ["Avoid CHOPPY regime for mean reversion setups."],
            "applied_changes": [],
        }
        base.update(kwargs)
        return base

    def test_returns_string(self):
        self.assertIsInstance(_build_weekly_html(self._review()), str)

    def test_contains_week_summary(self):
        html = _build_weekly_html(self._review())
        self.assertIn("Good week overall.", html)

    def test_contains_lesson(self):
        html = _build_weekly_html(self._review())
        self.assertIn("Avoid CHOPPY regime", html)

    def test_applied_config_change_appears(self):
        review = self._review(applied_changes=[{
            "parameter": "MIN_CONFIDENCE",
            "old_value": 7, "new_value": 8,
            "reason": "low conf trades underperformed",
            "status": "applied",
        }])
        html = _build_weekly_html(review)
        self.assertIn("MIN_CONFIDENCE", html)
        self.assertIn("config.py", html)

    def test_no_changes_message_shown(self):
        html = _build_weekly_html(self._review(applied_changes=[]))
        self.assertIn("No parameter changes", html)

    def test_diagnostic_section_included_when_provided(self):
        report = {"status": "PASS", "passed": 50, "total": 50,
                  "duration_seconds": 1.2, "failures": []}
        html = _build_weekly_html(self._review(), test_report=report)
        self.assertIn("PASS", html)
        self.assertIn("50/50", html)

    def test_diagnostic_fail_included(self):
        report = {
            "status": "FAIL", "passed": 48, "total": 50,
            "duration_seconds": 1.5,
            "failures": [{"test": "test_foo.TestBar.test_baz", "message": "AssertionError: 1 != 2"}],
        }
        html = _build_weekly_html(self._review(), test_report=report)
        self.assertIn("FAIL", html)
        self.assertIn("test_baz", html)


class TestBuildDiagnosticsSection(unittest.TestCase):

    def test_none_returns_empty_string(self):
        self.assertEqual(_build_diagnostics_section(None), "")

    def test_pass_shows_green(self):
        report = {"status": "PASS", "passed": 30, "total": 30,
                  "duration_seconds": 0.5, "failures": []}
        html = _build_diagnostics_section(report)
        self.assertIn("PASS", html)
        self.assertIn("#2e7d32", html)

    def test_fail_shows_red(self):
        report = {"status": "FAIL", "passed": 28, "total": 30,
                  "duration_seconds": 0.5,
                  "failures": [{"test": "test_x", "message": "oops"}]}
        html = _build_diagnostics_section(report)
        self.assertIn("FAIL", html)
        self.assertIn("#c62828", html)

    def test_failure_details_rendered(self):
        report = {
            "status": "FAIL", "passed": 9, "total": 10,
            "duration_seconds": 1.0,
            "failures": [{"test": "test_module.TestClass.test_my_check", "message": "Expected True"}],
        }
        html = _build_diagnostics_section(report)
        self.assertIn("test_my_check", html)


class TestAllRecipients(unittest.TestCase):

    def test_returns_owner_when_no_cc(self):
        from unittest.mock import patch
        with patch("notifications.emailer.EMAIL_TO", "owner@example.com"), \
             patch("notifications.emailer.EMAIL_CC", ""):
            result = _all_recipients()
        self.assertEqual(result, ["owner@example.com"])

    def test_includes_cc_addresses(self):
        from unittest.mock import patch
        with patch("notifications.emailer.EMAIL_TO", "owner@example.com"), \
             patch("notifications.emailer.EMAIL_CC", "a@example.com, b@example.com"):
            result = _all_recipients()
        self.assertIn("a@example.com", result)
        self.assertIn("b@example.com", result)
        self.assertEqual(len(result), 3)

    def test_returns_empty_when_no_email_to(self):
        with patch("notifications.emailer.EMAIL_TO", None), \
             patch("notifications.emailer.EMAIL_CC", ""):
            result = _all_recipients()
        self.assertEqual(result, [])


class TestNamedRecipients(unittest.TestCase):

    def test_name_colon_email_format(self):
        with patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@example.com"), \
             patch("notifications.emailer.EMAIL_TO", "sam@example.com"), \
             patch("notifications.emailer.EMAIL_CC", ""):
            result = _named_recipients()
        self.assertEqual(result, [("Sam", "sam@example.com")])

    def test_email_only_format_uses_there(self):
        with patch("notifications.emailer.EMAIL_RECIPIENTS", "sam@example.com"), \
             patch("notifications.emailer.EMAIL_TO", "sam@example.com"), \
             patch("notifications.emailer.EMAIL_CC", ""):
            result = _named_recipients()
        self.assertEqual(result, [("there", "sam@example.com")])

    def test_multiple_recipients(self):
        with patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@a.com,Jo:jo@b.com"), \
             patch("notifications.emailer.EMAIL_TO", ""), \
             patch("notifications.emailer.EMAIL_CC", ""):
            result = _named_recipients()
        self.assertEqual(len(result), 2)
        self.assertIn(("Sam", "sam@a.com"), result)
        self.assertIn(("Jo", "jo@b.com"), result)

    def test_falls_back_to_all_recipients_when_no_email_recipients(self):
        with patch("notifications.emailer.EMAIL_RECIPIENTS", ""), \
             patch("notifications.emailer.EMAIL_TO", "owner@example.com"), \
             patch("notifications.emailer.EMAIL_CC", ""):
            result = _named_recipients()
        self.assertEqual(result, [("there", "owner@example.com")])

    def test_whitespace_stripped_from_parts(self):
        with patch("notifications.emailer.EMAIL_RECIPIENTS", " Sam : sam@example.com "), \
             patch("notifications.emailer.EMAIL_TO", ""), \
             patch("notifications.emailer.EMAIL_CC", ""):
            result = _named_recipients()
        self.assertEqual(result[0], ("Sam", "sam@example.com"))


class TestBuildTradeCards(unittest.TestCase):

    def test_no_trades_returns_no_trades_message(self):
        record = {
            "trades_executed": [], "stop_losses_triggered": [],
            "buy_candidates": [], "position_decisions": [],
        }
        html = _build_trade_cards(record)
        self.assertIn("No trades", html)

    def test_buy_card_shown(self):
        record = {
            "trades_executed": [{"symbol": "AAPL", "action": "BUY", "detail": "$5000"}],
            "stop_losses_triggered": [],
            "buy_candidates": [{"symbol": "AAPL", "reasoning": "Strong momentum"}],
            "position_decisions": [],
        }
        html = _build_trade_cards(record)
        self.assertIn("AAPL", html)
        self.assertIn("BUY", html)

    def test_sell_card_shown(self):
        record = {
            "trades_executed": [{"symbol": "MSFT", "action": "SELL", "detail": "partial exit"}],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [{"symbol": "MSFT", "action": "SELL", "reasoning": "target hit"}],
        }
        html = _build_trade_cards(record)
        self.assertIn("MSFT", html)
        self.assertIn("SELL", html)

    def test_stop_loss_card_shown(self):
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [{"symbol": "NVDA", "pl_pct": -4.2}],
            "buy_candidates": [],
            "position_decisions": [],
        }
        html = _build_trade_cards(record)
        self.assertIn("NVDA", html)
        self.assertIn("STOP LOSS", html)

    def test_reasoning_appears_in_buy_card(self):
        record = {
            "trades_executed": [{"symbol": "AAPL", "action": "BUY", "detail": "$5000"}],
            "stop_losses_triggered": [],
            "buy_candidates": [{"symbol": "AAPL", "reasoning": "RSI oversold bounce"}],
            "position_decisions": [],
        }
        html = _build_trade_cards(record)
        self.assertIn("RSI oversold bounce", html)


class TestSendHtml(unittest.TestCase):

    def test_skips_when_no_credentials(self):
        with patch("notifications.emailer.EMAIL_FROM", ""), \
             patch("notifications.emailer.EMAIL_APP_PASSWORD", ""), \
             patch("notifications.emailer.smtplib.SMTP_SSL") as mock_smtp:
            from notifications.emailer import _send_html
            _send_html("Test", lambda name: "<p>hi</p>")
        mock_smtp.assert_not_called()

    def test_skips_when_no_recipients(self):
        with patch("notifications.emailer.EMAIL_FROM", "bot@example.com"), \
             patch("notifications.emailer.EMAIL_APP_PASSWORD", "secret"), \
             patch("notifications.emailer.EMAIL_RECIPIENTS", ""), \
             patch("notifications.emailer.EMAIL_TO", ""), \
             patch("notifications.emailer.EMAIL_CC", ""), \
             patch("notifications.emailer.smtplib.SMTP_SSL") as mock_smtp:
            from notifications.emailer import _send_html
            _send_html("Test", lambda name: "<p>hi</p>")
        mock_smtp.assert_not_called()

    def test_connects_to_gmail_ssl(self):
        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)
        with patch("notifications.emailer.EMAIL_FROM", "bot@example.com"), \
             patch("notifications.emailer.EMAIL_APP_PASSWORD", "secret"), \
             patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@example.com"), \
             patch("notifications.emailer.smtplib.SMTP_SSL", return_value=mock_server) as mock_ssl:
            from notifications.emailer import _send_html
            _send_html("Subject", lambda name: f"<p>Hi {name}</p>")
        mock_ssl.assert_called_once_with("smtp.gmail.com", 465)

    def test_handles_smtp_exception_gracefully(self):
        with patch("notifications.emailer.EMAIL_FROM", "bot@example.com"), \
             patch("notifications.emailer.EMAIL_APP_PASSWORD", "secret"), \
             patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@example.com"), \
             patch("notifications.emailer.smtplib.SMTP_SSL", side_effect=Exception("conn refused")):
            from notifications.emailer import _send_html
            try:
                _send_html("Subject", lambda name: "<p>hi</p>")
            except Exception:
                self.fail("_send_html raised on SMTP exception")


class TestSendSummary(unittest.TestCase):

    def _record(self, pnl=1000.0):
        return {
            "date": "2026-04-27",
            "daily_pnl": pnl,
            "account_before": {"portfolio_value": 100_000.0},
            "account_after":  {"portfolio_value": 100_000.0 + pnl, "cash": 18_000.0},
            "market_summary": "Steady day.",
            "trades_executed": [], "stop_losses_triggered": [],
            "buy_candidates": [], "position_decisions": [],
        }

    def test_positive_pnl_subject_has_plus(self):
        subjects = []
        def capture_send(subject, html_fn):
            subjects.append(subject)
        with patch("notifications.emailer._send_html", side_effect=capture_send):
            from notifications.emailer import send_summary
            send_summary(self._record(pnl=500.0))
        self.assertTrue(subjects[0].startswith("Trading Bot 2026-04-27 — +"))

    def test_negative_pnl_subject_has_minus(self):
        subjects = []
        def capture_send(subject, html_fn):
            subjects.append(subject)
        with patch("notifications.emailer._send_html", side_effect=capture_send):
            from notifications.emailer import send_summary
            send_summary(self._record(pnl=-300.0))
        self.assertIn("300", subjects[0])
        self.assertNotIn("+", subjects[0].split("—")[1])


class TestSendWeeklyReview(unittest.TestCase):

    def _review(self, applied=None):
        return {
            "week_summary": "Good week.",
            "what_worked": [], "what_didnt": [], "lessons": [],
            "applied_changes": applied or [],
        }

    def test_calls_send_html(self):
        with patch("notifications.emailer._send_html") as mock_send:
            from notifications.emailer import send_weekly_review
            send_weekly_review(self._review())
        mock_send.assert_called_once()

    def test_subject_includes_change_count_when_applied(self):
        subjects = []
        def capture(subject, html_fn):
            subjects.append(subject)
        applied = [{"parameter": "MIN_CONFIDENCE", "old_value": 7, "new_value": 8,
                    "reason": "test", "status": "applied"}]
        with patch("notifications.emailer._send_html", side_effect=capture):
            from notifications.emailer import send_weekly_review
            send_weekly_review(self._review(applied=applied))
        self.assertIn("1 config change", subjects[0])

    def test_subject_includes_test_status_when_provided(self):
        subjects = []
        def capture(subject, html_fn):
            subjects.append(subject)
        with patch("notifications.emailer._send_html", side_effect=capture):
            from notifications.emailer import send_weekly_review
            send_weekly_review(self._review(), test_report={"status": "PASS"})
        self.assertIn("PASS", subjects[0])

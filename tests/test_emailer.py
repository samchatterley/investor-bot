"""
Tests for the HTML-building logic in notifications/emailer.py.
No SMTP is exercised — send_summary and send_weekly_review are excluded.
"""
import unittest
from notifications.emailer import (
    _humanise_detail,
    _build_html,
    _build_weekly_html,
    _build_diagnostics_section,
    _all_recipients,
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
        from unittest.mock import patch
        with patch("notifications.emailer.EMAIL_TO", None), \
             patch("notifications.emailer.EMAIL_CC", ""):
            result = _all_recipients()
        self.assertEqual(result, [])

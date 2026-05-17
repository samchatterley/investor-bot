"""
Tests for the HTML-building logic in notifications/emailer.py.
SMTP is mocked — connections are never established.
"""

import unittest
from unittest.mock import MagicMock, patch

from notifications.emailer import (
    _all_recipients,
    _build_attribution_html,
    _build_closed_section,
    _build_diagnostics_section,
    _build_html,
    _build_positions_section,
    _build_trade_cards,
    _build_weekly_html,
    _get_live_positions,
    _humanise_detail,
    _named_recipients,
    _parse_unrealized_pct,
)


def _record(pnl=1000.0, trades=None, stops=None):
    return {
        "date": "2026-04-26",
        "daily_pnl": pnl,
        "account_before": {"portfolio_value": 100_000.0, "cash": 20_000.0},
        "account_after": {"portfolio_value": 100_000.0 + pnl, "cash": 18_000.0},
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

    def test_dry_run_part_within_multi_part_string(self):
        # Line 107-108: part.startswith("dry run") inside the parts loop
        result = _humanise_detail("dry run $100")
        self.assertIn("Simulated", result)


class TestBuildHtml(unittest.TestCase):
    def setUp(self):
        self._patcher = patch("notifications.emailer._get_live_positions", return_value={})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

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

    def test_no_trades_stat_shows_zero(self):
        html = _build_html(_record(trades=[]))
        # Trades stat in the P&L hero shows 0 when there are no trades
        self.assertIn(">0<", html)

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
        review = self._review(
            applied_changes=[
                {
                    "parameter": "MIN_CONFIDENCE",
                    "old_value": 7,
                    "new_value": 8,
                    "reason": "low conf trades underperformed",
                    "status": "applied",
                }
            ]
        )
        html = _build_weekly_html(review)
        self.assertIn("MIN_CONFIDENCE", html)
        self.assertIn("config.py", html)

    def test_no_changes_message_shown(self):
        html = _build_weekly_html(self._review(applied_changes=[]))
        self.assertIn("No parameter changes", html)

    def test_diagnostic_section_included_when_provided(self):
        report = {
            "status": "PASS",
            "passed": 50,
            "total": 50,
            "duration_seconds": 1.2,
            "failures": [],
        }
        html = _build_weekly_html(self._review(), test_report=report)
        self.assertIn("PASS", html)
        self.assertIn("50/50", html)

    def test_diagnostic_fail_included(self):
        report = {
            "status": "FAIL",
            "passed": 48,
            "total": 50,
            "duration_seconds": 1.5,
            "failures": [
                {"test": "test_foo.TestBar.test_baz", "message": "AssertionError: 1 != 2"}
            ],
        }
        html = _build_weekly_html(self._review(), test_report=report)
        self.assertIn("FAIL", html)
        self.assertIn("test_baz", html)

    def test_bullets_with_empty_list_shows_none_noted(self):
        # Line 393: _bullets([]) returns the "None noted" placeholder paragraph
        review = self._review(what_worked=[], what_didnt=[], lessons=[])
        html = _build_weekly_html(review)
        self.assertIn("None noted this week", html)

    def test_no_rejected_changes_omits_rejected_block(self):
        # Line 443: else branch — rejected is empty so rejected_block stays ""
        review = self._review(
            applied_changes=[
                {
                    "parameter": "MIN_CONFIDENCE",
                    "old_value": 7,
                    "new_value": 8,
                    "reason": "test",
                    "status": "applied",
                }
            ]
        )
        html = _build_weekly_html(review)
        self.assertNotIn("Suggestions outside safe bounds", html)

    def test_rejected_changes_section_included(self):
        # Line 443: rejected is non-empty → _section called to build rejected_block
        review = self._review(
            applied_changes=[
                {
                    "parameter": "SUPER_LEVER",
                    "old_value": 0,
                    "new_value": 99,
                    "reason": "test",
                    "status": "rejected",
                    "rejection_reason": "not in the safe-to-modify list",
                }
            ]
        )
        html = _build_weekly_html(review)
        self.assertIn("Suggestions outside safe bounds", html)
        self.assertIn("SUPER_LEVER", html)


class TestBuildDiagnosticsSection(unittest.TestCase):
    def test_none_returns_empty_string(self):
        self.assertEqual(_build_diagnostics_section(None), "")

    def test_pass_shows_green(self):
        report = {
            "status": "PASS",
            "passed": 30,
            "total": 30,
            "duration_seconds": 0.5,
            "failures": [],
        }
        html = _build_diagnostics_section(report)
        self.assertIn("PASS", html)
        self.assertIn("#2e7d32", html)

    def test_fail_shows_red(self):
        report = {
            "status": "FAIL",
            "passed": 28,
            "total": 30,
            "duration_seconds": 0.5,
            "failures": [{"test": "test_x", "message": "oops"}],
        }
        html = _build_diagnostics_section(report)
        self.assertIn("FAIL", html)
        self.assertIn("#c62828", html)

    def test_failure_details_rendered(self):
        report = {
            "status": "FAIL",
            "passed": 9,
            "total": 10,
            "duration_seconds": 1.0,
            "failures": [
                {"test": "test_module.TestClass.test_my_check", "message": "Expected True"}
            ],
        }
        html = _build_diagnostics_section(report)
        self.assertIn("test_my_check", html)


class TestAllRecipients(unittest.TestCase):
    def test_returns_owner_when_no_cc(self):
        from unittest.mock import patch

        with (
            patch("notifications.emailer.EMAIL_TO", "owner@example.com"),
            patch("notifications.emailer.EMAIL_CC", ""),
        ):
            result = _all_recipients()
        self.assertEqual(result, ["owner@example.com"])

    def test_includes_cc_addresses(self):
        from unittest.mock import patch

        with (
            patch("notifications.emailer.EMAIL_TO", "owner@example.com"),
            patch("notifications.emailer.EMAIL_CC", "a@example.com, b@example.com"),
        ):
            result = _all_recipients()
        self.assertIn("a@example.com", result)
        self.assertIn("b@example.com", result)
        self.assertEqual(len(result), 3)

    def test_returns_empty_when_no_email_to(self):
        with (
            patch("notifications.emailer.EMAIL_TO", None),
            patch("notifications.emailer.EMAIL_CC", ""),
        ):
            result = _all_recipients()
        self.assertEqual(result, [])


class TestNamedRecipients(unittest.TestCase):
    def test_name_colon_email_format(self):
        with (
            patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@example.com"),
            patch("notifications.emailer.EMAIL_TO", "sam@example.com"),
            patch("notifications.emailer.EMAIL_CC", ""),
        ):
            result = _named_recipients()
        self.assertEqual(result, [("Sam", "sam@example.com")])

    def test_email_only_format_uses_there(self):
        with (
            patch("notifications.emailer.EMAIL_RECIPIENTS", "sam@example.com"),
            patch("notifications.emailer.EMAIL_TO", "sam@example.com"),
            patch("notifications.emailer.EMAIL_CC", ""),
        ):
            result = _named_recipients()
        self.assertEqual(result, [("there", "sam@example.com")])

    def test_multiple_recipients(self):
        with (
            patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@a.com,Jo:jo@b.com"),
            patch("notifications.emailer.EMAIL_TO", ""),
            patch("notifications.emailer.EMAIL_CC", ""),
        ):
            result = _named_recipients()
        self.assertEqual(len(result), 2)
        self.assertIn(("Sam", "sam@a.com"), result)
        self.assertIn(("Jo", "jo@b.com"), result)

    def test_falls_back_to_all_recipients_when_no_email_recipients(self):
        with (
            patch("notifications.emailer.EMAIL_RECIPIENTS", ""),
            patch("notifications.emailer.EMAIL_TO", "owner@example.com"),
            patch("notifications.emailer.EMAIL_CC", ""),
        ):
            result = _named_recipients()
        self.assertEqual(result, [("there", "owner@example.com")])

    def test_whitespace_stripped_from_parts(self):
        with (
            patch("notifications.emailer.EMAIL_RECIPIENTS", " Sam : sam@example.com "),
            patch("notifications.emailer.EMAIL_TO", ""),
            patch("notifications.emailer.EMAIL_CC", ""),
        ):
            result = _named_recipients()
        self.assertEqual(result[0], ("Sam", "sam@example.com"))


class TestBuildTradeCards(unittest.TestCase):
    def test_no_buys_returns_empty(self):
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [],
        }
        html = _build_trade_cards(record)
        self.assertEqual(html, "")

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

    def test_sell_not_in_trade_cards(self):
        # SELLs belong in _build_closed_section, not _build_trade_cards
        record = {
            "trades_executed": [{"symbol": "MSFT", "action": "SELL", "detail": "partial exit"}],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [{"symbol": "MSFT", "action": "SELL", "reasoning": "target hit"}],
        }
        html = _build_trade_cards(record)
        self.assertEqual(html, "")

    def test_stop_loss_not_in_trade_cards(self):
        # Stop losses belong in _build_closed_section, not _build_trade_cards
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [{"symbol": "NVDA", "pl_pct": -4.2}],
            "buy_candidates": [],
            "position_decisions": [],
        }
        html = _build_trade_cards(record)
        self.assertEqual(html, "")


class TestBuildClosedSection(unittest.TestCase):
    def test_sell_card_shown(self):
        record = {
            "trades_executed": [{"symbol": "MSFT", "action": "SELL", "detail": "partial exit"}],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [{"symbol": "MSFT", "action": "SELL", "reasoning": "target hit"}],
        }
        html = _build_closed_section(record)
        self.assertIn("MSFT", html)
        self.assertIn("SELL", html)

    def test_stop_loss_card_shown(self):
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [{"symbol": "NVDA", "pl_pct": -4.2}],
            "buy_candidates": [],
            "position_decisions": [],
        }
        html = _build_closed_section(record)
        self.assertIn("NVDA", html)
        self.assertIn("STOP LOSS", html)

    def test_empty_when_nothing_closed(self):
        record = {
            "trades_executed": [{"symbol": "AAPL", "action": "BUY", "detail": "$5000"}],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [],
        }
        html = _build_closed_section(record)
        self.assertEqual(html, "")

    def test_stop_loss_pl_shown(self):
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [{"symbol": "TSLA", "pl_pct": -6.5}],
            "buy_candidates": [],
            "position_decisions": [],
        }
        html = _build_closed_section(record)
        self.assertIn("-6.5%", html)

    def test_sell_reasoning_shown(self):
        record = {
            "trades_executed": [{"symbol": "AMZN", "action": "SELL", "detail": "exit"}],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [
                {"symbol": "AMZN", "action": "SELL", "reasoning": "hit profit target"}
            ],
        }
        html = _build_closed_section(record)
        self.assertIn("hit profit target", html)

    def test_header_shows_count(self):
        record = {
            "trades_executed": [{"symbol": "AAPL", "action": "SELL", "detail": "exit"}],
            "stop_losses_triggered": [{"symbol": "NVDA", "pl_pct": -3.0}],
            "buy_candidates": [],
            "position_decisions": [],
        }
        html = _build_closed_section(record)
        self.assertIn("Closed today (2)", html)

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
        with (
            patch("notifications.emailer.EMAIL_FROM", ""),
            patch("notifications.emailer.EMAIL_APP_PASSWORD", ""),
            patch("notifications.emailer.smtplib.SMTP_SSL") as mock_smtp,
        ):
            from notifications.emailer import _send_html

            _send_html("Test", lambda name: "<p>hi</p>")
        mock_smtp.assert_not_called()

    def test_skips_when_no_recipients(self):
        with (
            patch("notifications.emailer.EMAIL_FROM", "bot@example.com"),
            patch("notifications.emailer.EMAIL_APP_PASSWORD", "secret"),
            patch("notifications.emailer.EMAIL_RECIPIENTS", ""),
            patch("notifications.emailer.EMAIL_TO", ""),
            patch("notifications.emailer.EMAIL_CC", ""),
            patch("notifications.emailer.smtplib.SMTP_SSL") as mock_smtp,
        ):
            from notifications.emailer import _send_html

            _send_html("Test", lambda name: "<p>hi</p>")
        mock_smtp.assert_not_called()

    def test_connects_to_gmail_ssl(self):
        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)
        with (
            patch("notifications.emailer.EMAIL_FROM", "bot@example.com"),
            patch("notifications.emailer.EMAIL_APP_PASSWORD", "secret"),
            patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@example.com"),
            patch("notifications.emailer.smtplib.SMTP_SSL", return_value=mock_server) as mock_ssl,
        ):
            from notifications.emailer import _send_html

            _send_html("Subject", lambda name: f"<p>Hi {name}</p>")
        mock_ssl.assert_called_once_with("smtp.gmail.com", 465)

    def test_handles_smtp_exception_gracefully(self):
        with (
            patch("notifications.emailer.EMAIL_FROM", "bot@example.com"),
            patch("notifications.emailer.EMAIL_APP_PASSWORD", "secret"),
            patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@example.com"),
            patch("notifications.emailer.smtplib.SMTP_SSL", side_effect=Exception("conn refused")),
        ):
            from notifications.emailer import _send_html

            try:
                _send_html("Subject", lambda name: "<p>hi</p>")
            except Exception:  # pragma: no cover
                self.fail("_send_html raised on SMTP exception")


class TestSendSummary(unittest.TestCase):
    def _record(self, pnl=1000.0):
        return {
            "date": "2026-04-27",
            "daily_pnl": pnl,
            "account_before": {"portfolio_value": 100_000.0},
            "account_after": {"portfolio_value": 100_000.0 + pnl, "cash": 18_000.0},
            "market_summary": "Steady day.",
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [],
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
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
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

        applied = [
            {
                "parameter": "MIN_CONFIDENCE",
                "old_value": 7,
                "new_value": 8,
                "reason": "test",
                "status": "applied",
            }
        ]
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


# ── Coverage gap additions ─────────────────────────────────────────────────────


class TestBuildTradeCardsDecisionPath(unittest.TestCase):
    """Lines 271-272: BUY from decisions when symbol not already in buy_reasons."""

    def test_decision_buy_adds_reasoning_when_not_in_candidates(self):
        """Line 271-272: symbol in decisions but NOT in buy_candidates → adds buy_reason."""
        record = {
            "trades_executed": [{"symbol": "TSLA", "action": "BUY", "detail": "$3000"}],
            "stop_losses_triggered": [],
            "buy_candidates": [],  # TSLA not here
            "position_decisions": [],
            "decisions": [
                {
                    "symbol": "TSLA",
                    "decision_type": "BUY",
                    "reasoning": "Breakout from consolidation",
                }
            ],
        }
        html = _build_trade_cards(record)
        self.assertIn("TSLA", html)
        self.assertIn("Breakout from consolidation", html)

    def test_signal_label_used_when_reasoning_absent(self):
        """Line 290: detail contains a signal name in _SIGNAL_LABELS → uses label as reasoning."""
        record = {
            "trades_executed": [{"symbol": "NVDA", "action": "BUY", "detail": "momentum"}],
            "stop_losses_triggered": [],
            "buy_candidates": [],  # no reasoning from candidates
            "position_decisions": [],
            "decisions": [],
        }
        html = _build_trade_cards(record)
        self.assertIn("NVDA", html)
        # momentum is in _SIGNAL_LABELS → its label "Upward momentum" should appear
        self.assertIn("Upward momentum", html)


class TestParseUnrealizedPct(unittest.TestCase):
    """Lines 362-374: test all four regex patterns and the None fallback."""

    def test_unrealized_pattern(self):
        """Pattern 1: '+4.49% unrealized'."""
        result = _parse_unrealized_pct("+4.49% unrealized")
        self.assertAlmostEqual(result, 4.49)

    def test_unrealised_variant(self):
        """Pattern 1: 'unrealised' spelling variant."""
        result = _parse_unrealized_pct("-1.2% unrealised")
        self.assertAlmostEqual(result, -1.2)

    def test_from_entry_pattern(self):
        """Pattern 2: '+4.5% from entry'."""
        result = _parse_unrealized_pct("+4.5% from entry")
        self.assertAlmostEqual(result, 4.5)

    def test_from_entry_paren_pattern(self):
        """Pattern 3: 'from entry (-0.097%)'."""
        result = _parse_unrealized_pct("Holding position from entry (-0.097%)")
        self.assertAlmostEqual(result, -0.097)

    def test_up_pattern(self):
        """Pattern 4: 'up 4.5%'."""
        result = _parse_unrealized_pct("Position is up 4.5%")
        self.assertAlmostEqual(result, 4.5)

    def test_no_match_returns_none(self):
        """Returns None when no pattern matches."""
        result = _parse_unrealized_pct("No percentage info here")
        self.assertIsNone(result)

    def test_float_conversion_error_continues_to_none(self):
        """Lines 372-373: regex matches but float(group(1)) raises ValueError → pass → None."""
        mock_match = MagicMock()
        mock_match.group.return_value = "not_a_float"
        with patch("notifications.emailer.re.search", side_effect=[mock_match, None, None, None]):
            result = _parse_unrealized_pct("whatever string")
        self.assertIsNone(result)


class TestGetLivePositions(unittest.TestCase):
    """Lines 378-383: _get_live_positions success and exception paths."""

    def test_returns_positions_dict_on_success(self):
        """Happy path: _load_all_positions returns a dict."""
        fake_positions = {"AAPL": {"shares": 10, "entry_price": 150.0}}
        with patch("execution.trader._load_all_positions", return_value=fake_positions):
            result = _get_live_positions()
        self.assertEqual(result, fake_positions)

    def test_returns_empty_dict_on_exception(self):
        """Exception path: import or call raises → returns {}."""
        with patch(
            "execution.trader._load_all_positions", side_effect=Exception("trader unavailable")
        ):
            result = _get_live_positions()
        self.assertEqual(result, {})

    def test_returns_empty_dict_when_import_fails(self):
        """ImportError (e.g. trader not available) → returns {}."""
        import builtins

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if "execution.trader" in name or name == "execution.trader":
                raise ImportError("execution module not available")
            return real_import(name, *args, **kwargs)  # pragma: no cover

        with patch("builtins.__import__", side_effect=_blocking_import):
            result = _get_live_positions()
        self.assertEqual(result, {})


class TestBuildPositionsSection(unittest.TestCase):
    """Lines 393-493: _build_positions_section with live positions and HOLD-only fallback."""

    def _record_with_hold(self, symbols=None):
        """Build a record with HOLD decisions for the given symbols."""
        syms = symbols or ["AAPL"]
        position_decisions = [
            {
                "symbol": sym,
                "action": "HOLD",
                "reasoning": f"+2.5% unrealized (holding {sym})",
                "summary": f"Holding {sym}",
            }
            for sym in syms
        ]
        return {
            "date": "2026-04-26",
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": position_decisions,
            "decisions": [],
        }

    def test_live_positions_path_sorted_keys(self):
        """Line 399: all_positions non-empty → symbols = sorted(all_positions.keys())."""
        live = {"NVDA": {"signal": "momentum", "entry_date": "2026-04-20"}}
        record = self._record_with_hold(["NVDA"])
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        self.assertIn("NVDA", html)
        self.assertIn("Open positions", html)

    def test_fallback_to_hold_decisions_when_no_live_positions(self):
        """Lines 404-407: all_positions empty → fall back to HOLD decisions."""
        record = self._record_with_hold(["MSFT"])
        with patch("notifications.emailer._get_live_positions", return_value={}):
            html = _build_positions_section(record)
        self.assertIn("MSFT", html)
        self.assertIn("Open positions", html)

    def test_returns_empty_string_when_no_symbols(self):
        """Returns '' when no live positions and no HOLD decisions."""
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [],
            "decisions": [],
        }
        with patch("notifications.emailer._get_live_positions", return_value={}):
            html = _build_positions_section(record)
        self.assertEqual(html, "")

    def test_pnl_pct_shown_in_green_when_positive(self):
        """Lines 459-462: pct >= 0 → green color in html."""
        live = {"AAPL": {"signal": "", "entry_date": ""}}
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [
                {
                    "symbol": "AAPL",
                    "action": "HOLD",
                    "reasoning": "+3.1% unrealized",
                    "summary": "",
                }
            ],
            "decisions": [],
        }
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        # green color for positive pct
        self.assertIn("#2e7d32", html)
        self.assertIn("+3.10%", html)

    def test_pnl_pct_shown_in_red_when_negative(self):
        """Lines 460-461: pct < 0 → red color."""
        live = {"AAPL": {"signal": "", "entry_date": ""}}
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [
                {
                    "symbol": "AAPL",
                    "action": "HOLD",
                    "reasoning": "-1.5% unrealized",
                    "summary": "",
                }
            ],
            "decisions": [],
        }
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        self.assertIn("#c62828", html)

    def test_days_held_computed_from_entry_date(self):
        """Lines 451-456: entry_date set → days_held computed."""
        live = {"AAPL": {"signal": "momentum", "entry_date": "2026-04-01"}}
        record = self._record_with_hold(["AAPL"])
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        # Should contain a "d" suffix for days held
        self.assertRegex(html, r"\d+d")

    def test_signal_label_from_pos_meta(self):
        """Lines 433-447: signal_key from pos_meta → label looked up."""
        live = {"AAPL": {"signal": "mean_reversion", "entry_date": ""}}
        record = self._record_with_hold(["AAPL"])
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        # mean_reversion maps to "Oversold bounce" in _SIGNAL_LABELS
        self.assertIn("Oversold bounce", html)

    def test_signal_from_decisions_when_not_in_pos_meta(self):
        """Lines 435-441: signal_key absent in pos_meta → look in decisions list."""
        live = {"AAPL": {"entry_date": ""}}  # no 'signal' key
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [
                {"symbol": "AAPL", "action": "HOLD", "reasoning": "", "summary": ""}
            ],
            "decisions": [
                {
                    "symbol": "AAPL",
                    "decision_type": "buy",
                    "key_signal": "momentum",
                }
            ],
        }
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        self.assertIn("AAPL", html)

    def test_buy_reasons_from_decisions_added(self):
        """Lines 419-422: BUY decision for symbol not in buy_candidates → buy_reasons."""
        live = {"TSLA": {"entry_date": ""}}
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [
                {"symbol": "TSLA", "action": "HOLD", "reasoning": "", "summary": ""}
            ],
            "decisions": [
                {
                    "symbol": "TSLA",
                    "decision_type": "BUY",
                    "reasoning": "Strong breakout setup",
                }
            ],
        }
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        self.assertIn("TSLA", html)
        self.assertIn("Strong breakout setup", html)

    def test_buy_reasons_from_buy_candidates(self):
        """Lines 416-418: buy_candidates with symbol → added to buy_reasons."""
        live = {"GOOG": {"entry_date": ""}}
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [{"symbol": "GOOG", "reasoning": "Strong momentum setup"}],
            "position_decisions": [
                {"symbol": "GOOG", "action": "HOLD", "reasoning": "", "summary": ""}
            ],
            "decisions": [],
        }
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        self.assertIn("GOOG", html)
        self.assertIn("Strong momentum setup", html)

    def test_invalid_entry_date_gracefully_handled(self):
        """Lines 455-456: entry_date_str can't be parsed → ValueError caught, days_held=''."""
        live = {"AAPL": {"signal": "", "entry_date": "not-a-date"}}
        record = self._record_with_hold(["AAPL"])
        with patch("notifications.emailer._get_live_positions", return_value=live):
            # Should not raise even with an invalid entry_date
            html = _build_positions_section(record)
        self.assertIn("AAPL", html)

    def test_non_hold_position_decision_not_indexed(self):
        """Line 394->392: position_decision with action != HOLD is skipped in hold_decisions."""
        live = {"AAPL": {"signal": "", "entry_date": ""}}
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [
                {"symbol": "AAPL", "action": "SELL", "reasoning": "", "summary": "sold"},
                {
                    "symbol": "AAPL",
                    "action": "HOLD",
                    "reasoning": "+1.0% unrealized",
                    "summary": "",
                },
            ],
            "decisions": [],
        }
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        self.assertIn("AAPL", html)

    def test_non_hold_action_skipped_in_fallback_symbol_list(self):
        """Line 405->403: non-HOLD action in fallback loop is skipped."""
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [
                {"symbol": "TSLA", "action": "SELL", "reasoning": "", "summary": ""},
                {"symbol": "AAPL", "action": "HOLD", "reasoning": "", "summary": ""},
            ],
            "decisions": [],
        }
        with patch("notifications.emailer._get_live_positions", return_value={}):
            html = _build_positions_section(record)
        self.assertIn("AAPL", html)
        self.assertNotIn("TSLA", html)

    def test_buy_candidate_with_empty_symbol_skipped(self):
        """Line 417->415: buy_candidate with empty symbol is skipped."""
        live = {"AAPL": {"signal": "", "entry_date": ""}}
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [
                {"symbol": "", "reasoning": "should be skipped"},
                {"symbol": "AAPL", "reasoning": "valid candidate"},
            ],
            "position_decisions": [
                {"symbol": "AAPL", "action": "HOLD", "reasoning": "", "summary": ""}
            ],
            "decisions": [],
        }
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        self.assertIn("AAPL", html)
        self.assertIn("valid candidate", html)

    def test_buy_reason_not_overwritten_when_symbol_in_buy_candidates(self):
        """Line 421->419: BUY decision for symbol already in buy_reasons is skipped."""
        live = {"AAPL": {"signal": "", "entry_date": ""}}
        record = {
            "trades_executed": [],
            "stop_losses_triggered": [],
            "buy_candidates": [{"symbol": "AAPL", "reasoning": "from candidates"}],
            "position_decisions": [
                {"symbol": "AAPL", "action": "HOLD", "reasoning": "", "summary": ""}
            ],
            "decisions": [
                {"symbol": "AAPL", "decision_type": "BUY", "reasoning": "from decisions"}
            ],
        }
        with patch("notifications.emailer._get_live_positions", return_value=live):
            html = _build_positions_section(record)
        self.assertIn("from candidates", html)
        self.assertNotIn("from decisions", html)


class TestNamedRecipientsEmptyPart(unittest.TestCase):
    """Line 36->31: EMAIL_RECIPIENTS part with no colon and empty string skips appending."""

    def test_trailing_comma_creates_empty_part_skipped(self):
        with (
            patch("notifications.emailer.EMAIL_RECIPIENTS", "Sam:sam@a.com,"),
            patch("notifications.emailer.EMAIL_TO", ""),
            patch("notifications.emailer.EMAIL_CC", ""),
        ):
            result = _named_recipients()
        self.assertEqual(result, [("Sam", "sam@a.com")])


class TestHumaniseDetailEmptyPart(unittest.TestCase):
    """Line 223->210: detail string with empty segment (adjacent pipes) skips append."""

    def test_empty_segment_between_pipes_skipped(self):
        result = _humanise_detail("momentum||confidence=8")
        self.assertIn("Upward momentum", result)
        self.assertIn("8/10", result)
        self.assertNotIn("None", result)


class TestTradeCardNoDetail(unittest.TestCase):
    """Line 230->237: _trade_card with empty detail_str → detail_row is empty string."""

    def test_buy_card_with_pipe_only_detail_gives_empty_detail_str(self):
        record = {
            "trades_executed": [{"symbol": "AAPL", "action": "BUY", "detail": "|"}],
            "stop_losses_triggered": [],
            "buy_candidates": [{"symbol": "AAPL", "reasoning": "Strong setup"}],
            "position_decisions": [],
            "decisions": [],
        }
        html = _build_trade_cards(record)
        self.assertIn("AAPL", html)
        self.assertNotIn("padding:6px 16px 14px;background:#f1f8f1", html)


class TestBuildTradeCardsNonBuyDecision(unittest.TestCase):
    """Line 271->270: decisions loop with non-BUY decision_type is skipped."""

    def test_non_buy_decision_not_added_to_buy_reasons(self):
        record = {
            "trades_executed": [{"symbol": "NVDA", "action": "BUY", "detail": "$3000"}],
            "stop_losses_triggered": [],
            "buy_candidates": [],
            "position_decisions": [],
            "decisions": [
                {"symbol": "NVDA", "decision_type": "HOLD", "reasoning": "already holding"}
            ],
        }
        html = _build_trade_cards(record)
        self.assertIn("NVDA", html)
        self.assertNotIn("already holding", html)


class TestBuildAttributionHtml(unittest.TestCase):
    """Lines 695-757: _build_attribution_html renders 4 breakdown tables."""

    def _attribution(self, total=5, by_signal=None, by_regime=None, by_sector=None, by_hold=None):
        return {
            "by_signal": by_signal
            or {"momentum": {"trades": 3, "win_rate": 66.7, "avg_return_pct": 2.1}},
            "by_regime": by_regime
            or {"BULL_TRENDING": {"trades": 3, "win_rate": 66.7, "avg_return_pct": 2.1}},
            "by_sector": by_sector
            or {"Technology": {"trades": 2, "win_rate": 50.0, "avg_return_pct": -0.5}},
            "by_hold_days": by_hold
            or {"1-2d": {"trades": 2, "win_rate": 50.0, "avg_return_pct": 1.0}},
            "total_trades": total,
            "period_days": 90,
        }

    def test_empty_dict_returns_empty_string(self):
        self.assertEqual(_build_attribution_html({}), "")

    def test_zero_total_trades_returns_empty_string(self):
        attr = self._attribution(total=0)
        self.assertEqual(_build_attribution_html(attr), "")

    def test_all_empty_breakdowns_returns_empty_string(self):
        attr = {
            "by_signal": {},
            "by_regime": {},
            "by_sector": {},
            "by_hold_days": {},
            "total_trades": 5,
            "period_days": 90,
        }
        self.assertEqual(_build_attribution_html(attr), "")

    def test_with_data_returns_html_with_header(self):
        html = _build_attribution_html(self._attribution())
        self.assertIn("Performance Attribution", html)
        self.assertIn("90", html)
        self.assertIn("5 trades", html)

    def test_signal_label_appears_in_output(self):
        html = _build_attribution_html(self._attribution())
        self.assertIn("momentum", html)

    def test_positive_avg_return_uses_green_colour(self):
        attr = self._attribution(
            by_signal={"momentum": {"trades": 3, "win_rate": 100.0, "avg_return_pct": 2.5}},
            by_regime={},
            by_sector={},
            by_hold={},
        )
        html = _build_attribution_html(attr)
        self.assertIn("#2e7d32", html)

    def test_negative_avg_return_uses_red_colour(self):
        attr = self._attribution(
            by_signal={"mean_reversion": {"trades": 2, "win_rate": 0.0, "avg_return_pct": -1.3}},
            by_regime={},
            by_sector={},
            by_hold={},
        )
        html = _build_attribution_html(attr)
        self.assertIn("#c62828", html)

    def test_period_days_in_header(self):
        attr = self._attribution()
        attr["period_days"] = 30
        html = _build_attribution_html(attr)
        self.assertIn("30", html)

    def test_win_rate_formatted_as_percentage(self):
        attr = self._attribution(
            by_signal={"momentum": {"trades": 3, "win_rate": 66.7, "avg_return_pct": 1.0}},
            by_regime={},
            by_sector={},
            by_hold={},
        )
        html = _build_attribution_html(attr)
        self.assertIn("67%", html)

    def test_regime_breakdown_appears(self):
        html = _build_attribution_html(self._attribution())
        self.assertIn("BULL_TRENDING", html)

    def test_sector_breakdown_appears(self):
        html = _build_attribution_html(self._attribution())
        self.assertIn("Technology", html)

    def test_hold_duration_breakdown_appears(self):
        html = _build_attribution_html(self._attribution())
        self.assertIn("1-2d", html)

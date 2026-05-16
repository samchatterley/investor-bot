"""Tests for notifications/alerts.py — emergency alert emails."""

import unittest
from unittest.mock import MagicMock, patch


class TestAlertSend(unittest.TestCase):
    def _send_with_creds(self, subject, body):
        """Call _send with valid-looking credentials and a mocked SMTP server."""
        mock_server = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("notifications.alerts.EMAIL_FROM", "bot@gmail.com"),
            patch("notifications.alerts.EMAIL_TO", "owner@gmail.com"),
            patch("notifications.alerts.EMAIL_APP_PASSWORD", "secret"),
            patch("notifications.alerts.smtplib.SMTP_SSL", mock_smtp_cls),
        ):
            from notifications.alerts import _send

            _send(subject, body)

        return mock_smtp_cls, mock_server

    def test_send_skips_when_no_credentials(self):
        with (
            patch("notifications.alerts.EMAIL_FROM", ""),
            patch("notifications.alerts.EMAIL_TO", "owner@example.com"),
            patch("notifications.alerts.EMAIL_APP_PASSWORD", ""),
            patch("notifications.alerts.smtplib.SMTP_SSL") as mock_smtp,
        ):
            from notifications.alerts import _send

            _send("subject", "body")
            mock_smtp.assert_not_called()

    def test_send_skips_when_no_recipient(self):
        with (
            patch("notifications.alerts.EMAIL_FROM", "bot@gmail.com"),
            patch("notifications.alerts.EMAIL_TO", ""),
            patch("notifications.alerts.EMAIL_APP_PASSWORD", "secret"),
            patch("notifications.alerts.smtplib.SMTP_SSL") as mock_smtp,
        ):
            from notifications.alerts import _send

            _send("subject", "body")
            mock_smtp.assert_not_called()

    def test_send_connects_to_gmail_smtp(self):
        mock_smtp_cls = MagicMock()
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("notifications.alerts.EMAIL_FROM", "bot@gmail.com"),
            patch("notifications.alerts.EMAIL_TO", "owner@gmail.com"),
            patch("notifications.alerts.EMAIL_APP_PASSWORD", "secret"),
            patch("notifications.alerts.smtplib.SMTP_SSL", mock_smtp_cls),
        ):
            from notifications.alerts import _send

            _send("test", "body")

        mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 465)

    def test_send_does_not_raise_on_smtp_error(self):
        with (
            patch("notifications.alerts.EMAIL_FROM", "bot@gmail.com"),
            patch("notifications.alerts.EMAIL_TO", "owner@gmail.com"),
            patch("notifications.alerts.EMAIL_APP_PASSWORD", "secret"),
            patch(
                "notifications.alerts.smtplib.SMTP_SSL", side_effect=Exception("connection refused")
            ),
        ):
            from notifications.alerts import _send

            try:
                _send("subject", "body")
            except Exception:  # pragma: no cover
                self.fail("_send raised an exception instead of handling it")

    def test_send_calls_login_and_sendmail(self):
        mock_smtp_cls, mock_server = self._send_with_creds("Hello", "World")
        mock_server.login.assert_called_once_with("bot@gmail.com", "secret")
        mock_server.sendmail.assert_called_once()


class TestAlertFunctions(unittest.TestCase):
    def _capture_send(self):
        """Context manager that captures _send calls and returns (subject, body)."""
        calls = []

        def fake_send(subject, body):
            calls.append((subject, body))

        return fake_send, calls

    def test_alert_circuit_breaker_subject(self):
        fake_send, calls = self._capture_send()
        with patch("notifications.alerts._send", fake_send):
            from notifications.alerts import alert_circuit_breaker

            alert_circuit_breaker(-8.5)
        self.assertEqual(len(calls), 1)
        self.assertIn("Circuit breaker", calls[0][0])
        self.assertIn("8.5", calls[0][1])

    def test_alert_daily_loss_subject(self):
        fake_send, calls = self._capture_send()
        with patch("notifications.alerts._send", fake_send):
            from notifications.alerts import alert_daily_loss

            alert_daily_loss(-5.2)
        self.assertEqual(len(calls), 1)
        self.assertIn("loss", calls[0][0].lower())
        self.assertIn("5.2", calls[0][1])

    def test_alert_error_includes_context_and_error(self):
        fake_send, calls = self._capture_send()
        with patch("notifications.alerts._send", fake_send):
            from notifications.alerts import alert_error

            alert_error("main.run", "API timeout")
        self.assertEqual(len(calls), 1)
        self.assertIn("main.run", calls[0][0])
        self.assertIn("API timeout", calls[0][1])

import json
import os
import shutil
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import config as cfg
from analysis.weekly_review import _apply_config_changes, get_latest_review


class TestApplyConfigChanges(unittest.TestCase):
    """_apply_config_changes writes to runtime_config.json, not config.py."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.runtime_path = os.path.join(self.tmpdir, "runtime_config.json")
        self.patcher = patch("analysis.weekly_review._RUNTIME_CONFIG_PATH", self.runtime_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmpdir)

    def test_empty_list_returns_empty(self):
        result = _apply_config_changes([])
        self.assertEqual(result, [])

    def test_apply_when_runtime_config_missing_uses_empty_overrides(self):
        # runtime_config.json doesn't exist yet — should not raise
        result = _apply_config_changes(
            [{"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "test"}]
        )
        self.assertGreater(len(result), 0)
        self.assertIn(result[0]["status"], ("applied", "clamped", "unchanged"))

    def test_valid_change_returns_applied_status(self):
        result = _apply_config_changes(
            [{"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "test"}]
        )
        self.assertEqual(result[0]["status"], "applied")
        # File must NOT be written — auto-parameter modification is disabled
        self.assertFalse(os.path.exists(self.runtime_path))

    def test_unknown_parameter_is_rejected(self):
        result = _apply_config_changes(
            [{"parameter": "SUPER_SECRET_LEVER", "proposed_value": 99, "reason": "test"}]
        )
        self.assertEqual(result[0]["status"], "rejected")

    def test_value_above_max_is_clamped(self):
        result = _apply_config_changes(
            [{"parameter": "MIN_CONFIDENCE", "proposed_value": 99, "reason": "test"}]
        )
        self.assertEqual(result[0]["status"], "clamped")
        self.assertEqual(result[0]["new_value"], 9)

    def test_value_below_min_is_clamped(self):
        result = _apply_config_changes(
            [{"parameter": "TRAILING_STOP_PCT", "proposed_value": 0.5, "reason": "test"}]
        )
        self.assertEqual(result[0]["status"], "clamped")
        self.assertEqual(result[0]["new_value"], 2.0)

    def test_unchanged_value_not_written(self):
        result = _apply_config_changes(
            [
                {
                    "parameter": "MIN_CONFIDENCE",
                    "proposed_value": cfg.MIN_CONFIDENCE,
                    "reason": "no change",
                }
            ]
        )
        self.assertEqual(result[0]["status"], "unchanged")
        # File should not be created when nothing changed
        self.assertFalse(os.path.exists(self.runtime_path))

    def test_old_and_new_values_returned(self):
        result = _apply_config_changes(
            [
                {
                    "parameter": "MAX_HOLD_DAYS",
                    "proposed_value": cfg.MAX_HOLD_DAYS + 2,
                    "reason": "test",
                }
            ]
        )
        self.assertEqual(result[0]["old_value"], cfg.MAX_HOLD_DAYS)
        self.assertEqual(result[0]["new_value"], cfg.MAX_HOLD_DAYS + 2)

    def test_valid_change_with_existing_runtime_config_reads_current_overrides(self):
        # Line 55: json.load(f) fires when runtime config file exists with valid JSON
        import json as _json

        with open(self.runtime_path, "w") as f:
            _json.dump({"MIN_CONFIDENCE": 8}, f)
        result = _apply_config_changes(
            [
                {
                    "parameter": "MAX_HOLD_DAYS",
                    "proposed_value": cfg.MAX_HOLD_DAYS + 1,
                    "reason": "test",
                }
            ]
        )
        self.assertGreater(len(result), 0)
        self.assertIn(result[0]["status"], ("applied", "clamped", "unchanged"))

    def test_multiple_changes_validated_independently(self):
        result = _apply_config_changes(
            [
                {"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "test"},
                {"parameter": "MAX_HOLD_DAYS", "proposed_value": 5, "reason": "test"},
            ]
        )
        statuses = {r["parameter"]: r["status"] for r in result}
        self.assertIn(statuses["MIN_CONFIDENCE"], ("applied", "clamped"))
        self.assertIn(statuses["MAX_HOLD_DAYS"], ("applied", "clamped"))
        # File must NOT be written — auto-parameter modification is disabled
        self.assertFalse(os.path.exists(self.runtime_path))


class TestGetLatestReview(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patcher = patch("analysis.weekly_review.LOG_DIR", self.tmpdir)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmpdir)

    def _write_review(self, filename, lessons):
        import json

        path = os.path.join(self.tmpdir, filename)
        with open(path, "w") as f:
            json.dump({"lessons": lessons, "week_summary": "test"}, f)

    def test_returns_empty_list_when_no_reviews(self):
        self.assertEqual(get_latest_review(), [])

    def test_returns_lessons_from_single_review(self):
        lessons = [
            {"lesson": "lesson one", "applies_when": "ANY", "expiry": "2099-12-31"},
            {"lesson": "lesson two", "applies_when": "CHOPPY", "expiry": "2099-12-31"},
        ]
        self._write_review("weekly_review_2026-04-20.json", lessons)
        result = get_latest_review()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["lesson"], "lesson one")
        self.assertEqual(result[1]["lesson"], "lesson two")

    def test_returns_lessons_from_most_recent_review(self):
        self._write_review(
            "weekly_review_2026-04-13.json",
            [{"lesson": "old lesson", "applies_when": "ANY", "expiry": "2099-12-31"}],
        )
        self._write_review(
            "weekly_review_2026-04-20.json",
            [{"lesson": "new lesson", "applies_when": "ANY", "expiry": "2099-12-31"}],
        )
        result = get_latest_review()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["lesson"], "new lesson")

    def test_plain_string_lessons_wrapped_for_backward_compat(self):
        self._write_review("weekly_review_2026-04-20.json", ["plain string lesson"])
        result = get_latest_review()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["lesson"], "plain string lesson")
        self.assertEqual(result[0]["applies_when"], "ANY")

    def test_expired_lessons_excluded(self):
        lessons = [
            {"lesson": "active lesson", "applies_when": "ANY", "expiry": "2099-12-31"},
            {"lesson": "expired lesson", "applies_when": "ANY", "expiry": "2020-01-01"},
        ]
        self._write_review("weekly_review_2026-04-20.json", lessons)
        result = get_latest_review()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["lesson"], "active lesson")

    def test_regime_filter_applies(self):
        lessons = [
            {"lesson": "any regime lesson", "applies_when": "ANY", "expiry": "2099-12-31"},
            {"lesson": "choppy only lesson", "applies_when": "CHOPPY", "expiry": "2099-12-31"},
        ]
        self._write_review("weekly_review_2026-04-20.json", lessons)
        result = get_latest_review(regime="BULL_TRENDING")
        texts = [r["lesson"] for r in result]
        self.assertIn("any regime lesson", texts)
        self.assertNotIn("choppy only lesson", texts)

    def test_ignores_non_review_files(self):
        import json

        with open(os.path.join(self.tmpdir, "2026-04-20.json"), "w") as f:
            json.dump({"date": "2026-04-20"}, f)
        self.assertEqual(get_latest_review(), [])

    def test_returns_empty_on_malformed_file(self):
        path = os.path.join(self.tmpdir, "weekly_review_2026-04-20.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        self.assertEqual(get_latest_review(), [])


class TestCurrentParamValues(unittest.TestCase):
    def test_returns_all_four_params(self):
        from analysis.weekly_review import _current_param_values

        result = _current_param_values()
        for key in ["MIN_CONFIDENCE", "TRAILING_STOP_PCT", "PARTIAL_PROFIT_PCT", "MAX_HOLD_DAYS"]:
            self.assertIn(key, result)

    def test_values_are_numeric(self):
        from analysis.weekly_review import _current_param_values

        result = _current_param_values()
        for val in result.values():
            self.assertIsInstance(val, (int, float))

    def test_values_within_safe_bounds(self):
        from analysis.weekly_review import _SAFE_PARAMS, _current_param_values

        result = _current_param_values()
        for param, spec in _SAFE_PARAMS.items():
            self.assertGreaterEqual(result[param], spec["min"], f"{param} below minimum")
            self.assertLessEqual(result[param], spec["max"], f"{param} above maximum")


class TestRunWeeklyReview(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.runtime_path = os.path.join(self.tmpdir, "runtime_config.json")
        self.log_patcher = patch("analysis.weekly_review.LOG_DIR", self.tmpdir)
        self.runtime_patcher = patch(
            "analysis.weekly_review._RUNTIME_CONFIG_PATH", self.runtime_path
        )
        self.get_attribution_patcher = patch(
            "analysis.weekly_review.get_attribution", return_value={}
        )
        self.get_win_rates_patcher = patch("analysis.weekly_review.get_win_rates", return_value={})
        self.explog_patcher = patch(
            "analysis.weekly_review.EXPERIMENT_LOG_PATH",
            os.path.join(self.tmpdir, "EXPERIMENT_LOG.md"),
        )
        self.log_patcher.start()
        self.runtime_patcher.start()
        self.mock_get_attribution = self.get_attribution_patcher.start()
        self.get_win_rates_patcher.start()
        self.explog_patcher.start()
        self.addCleanup(self.log_patcher.stop)
        self.addCleanup(self.runtime_patcher.stop)
        self.addCleanup(self.get_attribution_patcher.stop)
        self.addCleanup(self.get_win_rates_patcher.stop)
        self.addCleanup(self.explog_patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmpdir)

    def _make_record(self, date_str, pnl=100.0):
        return {
            "date": date_str,
            "account_before": {"portfolio_value": 100_000},
            "account_after": {"portfolio_value": 100_000 + pnl},
            "daily_pnl": pnl,
            "trades_executed": [],
            "market_summary": "quiet",
        }

    def _mock_ai_response(self, review_dict):
        msg = MagicMock()
        msg.content = [MagicMock(text=json.dumps(review_dict))]
        return msg

    def test_returns_none_when_no_recent_records(self):
        from analysis.weekly_review import run_weekly_review

        with patch("analysis.weekly_review.load_history", return_value=[]):
            result = run_weekly_review()
        self.assertIsNone(result)

    def test_returns_review_dict_on_success(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "Good week",
            "what_worked": ["momentum"],
            "what_didnt": [],
            "lessons": ["Stay disciplined"],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIsNotNone(result)
        self.assertIn("week_summary", result)
        self.assertIn("proposed_changes", result)

    def test_edge_anatomy_failure_does_not_break_review(self):
        # telemetry must never break the weekly review — a raising build_edge_anatomy_lines is swallowed
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "ok",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
            patch(
                "analysis.weekly_review.build_edge_anatomy_lines",
                side_effect=RuntimeError("boom"),
            ),
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIsNotNone(result)  # review completes despite the telemetry failure

    def test_short_gate_telemetry_failure_does_not_break_review(self):
        # a raising build_short_gate_lines must also be swallowed
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "ok",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
            patch(
                "analysis.weekly_review.build_short_gate_lines",
                side_effect=RuntimeError("boom"),
            ),
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIsNotNone(result)

    def test_candidate_pipeline_telemetry_failure_does_not_break_review(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "ok",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
            patch(
                "analysis.weekly_review.build_candidate_lines",
                side_effect=RuntimeError("boom"),
            ),
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIsNotNone(result)

    def test_ledger_telemetry_failure_does_not_break_review(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "ok",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
            patch(
                "analysis.weekly_review.load_ledger",
                side_effect=RuntimeError("boom"),
            ),
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIsNotNone(result)

    def test_reconciliation_telemetry_failure_does_not_break_review(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "ok",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
            patch(
                "analysis.weekly_review.load_reconciliation_summary",
                side_effect=RuntimeError("boom"),
            ),
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIsNotNone(result)

    def test_experiment_monitoring_recorded_and_logged(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "Quiet week",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=2)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIn("experiment_monitoring", result)
        self.assertTrue(result["experiment_monitoring"])
        with open(os.path.join(self.tmpdir, "EXPERIMENT_LOG.md")) as f:
            self.assertIn("Monitoring only", f.read())

    def test_saves_review_file_to_log_dir(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            run_weekly_review()
        review_files = [f for f in os.listdir(self.tmpdir) if f.startswith("weekly_review_")]
        self.assertEqual(len(review_files), 1)

    def test_applies_config_changes_from_review(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [
                {"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "test"}
            ],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertTrue(
            any(c["status"] in ("applied", "clamped") for c in result["proposed_changes"])
        )

    def test_json_decode_error_returns_degraded_review_not_none(self):
        # When records exist but the AI response is unparseable, return a data-backed degraded
        # review (so the email reports real activity) — NOT None (which falls back to the stub
        # that falsely reports "no trade history available").
        from analysis.weekly_review import run_weekly_review

        bad_msg = MagicMock()
        bad_msg.content = [MagicMock(text="not valid json")]
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = bad_msg
            result = run_weekly_review()
        self.assertIsNotNone(result)
        self.assertTrue(result["review_degraded"])
        self.assertIn("trade(s) executed", result["week_summary"])
        # Monitoring telemetry must survive an AI failure (the 2026-06-28 dropped-entry bug):
        # it's attached to the degraded review AND written to EXPERIMENT_LOG.md.
        self.assertTrue(result["experiment_monitoring"])
        with open(os.path.join(self.tmpdir, "EXPERIMENT_LOG.md")) as f:
            self.assertIn("Monitoring only", f.read())

    def test_general_exception_returns_degraded_review_not_none(self):
        from analysis.weekly_review import run_weekly_review

        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.side_effect = RuntimeError("API crashed")
            result = run_weekly_review()
        self.assertIsNotNone(result)
        self.assertTrue(result["review_degraded"])

    def test_parses_markdown_wrapped_json(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "Wrapped JSON week",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        wrapped = f"```json\n{json.dumps(fake_review)}\n```"
        msg = MagicMock()
        msg.content = [MagicMock(text=wrapped)]
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = msg
            result = run_weekly_review()
        self.assertIsNotNone(result)
        self.assertEqual(result["week_summary"], "Wrapped JSON week")

    def test_backtick_wrapped_without_json_label(self):
        """Line 267->269: raw starts with ``` but NOT 'json' — skips the [4:] strip."""
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "Backtick no-label week",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        wrapped = f"```\n{json.dumps(fake_review)}\n```"
        msg = MagicMock()
        msg.content = [MagicMock(text=wrapped)]
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = msg
            result = run_weekly_review()
        self.assertIsNotNone(result)
        self.assertEqual(result["week_summary"], "Backtick no-label week")

    def test_logs_proposed_changes_with_applied_or_clamped_status(self):
        """Line 285->284: change status in ('applied','clamped') logs the change."""
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [
                {"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "test"},
                {"parameter": "SUPER_SECRET", "proposed_value": 99, "reason": "rejected"},
            ],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIsNotNone(result)
        statuses = {c["parameter"]: c["status"] for c in result["proposed_changes"]}
        self.assertIn("MIN_CONFIDENCE", statuses)
        self.assertIn("SUPER_SECRET", statuses)
        self.assertEqual(statuses["SUPER_SECRET"], "rejected")

    def test_week_attribution_key_in_return_dict(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        self.mock_get_attribution.return_value = {"total_trades": 3, "by_signal": {}}
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            result = run_weekly_review()
        self.assertIsNotNone(result)
        self.assertIn("week_attribution", result)
        self.assertEqual(result["week_attribution"]["total_trades"], 3)

    def test_get_attribution_called_with_days_7(self):
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_anthropic.return_value.messages.create.return_value = self._mock_ai_response(
                fake_review
            )
            run_weekly_review()
        self.mock_get_attribution.assert_called_once_with(7)

    def test_attribution_block_injected_into_prompt(self):
        """When get_attribution returns data, Claude's prompt includes the attribution block."""
        from analysis.weekly_review import run_weekly_review

        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        self.mock_get_attribution.return_value = {
            "total_trades": 5,
            "by_signal": {"momentum": {"trades": 5, "win_rate": 60.0, "avg_return_pct": 1.5}},
            "by_regime": {},
            "by_sector": {},
            "best_signal": "momentum",
            "worst_signal": None,
        }
        with (
            patch(
                "analysis.weekly_review.load_history",
                return_value=[self._make_record((date.today() - timedelta(days=3)).isoformat())],
            ),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = mock_anthropic.return_value
            mock_client.messages.create.return_value = self._mock_ai_response(fake_review)
            run_weekly_review()
        call_args = mock_client.messages.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        self.assertIn("7-DAY TRADE ATTRIBUTION", prompt_text)
        self.assertIn("momentum", prompt_text)

    def test_trade_summary_enriched_with_signal_and_confidence(self):
        """BUY trades with pipe-separated detail appear with signal+confidence in the prompt."""
        from analysis.weekly_review import run_weekly_review

        record = self._make_record((date.today() - timedelta(days=3)).isoformat())
        record["trades_executed"] = [
            {
                "symbol": "AAPL",
                "action": "BUY",
                "detail": "$500.00 | momentum | confidence=8",
            }
        ]
        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch("analysis.weekly_review.load_history", return_value=[record]),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = mock_anthropic.return_value
            mock_client.messages.create.return_value = self._mock_ai_response(fake_review)
            run_weekly_review()
        prompt_text = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        self.assertIn('"signal": "momentum"', prompt_text)
        self.assertIn('"confidence": 8', prompt_text)

    def test_buy_no_signal_omits_signal_key(self):
        """BUY trade where _parse_detail returns sig=None: signal key absent, no KeyError."""
        from analysis.weekly_review import run_weekly_review

        record = self._make_record((date.today() - timedelta(days=3)).isoformat())
        record["trades_executed"] = [{"symbol": "AAPL", "action": "BUY", "detail": "$500.00"}]
        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch("analysis.weekly_review.load_history", return_value=[record]),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = mock_anthropic.return_value
            mock_client.messages.create.return_value = self._mock_ai_response(fake_review)
            result = run_weekly_review()
        self.assertIsNotNone(result)
        prompt_text = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        self.assertNotIn('"signal"', prompt_text)

    def test_buy_no_confidence_omits_confidence_key(self):
        """BUY trade where _parse_detail returns conf=None: confidence key absent."""
        from analysis.weekly_review import run_weekly_review

        record = self._make_record((date.today() - timedelta(days=3)).isoformat())
        record["trades_executed"] = [
            {"symbol": "AAPL", "action": "BUY", "detail": "$500.00 | breakout"}
        ]
        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch("analysis.weekly_review.load_history", return_value=[record]),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = mock_anthropic.return_value
            mock_client.messages.create.return_value = self._mock_ai_response(fake_review)
            result = run_weekly_review()
        self.assertIsNotNone(result)
        prompt_text = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        self.assertNotIn('"confidence"', prompt_text)

    def test_sell_trade_stores_exit_reason(self):
        """Non-BUY trade: detail stored as exit_reason in prompt."""
        from analysis.weekly_review import run_weekly_review

        record = self._make_record((date.today() - timedelta(days=3)).isoformat())
        record["trades_executed"] = [
            {"symbol": "AAPL", "action": "SELL", "detail": "trailing stop hit"}
        ]
        fake_review = {
            "week_summary": "OK",
            "what_worked": [],
            "what_didnt": [],
            "lessons": [],
            "config_changes": [],
        }
        with (
            patch("analysis.weekly_review.load_history", return_value=[record]),
            patch("analysis.weekly_review.anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = mock_anthropic.return_value
            mock_client.messages.create.return_value = self._mock_ai_response(fake_review)
            result = run_weekly_review()
        self.assertIsNotNone(result)
        prompt_text = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        self.assertIn("trailing stop hit", prompt_text)


class TestParseDetail(unittest.TestCase):
    def setUp(self):
        from analysis.weekly_review import _parse_detail

        self._parse_detail = _parse_detail

    def test_standard_buy_extracts_signal_and_confidence(self):
        sig, conf = self._parse_detail("$500.00 | momentum | confidence=8")
        self.assertEqual(sig, "momentum")
        self.assertEqual(conf, 8)

    def test_confidence_only_string(self):
        sig, conf = self._parse_detail("confidence=7")
        self.assertIsNone(sig)
        self.assertEqual(conf, 7)

    def test_signal_only_no_confidence(self):
        sig, conf = self._parse_detail("$200.00 | mean_reversion")
        self.assertEqual(sig, "mean_reversion")
        self.assertIsNone(conf)

    def test_dry_run_label_excluded_from_signal(self):
        sig, conf = self._parse_detail("$500.00 | momentum | confidence=8 | dry run")
        self.assertEqual(sig, "momentum")
        self.assertEqual(conf, 8)

    def test_kelly_label_excluded_from_signal(self):
        sig, conf = self._parse_detail("$300.00 | momentum | Kelly=0.5 | confidence=7")
        self.assertEqual(sig, "momentum")
        self.assertEqual(conf, 7)

    def test_empty_string_returns_none_none(self):
        sig, conf = self._parse_detail("")
        self.assertIsNone(sig)
        self.assertIsNone(conf)

    def test_price_only_part_excluded_from_signal(self):
        # A bare "$price" part is excluded; only signal-named parts are kept
        sig, conf = self._parse_detail("$198.50")
        self.assertIsNone(sig)
        self.assertIsNone(conf)

    def test_invalid_confidence_value_skipped(self):
        sig, conf = self._parse_detail("$500.00 | momentum | confidence=abc")
        self.assertEqual(sig, "momentum")
        self.assertIsNone(conf)

    def test_dollar_prefix_part_excluded_from_signal(self):
        sig, conf = self._parse_detail("$1000.00 | breakout | confidence=9")
        self.assertEqual(sig, "breakout")
        self.assertEqual(conf, 9)


class TestMapCandidateEvidence(unittest.TestCase):
    def test_maps_present_evidence(self):
        from analysis.weekly_review import _map_candidate_evidence
        from experiment.candidate_registry import default_candidates

        conf = {"conf=8": (60, 0.30)}
        short = {"guidance_downgrade": [250, 1.6, 65.0]}
        out = {
            c.id: (n, e)
            for c, n, e in _map_candidate_evidence(default_candidates(), conf, short, {}, [])
        }
        self.assertEqual(out["min_confidence_7_to_8"], (60, 0.30))
        self.assertEqual(out["ungate_guidance_downgrade_shorts"], (250, 1.6))

    def test_absent_evidence_maps_to_none(self):
        from analysis.weekly_review import _map_candidate_evidence
        from experiment.candidate_registry import default_candidates

        out = {
            c.id: (n, e)
            for c, n, e in _map_candidate_evidence(default_candidates(), {}, {}, {}, [])
        }
        self.assertTrue(all(v == (None, None) for v in out.values()))

    def test_mined_candidate_scored_from_research_signal(self):
        from analysis.weekly_review import _map_candidate_evidence
        from experiment.candidate_registry import Candidate
        from experiment.research_signals import ResearchSignal

        cand = Candidate(
            id="mined_rsi_14_ge",
            hypothesis="h",
            action="a",
            metric="R",
            min_n=60,
            min_effect=0.15,
            source="mined",
        )
        sig = ResearchSignal("mined_rsi_14_ge", "rsi_14", ">=", 60.0, "long", created="2026-07-01")
        obs = [
            {"date": "2026-07-10", "features": {"rsi_14": 70}, "outcomes": {"forward_r_5d": 2.0}},
            {"date": "2026-07-10", "features": {"rsi_14": 30}, "outcomes": {"forward_r_5d": 0.0}},
        ]
        _c, n, effect = _map_candidate_evidence([cand], {}, {}, {"mined_rsi_14_ge": sig}, obs)[0]
        self.assertEqual(n, 1)
        self.assertAlmostEqual(effect, 1.0)  # fired mean 2.0 - field mean 1.0

    def test_mined_candidate_without_research_signal_is_none(self):
        from analysis.weekly_review import _map_candidate_evidence
        from experiment.candidate_registry import Candidate

        cand = Candidate(
            id="mined_orphan",
            hypothesis="h",
            action="a",
            metric="R",
            min_n=60,
            min_effect=0.15,
            source="mined",
        )
        out = _map_candidate_evidence([cand], {}, {}, {}, [])  # no research signal registered
        self.assertEqual(out[0][1:], (None, None))

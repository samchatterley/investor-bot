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

    def _read(self) -> dict:
        if not os.path.exists(self.runtime_path):
            return {}
        with open(self.runtime_path) as f:
            return json.load(f)

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
        self.log_patcher.start()
        self.runtime_patcher.start()
        self.addCleanup(self.log_patcher.stop)
        self.addCleanup(self.runtime_patcher.stop)
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

    def test_returns_none_on_json_decode_error(self):
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
        self.assertIsNone(result)

    def test_returns_none_on_general_exception(self):
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
        self.assertIsNone(result)

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

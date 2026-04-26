import os
import shutil
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from analysis.weekly_review import _apply_config_changes


_SAMPLE_CONFIG = textwrap.dedent("""\
    MIN_CONFIDENCE = 7
    TRAILING_STOP_PCT = 4.0
    PARTIAL_PROFIT_PCT = 8.0
    MAX_HOLD_DAYS = 3
""")


class TestApplyConfigChanges(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "config.py")
        with open(self.config_path, "w") as f:
            f.write(_SAMPLE_CONFIG)
        self.patcher = patch("analysis.weekly_review._CONFIG_PATH", self.config_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmpdir)

    def _read(self):
        with open(self.config_path) as f:
            return f.read()

    def test_empty_list_returns_empty(self):
        result = _apply_config_changes([])
        self.assertEqual(result, [])

    def test_valid_change_is_applied(self):
        result = _apply_config_changes([
            {"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "test"}
        ])
        self.assertEqual(result[0]["status"], "applied")
        self.assertIn("MIN_CONFIDENCE = 8", self._read())

    def test_unknown_parameter_is_rejected(self):
        result = _apply_config_changes([
            {"parameter": "SUPER_SECRET_LEVER", "proposed_value": 99, "reason": "test"}
        ])
        self.assertEqual(result[0]["status"], "rejected")

    def test_value_above_max_is_clamped(self):
        result = _apply_config_changes([
            {"parameter": "MIN_CONFIDENCE", "proposed_value": 99, "reason": "test"}
        ])
        self.assertEqual(result[0]["status"], "clamped")
        self.assertEqual(result[0]["new_value"], 9)

    def test_value_below_min_is_clamped(self):
        result = _apply_config_changes([
            {"parameter": "TRAILING_STOP_PCT", "proposed_value": 0.5, "reason": "test"}
        ])
        self.assertEqual(result[0]["status"], "clamped")
        self.assertEqual(result[0]["new_value"], 2.0)

    def test_unchanged_value_not_written(self):
        original = self._read()
        result = _apply_config_changes([
            {"parameter": "MIN_CONFIDENCE", "proposed_value": 7, "reason": "no change"}
        ])
        self.assertEqual(result[0]["status"], "unchanged")
        self.assertEqual(self._read(), original)

    def test_old_and_new_values_returned(self):
        result = _apply_config_changes([
            {"parameter": "MAX_HOLD_DAYS", "proposed_value": 5, "reason": "test"}
        ])
        self.assertEqual(result[0]["old_value"], 3)
        self.assertEqual(result[0]["new_value"], 5)

    def test_multiple_changes_applied_independently(self):
        result = _apply_config_changes([
            {"parameter": "MIN_CONFIDENCE", "proposed_value": 8, "reason": "test"},
            {"parameter": "MAX_HOLD_DAYS", "proposed_value": 5, "reason": "test"},
        ])
        statuses = {r["parameter"]: r["status"] for r in result}
        self.assertEqual(statuses["MIN_CONFIDENCE"], "applied")
        self.assertEqual(statuses["MAX_HOLD_DAYS"], "applied")
        content = self._read()
        self.assertIn("MIN_CONFIDENCE = 8", content)
        self.assertIn("MAX_HOLD_DAYS = 5", content)

import json
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

import config


class TestValidate(unittest.TestCase):

    def _patch(self, attr, value):
        original = getattr(config, attr)
        setattr(config, attr, value)
        self.addCleanup(setattr, config, attr, original)

    def test_valid_config_passes(self):
        config.validate()

    def test_max_position_pct_above_one_fails(self):
        self._patch("MAX_POSITION_PCT", 1.5)
        with self.assertRaises(ValueError):
            config.validate()

    def test_max_position_pct_zero_fails(self):
        self._patch("MAX_POSITION_PCT", 0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_cash_reserve_zero_fails(self):
        self._patch("CASH_RESERVE_PCT", 0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_cash_reserve_above_one_fails(self):
        self._patch("CASH_RESERVE_PCT", 1.1)
        with self.assertRaises(ValueError):
            config.validate()

    def test_min_confidence_above_ten_fails(self):
        self._patch("MIN_CONFIDENCE", 11)
        with self.assertRaises(ValueError):
            config.validate()

    def test_min_confidence_zero_fails(self):
        self._patch("MIN_CONFIDENCE", 0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_trailing_stop_zero_fails(self):
        self._patch("TRAILING_STOP_PCT", 0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_max_hold_days_zero_fails(self):
        self._patch("MAX_HOLD_DAYS", 0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_max_positions_zero_fails(self):
        self._patch("MAX_POSITIONS", 0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_max_single_order_zero_fails(self):
        self._patch("MAX_SINGLE_ORDER_USD", 0.0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_error_message_lists_all_failures(self):
        self._patch("MAX_POSITION_PCT", 2.0)
        self._patch("MIN_CONFIDENCE", 0)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        msg = str(ctx.exception)
        self.assertIn("MAX_POSITION_PCT", msg)
        self.assertIn("MIN_CONFIDENCE", msg)


class TestTodayEt(unittest.TestCase):

    def test_returns_date_object(self):
        self.assertIsInstance(config.today_et(), date)

    def test_date_year_is_plausible(self):
        self.assertGreater(config.today_et().year, 2020)


# ── Runtime override tests ────────────────────────────────────────────────────

class TestRuntimeOverrideConstants(unittest.TestCase):
    """Structural checks on the allowlist and bounds table."""

    def test_all_override_keys_have_bounds_entry(self):
        for key in config.RUNTIME_OVERRIDE_KEYS:
            self.assertIn(key, config.RUNTIME_OVERRIDE_BOUNDS,
                          f"{key} is in RUNTIME_OVERRIDE_KEYS but missing from RUNTIME_OVERRIDE_BOUNDS")

    def test_all_bounds_entries_are_in_allowlist(self):
        for key in config.RUNTIME_OVERRIDE_BOUNDS:
            self.assertIn(key, config.RUNTIME_OVERRIDE_KEYS,
                          f"{key} is in RUNTIME_OVERRIDE_BOUNDS but missing from RUNTIME_OVERRIDE_KEYS")

    def test_bounds_entries_have_correct_shape(self):
        for key, bounds in config.RUNTIME_OVERRIDE_BOUNDS.items():
            self.assertEqual(len(bounds), 3, f"{key}: bounds must be (type, min, max)")
            typ, lo, hi = bounds
            self.assertIn(typ, (int, float), f"{key}: type must be int or float")
            self.assertLess(lo, hi, f"{key}: min must be less than max")

    def test_sensitive_keys_not_in_allowlist(self):
        for key in ("IS_PAPER", "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
                    "ANTHROPIC_API_KEY", "STOCK_UNIVERSE", "HALT_FILE",
                    "ALPACA_BASE_URL", "LOG_DIR"):
            self.assertNotIn(key, config.RUNTIME_OVERRIDE_KEYS,
                             f"Sensitive key {key} must not be runtime-overrideable")


class TestRuntimeOverrideBase(unittest.TestCase):
    """Shared setup: writes a temp config file and calls _load_runtime_overrides()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "runtime_config.json")
        self._originals = {k: getattr(config, k) for k in config.RUNTIME_OVERRIDE_KEYS}
        self.path_patcher = patch.object(config, "_RUNTIME_CONFIG_PATH", self.config_path)
        self.path_patcher.start()

    def tearDown(self):
        self.path_patcher.stop()
        for key, val in self._originals.items():
            setattr(config, key, val)

    def _write(self, data: dict):
        with open(self.config_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        config._load_runtime_overrides()


class TestRuntimeOverrideApply(TestRuntimeOverrideBase):

    def test_valid_int_key_applied(self):
        self._write({"MIN_CONFIDENCE": 9})
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, 9)

    def test_valid_float_key_applied(self):
        self._write({"TRAILING_STOP_PCT": 5.0})
        self._load()
        self.assertAlmostEqual(config.TRAILING_STOP_PCT, 5.0)

    def test_int_key_at_minimum_boundary_applied(self):
        lo = config.RUNTIME_OVERRIDE_BOUNDS["MIN_CONFIDENCE"][1]
        self._write({"MIN_CONFIDENCE": lo})
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, lo)

    def test_int_key_at_maximum_boundary_applied(self):
        hi = config.RUNTIME_OVERRIDE_BOUNDS["MIN_CONFIDENCE"][2]
        self._write({"MIN_CONFIDENCE": hi})
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, hi)

    def test_float_key_at_minimum_boundary_applied(self):
        lo = config.RUNTIME_OVERRIDE_BOUNDS["TRAILING_STOP_PCT"][1]
        self._write({"TRAILING_STOP_PCT": lo})
        self._load()
        self.assertAlmostEqual(config.TRAILING_STOP_PCT, lo)

    def test_float_key_at_maximum_boundary_applied(self):
        hi = config.RUNTIME_OVERRIDE_BOUNDS["TRAILING_STOP_PCT"][2]
        self._write({"TRAILING_STOP_PCT": hi})
        self._load()
        self.assertAlmostEqual(config.TRAILING_STOP_PCT, hi)

    def test_multiple_valid_keys_all_applied(self):
        self._write({"MIN_CONFIDENCE": 9, "MAX_HOLD_DAYS": 5, "MAX_ORDERS_PER_RUN": 2})
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, 9)
        self.assertEqual(config.MAX_HOLD_DAYS, 5)
        self.assertEqual(config.MAX_ORDERS_PER_RUN, 2)

    def test_int_coercion_from_whole_number_float(self):
        # JSON numbers may come in as floats; 9.0 → 9 is safe
        self._write({"MIN_CONFIDENCE": 9.0})
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, 9)
        self.assertIsInstance(config.MIN_CONFIDENCE, int)

    def test_float_coercion_from_int(self):
        # int 5 coerced to float 5.0 for a float field
        self._write({"TRAILING_STOP_PCT": 5})
        self._load()
        self.assertAlmostEqual(config.TRAILING_STOP_PCT, 5.0)


class TestRuntimeOverrideReject(TestRuntimeOverrideBase):

    def test_unknown_key_not_applied(self):
        original = config.IS_PAPER
        self._write({"IS_PAPER": False})
        self._load()
        self.assertEqual(config.IS_PAPER, original)

    def test_sensitive_api_key_not_applied(self):
        original = config.ANTHROPIC_API_KEY
        self._write({"ANTHROPIC_API_KEY": "hacked"})
        self._load()
        self.assertEqual(config.ANTHROPIC_API_KEY, original)

    def test_stock_universe_not_overrideable(self):
        original = list(config.STOCK_UNIVERSE)
        self._write({"STOCK_UNIVERSE": ["XYZ"]})
        self._load()
        self.assertEqual(config.STOCK_UNIVERSE, original)

    def test_value_above_max_rejected(self):
        hi = config.RUNTIME_OVERRIDE_BOUNDS["MIN_CONFIDENCE"][2]
        self._write({"MIN_CONFIDENCE": hi + 1})
        original = self._originals["MIN_CONFIDENCE"]
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, original)

    def test_value_below_min_rejected(self):
        lo = config.RUNTIME_OVERRIDE_BOUNDS["MIN_CONFIDENCE"][1]
        self._write({"MIN_CONFIDENCE": lo - 1})
        original = self._originals["MIN_CONFIDENCE"]
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, original)

    def test_string_value_for_int_key_rejected(self):
        original = self._originals["MIN_CONFIDENCE"]
        self._write({"MIN_CONFIDENCE": "high"})
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, original)

    def test_none_value_rejected(self):
        original = self._originals["MIN_CONFIDENCE"]
        self._write({"MIN_CONFIDENCE": None})
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, original)

    def test_mixed_valid_and_invalid_partial_apply(self):
        original_conf = self._originals["MIN_CONFIDENCE"]
        self._write({"MIN_CONFIDENCE": 99, "MAX_HOLD_DAYS": 3})
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, original_conf)  # rejected
        self.assertEqual(config.MAX_HOLD_DAYS, 3)               # applied

    def test_max_position_pct_not_runtime_overridable(self):
        # MAX_POSITION_PCT is a deprecated legacy constant — removed from RUNTIME_OVERRIDE_KEYS
        # to prevent accidental 45% position sizing via self-modification.
        original = config.MAX_POSITION_PCT
        self._write({"MAX_POSITION_PCT": 0.25})
        self._load()
        self.assertEqual(config.MAX_POSITION_PCT, original)


class TestRuntimeOverrideEdgeCases(TestRuntimeOverrideBase):

    def test_missing_file_no_change_no_crash(self):
        # No file written — _RUNTIME_CONFIG_PATH does not exist
        original_conf = self._originals["MIN_CONFIDENCE"]
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, original_conf)

    def test_malformed_json_no_change_no_crash(self):
        with open(self.config_path, "w") as f:
            f.write("not json {{{")
        original_conf = self._originals["MIN_CONFIDENCE"]
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, original_conf)

    def test_empty_json_object_no_change(self):
        self._write({})
        original_conf = self._originals["MIN_CONFIDENCE"]
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, original_conf)

    def test_repeated_loads_are_idempotent(self):
        self._write({"MIN_CONFIDENCE": 8})
        self._load()
        self._load()
        self.assertEqual(config.MIN_CONFIDENCE, 8)


class TestRuntimeOverrideAuditEvents(TestRuntimeOverrideBase):

    def test_applied_key_emits_applied_event(self):
        self._write({"MIN_CONFIDENCE": 9})
        with patch.object(config, "_audit_config_event") as mock_audit:
            self._load()
        mock_audit.assert_called_once_with(
            "CONFIG_OVERRIDE_APPLIED", {"key": "MIN_CONFIDENCE", "value": 9}
        )

    def test_unknown_key_emits_rejected_event(self):
        self._write({"IS_PAPER": False})
        with patch.object(config, "_audit_config_event") as mock_audit:
            self._load()
        calls = mock_audit.call_args_list
        self.assertEqual(len(calls), 1)
        event_type, payload = calls[0].args
        self.assertEqual(event_type, "CONFIG_OVERRIDE_REJECTED")
        self.assertEqual(payload["key"], "IS_PAPER")
        self.assertIn("allowlist", payload["reason"])

    def test_out_of_bounds_emits_rejected_event(self):
        self._write({"MIN_CONFIDENCE": 99})
        with patch.object(config, "_audit_config_event") as mock_audit:
            self._load()
        calls = mock_audit.call_args_list
        self.assertEqual(len(calls), 1)
        event_type, payload = calls[0].args
        self.assertEqual(event_type, "CONFIG_OVERRIDE_REJECTED")
        self.assertIn("bounds", payload["reason"])

    def test_bad_type_emits_rejected_event(self):
        self._write({"MIN_CONFIDENCE": "high"})
        with patch.object(config, "_audit_config_event") as mock_audit:
            self._load()
        calls = mock_audit.call_args_list
        event_type, payload = calls[0].args
        self.assertEqual(event_type, "CONFIG_OVERRIDE_REJECTED")
        self.assertIn("coerce", payload["reason"])

    def test_mixed_batch_emits_correct_events(self):
        self._write({"MIN_CONFIDENCE": 8, "IS_PAPER": False, "MAX_HOLD_DAYS": 99})
        with patch.object(config, "_audit_config_event") as mock_audit:
            self._load()
        event_types = [c.args[0] for c in mock_audit.call_args_list]
        self.assertIn("CONFIG_OVERRIDE_APPLIED", event_types)
        self.assertEqual(event_types.count("CONFIG_OVERRIDE_REJECTED"), 2)

    def test_no_file_emits_no_events(self):
        with patch.object(config, "_audit_config_event") as mock_audit:
            self._load()
        mock_audit.assert_not_called()

    def test_empty_file_emits_no_events(self):
        self._write({})
        with patch.object(config, "_audit_config_event") as mock_audit:
            self._load()
        mock_audit.assert_not_called()

    def test_audit_failure_does_not_crash_startup(self):
        self._write({"MIN_CONFIDENCE": 9})
        with patch.object(config, "_audit_config_event", side_effect=RuntimeError("db down")):
            # Must not raise
            self._load()
        self.assertEqual(config.MIN_CONFIDENCE, 9)

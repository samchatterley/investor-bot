"""Tests for core/deps.py — RunConfig and TradingDeps."""

import dataclasses
import unittest
from dataclasses import fields
from unittest.mock import MagicMock

from conftest import _default_run_config, make_test_deps
from core.deps import RunConfig, TradingDeps


class TestRunConfig(unittest.TestCase):
    def test_from_config_returns_instance(self):
        rc = RunConfig.from_config()
        self.assertIsInstance(rc, RunConfig)

    def test_from_config_is_frozen(self):
        rc = RunConfig.from_config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rc.is_paper = False  # type: ignore[misc]

    def test_types_are_correct(self):
        rc = RunConfig.from_config()
        self.assertIsInstance(rc.is_paper, bool)
        self.assertIsInstance(rc.max_positions, int)
        self.assertIsInstance(rc.trailing_stop_pct, float)
        self.assertIsInstance(rc.signal_max_hold_days, dict)
        self.assertIsInstance(rc.small_account_mode, bool)

    def test_signal_max_hold_days_is_a_copy(self):
        import config

        rc = RunConfig.from_config()
        self.assertIsNot(rc.signal_max_hold_days, config.SIGNAL_MAX_HOLD_DAYS)

    def test_all_fields_populated(self):
        rc = RunConfig.from_config()
        for f in fields(rc):
            self.assertIsNotNone(getattr(rc, f.name), f"Field {f.name!r} is None")


class TestTradingDepsProduction(unittest.TestCase):
    def test_returns_trading_deps(self):
        deps = TradingDeps.production()
        self.assertIsInstance(deps, TradingDeps)

    def test_run_config_is_run_config(self):
        deps = TradingDeps.production()
        self.assertIsInstance(deps.run_config, RunConfig)

    def test_all_module_fields_are_not_none(self):
        deps = TradingDeps.production()
        for f in fields(deps):
            self.assertIsNotNone(getattr(deps, f.name), f"Field {f.name!r} is None")


class TestMakeTestDeps(unittest.TestCase):
    def test_returns_trading_deps(self):
        deps = make_test_deps()
        self.assertIsInstance(deps, TradingDeps)

    def test_module_fields_are_mocks(self):
        deps = make_test_deps()
        for f in fields(TradingDeps):
            if f.name == "run_config":
                continue
            self.assertIsInstance(
                getattr(deps, f.name), MagicMock, f"{f.name!r} is not a MagicMock"
            )

    def test_run_config_has_typed_defaults(self):
        deps = make_test_deps()
        self.assertIsInstance(deps.run_config, RunConfig)
        self.assertTrue(deps.run_config.is_paper)
        self.assertEqual(deps.run_config.max_positions, 5)
        self.assertEqual(deps.run_config.min_confidence, 7)

    def test_override_module_field(self):
        custom = MagicMock()
        deps = make_test_deps(trader=custom)
        self.assertIs(deps.trader, custom)

    def test_override_run_config(self):
        rc = RunConfig.from_config()
        deps = make_test_deps(run_config=rc)
        self.assertIs(deps.run_config, rc)


class TestDefaultRunConfig(unittest.TestCase):
    def test_returns_run_config(self):
        rc = _default_run_config()
        self.assertIsInstance(rc, RunConfig)

    def test_override_single_field(self):
        rc = _default_run_config(min_confidence=9)
        self.assertEqual(rc.min_confidence, 9)

    def test_is_frozen(self):
        rc = _default_run_config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rc.min_confidence = 9  # type: ignore[misc]

"""Shared test utilities for the InvestorBot test suite."""

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

import experiment.collection as _collection
from core.deps import RunConfig, TradingDeps


@pytest.fixture(autouse=True)
def _isolate_experiment_observations(tmp_path, monkeypatch):
    """Never let the test suite write to the live experiment observations log.

    main._run_ai_phase calls log_run_observations with the default path, which resolves to
    experiment.collection.OBSERVATIONS_PATH (logs/experiment_observations.jsonl) at call time.
    Redirect that constant to a per-test temp file so exercising the decision loop in tests cannot
    pollute the real dataset the bot collects in production.
    """
    monkeypatch.setattr(_collection, "OBSERVATIONS_PATH", str(tmp_path / "experiment_obs.jsonl"))


@pytest.fixture(autouse=True)
def _isolate_shadow_catalyst_log(tmp_path, monkeypatch):
    """Never let the test suite write to the live shadow-catalyst-shorts log.

    main._run_inner calls shadow_catalyst_shorts.capture() (fail-safe instrumentation), which appends
    to SHADOW_LOG_PATH. Redirect it to a per-test temp file so exercising the run loop in tests cannot
    pollute the real measurement log.
    """
    import analysis.shadow_catalyst_shorts as _shadow

    monkeypatch.setattr(_shadow, "SHADOW_LOG_PATH", str(tmp_path / "shadow_catalyst_shorts.jsonl"))


@pytest.fixture(autouse=True)
def _isolate_shadow_popper_log(tmp_path, monkeypatch):
    """Never let the test suite write to the live shadow-popper-shorts log (mirror of the
    catalyst-shadow isolation above; main._run_inner calls shadow_popper_shorts.capture)."""
    import analysis.shadow_popper_shorts as _popper

    monkeypatch.setattr(_popper, "SHADOW_LOG_PATH", str(tmp_path / "shadow_popper_shorts.jsonl"))


@pytest.fixture(autouse=True)
def _pin_index_hedge_disabled(monkeypatch):
    """Keep tests deterministic regardless of the operator's .env. config.load_dotenv() runs at
    import, so an enabled deployment toggle (INDEX_HEDGE_ENABLED=true) would otherwise fire the live
    index-hedge order path into unmocked decision-loop tests. Pin it off here; hedge-specific tests
    opt in by patching config.INDEX_HEDGE_ENABLED=True in their own context.
    """
    import config

    monkeypatch.setattr(config, "INDEX_HEDGE_ENABLED", False)


def _default_run_config(**overrides) -> RunConfig:
    defaults: dict = {
        "bear_market_spy_threshold": 0.8,
        "cash_reserve_pct": 0.05,
        "earnings_warning_days": 2,
        "is_paper": True,
        "lookback_days": 30,
        "max_daily_loss_usd": 0.0,
        "max_daily_notional_usd": 5_000.0,
        "max_deployed_usd": 50_000.0,
        "max_experiment_drawdown_usd": 0.0,
        "max_hold_days": 5,
        "max_orders_per_run": 3,
        "max_positions": 5,
        "max_price_usd": 0.0,
        "max_sector_positions": 2,
        "max_short_hedge_ratio": 0.5,
        "max_short_hold_days": 3,
        "max_short_positions": 3,
        "max_single_order_usd": 1_000.0,
        "min_confidence": 7,
        "min_price_usd": 0.0,
        "short_size_scale": 0.5,
        "signal_max_hold_days": {},
        "small_account_mode": False,
        "top_movers_count": 20,
        "trailing_stop_pct": 5.0,
    }
    return RunConfig(**{**defaults, **overrides})


def make_test_deps(**overrides) -> TradingDeps:
    """Return a TradingDeps with all module fields as MagicMocks.

    Keyword args override specific fields. run_config defaults to a
    safe paper-trading RunConfig unless overridden.
    """
    deps_fields = {f.name: MagicMock() for f in fields(TradingDeps) if f.name != "run_config"}
    deps_fields["run_config"] = _default_run_config()
    deps_fields.update(overrides)
    return TradingDeps(**deps_fields)

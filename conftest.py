"""Shared test utilities for the InvestorBot test suite."""

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

import experiment.collection as _collection
from core.deps import RunConfig, TradingDeps


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """No test may touch the live logs/investorbot.db.

    Point the DB at a per-test throwaway file (and reset the init flag / no-op the legacy-JSON
    import) so ANY get_db()/init_db() — even from a test that forgot to patch _DB_PATH — hits a
    disposable DB, never the running bot's database. This is the root-cause fix for the order_intents
    pollution: tests calling the real place_buy_order (e.g. test_live_safety with "SOFI"/"AAPL")
    wrote ib-<symbol>-BUY-<today> intents straight into the live DB on every suite run. Tests that
    already patch _DB_PATH still win (their patch nests over this one)."""
    import utils.db as _db

    monkeypatch.setattr(_db, "_DB_PATH", str(tmp_path / "investorbot_test.db"))
    monkeypatch.setattr(_db, "_initialized", False)
    monkeypatch.setattr(_db, "_migrate_json_state", lambda: None)
    _db.init_db()


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
def _isolate_macro_gate_shadow(tmp_path, monkeypatch):
    """Never let the test suite write to the live macro-gate-shadow log (mirror of the shadow-log
    isolation above; main._execute_buy_phase calls macro_gate_shadow.capture on a macro skip)."""
    import analysis.macro_gate_shadow as _mg

    monkeypatch.setattr(_mg, "SHADOW_LOG_PATH", str(tmp_path / "macro_gate_shadow.jsonl"))


@pytest.fixture(autouse=True)
def _isolate_candidate_registry(tmp_path, monkeypatch):
    """Never let the test suite create/overwrite the live candidate registry (weekly_review calls
    load_registry(), which seeds + persists defaults when the file is missing)."""
    import experiment.candidate_registry as _cr

    monkeypatch.setattr(_cr, "REGISTRY_PATH", str(tmp_path / "candidate_registry.json"))


@pytest.fixture(autouse=True)
def _isolate_research_signals(tmp_path, monkeypatch):
    """Never let the test suite read/write the live research-signal tier (weekly_review + the miner
    script load/save it)."""
    import experiment.research_signals as _rs

    monkeypatch.setattr(_rs, "RESEARCH_SIGNALS_PATH", str(tmp_path / "research_signals.json"))


@pytest.fixture(autouse=True)
def _isolate_dof_ledger(tmp_path, monkeypatch):
    """Never let the test suite read/write the live degrees-of-freedom ledger (weekly_review calls
    load_ledger(), which seeds + persists a fresh ledger when the file is missing)."""
    import experiment.dof_ledger as _dof

    monkeypatch.setattr(_dof, "LEDGER_PATH", str(tmp_path / "dof_ledger.json"))


@pytest.fixture(autouse=True)
def _isolate_reconciliation(tmp_path, monkeypatch):
    """Never let the test suite read/write the live replay-fidelity summary (weekly_review loads it)."""
    import experiment.reconciliation as _rec

    monkeypatch.setattr(_rec, "RECONCILIATION_PATH", str(tmp_path / "reconciliation_summary.json"))


@pytest.fixture(autouse=True)
def _isolate_scored_observations(tmp_path, monkeypatch):
    """Never let the test suite read the live scored-observation log. The weekly review's telemetry and
    candidate-authoring load it via monitoring._SCORED_PATH (a relative path that otherwise resolves to the
    running bot's real logs/experiment_scored.jsonl); redirect it to a per-test temp file so tests are
    deterministic and never run authoring/telemetry over live data."""
    import experiment.monitoring as _mon

    monkeypatch.setattr(_mon, "_SCORED_PATH", str(tmp_path / "experiment_scored.jsonl"))


@pytest.fixture(autouse=True)
def _reset_feed_status():
    """feed_status is an in-process recorder read by run_startup_health_check; reset it around every
    test so one test's degraded-feed recording can't leak into another's health-check assertions."""
    from utils import feed_status

    feed_status.reset()
    yield
    feed_status.reset()


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

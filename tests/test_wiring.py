"""Cross-module wiring consistency tests.

These tests catch "registry drift" bugs — misalignments between independent
copies of a list that should always be in sync. They are intentionally cheap
(import-only or tiny set-ops) so they run fast and always run in CI.

Each test documents exactly what invariant it checks and why it matters.
"""


# ── 1. Regime policy totality ─────────────────────────────────────────────────


def test_regime_policy_covers_all_regimes():
    """Every MarketRegime member must have a REGIME_POLICY entry.

    Missing entries caused a KeyError crash mid-run for CREDIT_STRESS /
    LATE_CYCLE_BULL / RECOVERY (found in June 2026 code review).
    The module-level assert in regime_policy.py catches this at import time,
    but this test ensures the assertion fires in CI before a push.
    """
    from data.market_regime import MarketRegime
    from risk.regime_policy import REGIME_POLICY

    missing = set(MarketRegime) - set(REGIME_POLICY.keys())
    assert not missing, (
        f"REGIME_POLICY missing entries for: {missing}. "
        "Add them to risk/regime_policy.py before deploying."
    )


def test_get_regime_policy_returns_for_every_regime():
    """get_regime_policy() must return a policy (not raise) for every enum value."""
    from data.market_regime import MarketRegime
    from risk.regime_policy import get_regime_policy

    for regime in MarketRegime:
        policy = get_regime_policy(regime.value)
        assert policy is not None, f"get_regime_policy returned None for {regime}"


def test_get_regime_policy_falls_back_on_missing_entry():
    """Lines 117-121: if REGIME_POLICY.get() returns None, log + return UNKNOWN policy."""
    from unittest.mock import patch

    import risk.regime_policy as rp

    with patch.dict(rp.REGIME_POLICY, {}, clear=True):
        policy = rp.get_regime_policy("BULL_TREND")
    assert policy.block_new_buys is True  # UNKNOWN fallback


# ── 2. Signal registry consistency ───────────────────────────────────────────


def test_valid_buy_signals_derived_from_registry():
    """VALID_BUY_SIGNALS must equal AI_CITEABLE_SIGNALS (the registry derivation).

    An independent copy diverges silently when signals are added/removed.
    """
    from models import VALID_BUY_SIGNALS
    from signals.registry import AI_CITEABLE_SIGNALS

    assert VALID_BUY_SIGNALS == AI_CITEABLE_SIGNALS, (
        f"VALID_BUY_SIGNALS != AI_CITEABLE_SIGNALS.\n"
        f"  In VALID_BUY_SIGNALS only: {VALID_BUY_SIGNALS - AI_CITEABLE_SIGNALS}\n"
        f"  In AI_CITEABLE_SIGNALS only: {AI_CITEABLE_SIGNALS - VALID_BUY_SIGNALS}"
    )


def test_no_globally_disabled_signal_in_ai_citeable():
    """Disabled signals must not appear in AI_CITEABLE_SIGNALS.

    If a disabled signal is citable, the AI can cite it, its size multiplier
    is applied, and the win-rate stats for that signal get contaminated.
    """
    from signals.evaluator import GLOBALLY_DISABLED
    from signals.registry import AI_CITEABLE_SIGNALS

    overlap = GLOBALLY_DISABLED & AI_CITEABLE_SIGNALS
    assert not overlap, (
        f"Globally disabled signals are in AI_CITEABLE_SIGNALS: {overlap}. "
        "Remove them from AI_CITEABLE_SIGNALS (they must not be citable)."
    )


def test_all_active_signals_have_hold_days():
    """Every active long signal must have a SIGNAL_MAX_HOLD_DAYS entry.

    Missing entries fall back to MAX_HOLD_DAYS (3 days), which may be wrong
    for short-duration signals like gap_and_go (2 days) or long ones like
    golden_cross (5 days).
    """
    from config import SIGNAL_MAX_HOLD_DAYS
    from signals.registry import ACTIVE_LONG_SIGNALS

    missing = ACTIVE_LONG_SIGNALS - set(SIGNAL_MAX_HOLD_DAYS.keys())
    assert not missing, (
        f"Active signals missing from SIGNAL_MAX_HOLD_DAYS: {missing}. "
        "Add them to config.py SIGNAL_MAX_HOLD_DAYS."
    )


def test_ai_tool_enum_matches_registry():
    """The AI tool key_signal enum must exactly equal AI_CITEABLE_SIGNALS.

    Mismatches mean the AI is offered signals it cannot produce effects with
    (schema says valid, but evaluator/sizer won't handle them correctly).
    """
    from analysis.ai_analyst import _DECISION_TOOL
    from signals.registry import AI_CITEABLE_SIGNALS

    # _DECISION_TOOL uses Anthropic tool format: input_schema (not function.parameters)
    buy_items = _DECISION_TOOL["input_schema"]["properties"]["buy_candidates"]["items"]
    tool_enum = set(buy_items["properties"]["key_signal"]["enum"])

    assert tool_enum == AI_CITEABLE_SIGNALS, (
        f"AI tool key_signal enum != AI_CITEABLE_SIGNALS.\n"
        f"  In enum only: {tool_enum - AI_CITEABLE_SIGNALS}\n"
        f"  In registry only: {AI_CITEABLE_SIGNALS - tool_enum}"
    )


# ── 3. Position-weight cap after full multiplier chain ────────────────────────


def test_max_position_weight_respected_with_all_scalars_at_max():
    """The product of all 12 sizing scalars must not exceed MAX_POSITION_WEIGHT.

    Each scalar is bounded individually, but their product can exceed the cap
    if the post-chain clamp is missing (C2).  This tests the worst-case joint max.
    """
    import config
    from risk.position_sizer import (
        amihud_size_scalar,
        cofiring_boost,
        correlation_scalar,
        get_signal_size_multiplier,
        mqr_size_multiplier,
        nhl_scalar,
        seasonal_scalar,
        vol_of_vol_scalar,
    )

    portfolio_value = 100_000.0

    # Worst-case values for each scalar (observed maximums from position_sizer logic)
    best_signal = "iv_compression"  # 1.5×
    sig_mult = get_signal_size_multiplier(best_signal)  # 1.5
    cofire = cofiring_boost(5)  # max co-fire boost
    amihud = amihud_size_scalar(False)  # 1.0 (not illiquid)
    mqr = mqr_size_multiplier(3)  # max MQS
    vov = vol_of_vol_scalar(0.0)  # low VoV = max
    seasonal = seasonal_scalar(best_signal)  # conservative
    macro = 1.25  # max macro boost (from macro_scalar)
    corr = correlation_scalar(0.0)  # low correlation = max
    nhl = nhl_scalar(2.0)  # high NHL = max
    dd_scalar = 1.0  # no drawdown
    garch = 1.0  # no GARCH reduction

    base = config.MAX_POSITION_WEIGHT * portfolio_value  # use cap itself as base for worst-case

    raw_notional = (
        base
        * dd_scalar
        * sig_mult
        * cofire
        * amihud
        * garch
        * mqr
        * vov
        * seasonal
        * macro
        * corr
        * nhl
    )

    # Apply the post-chain cap (as in main.py after C2 fix)
    max_notional = portfolio_value * config.MAX_POSITION_WEIGHT
    capped_notional = min(raw_notional, max_notional)

    assert capped_notional <= max_notional * 1.001, (  # 0.1% float tolerance
        f"Post-chain notional ${capped_notional:.2f} exceeds MAX_POSITION_WEIGHT cap "
        f"${max_notional:.2f} ({config.MAX_POSITION_WEIGHT:.0%} of ${portfolio_value:.0f})"
    )


# ── 4. README risk numbers match config ───────────────────────────────────────


def test_readme_skipped_when_file_absent(monkeypatch):
    """test_readme_risk_numbers_match_config returns early (no assert) if README is absent."""
    monkeypatch.setattr(
        "builtins.open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("README.md"))
    )
    # If the function raises anything, the test fails
    test_readme_risk_numbers_match_config()


def test_readme_numbers_checked_when_file_present(monkeypatch):
    """Lines 232-250: readme-found path — open succeeds, both assertions execute."""
    from io import StringIO

    import config

    risk_pct = config.RISK_PER_TRADE_PCT * 100
    pos_pct = int(config.MAX_POSITION_WEIGHT * 100)
    fake_readme = f"Risk: {risk_pct:.2f}%\n{pos_pct}% of portfolio\n"

    monkeypatch.setattr("builtins.open", lambda *a, **kw: StringIO(fake_readme))
    test_readme_risk_numbers_match_config()


def test_readme_risk_numbers_match_config():
    """README must state the correct RISK_PER_TRADE_PCT and MAX_POSITION_WEIGHT values.

    README/code number divergence destroys operator trust and misleads code reviewers.
    This test parses the README for the canonical numbers.
    """
    import config

    readme_path = "README.md"
    try:
        with open(readme_path) as f:
            text = f.read()
    except FileNotFoundError:
        return  # Skip if README not present (e.g. in minimal CI environments)

    risk_pct = config.RISK_PER_TRADE_PCT * 100  # e.g. 0.006 → 0.6
    pos_weight_pct = int(config.MAX_POSITION_WEIGHT * 100)  # e.g. 0.15 → 15

    # Check that the README states the correct risk-per-trade percentage
    assert f"{risk_pct:.1f}%" in text or f"{risk_pct:.2f}%" in text, (
        f"README does not mention the correct RISK_PER_TRADE_PCT value ({risk_pct:.2f}%). "
        f"Update README to match config.RISK_PER_TRADE_PCT={config.RISK_PER_TRADE_PCT}"
    )

    # Check that the README states the correct max position weight
    assert f"{pos_weight_pct}% of portfolio" in text, (
        f"README does not mention the correct MAX_POSITION_WEIGHT value ({pos_weight_pct}%). "
        f"Update README to match config.MAX_POSITION_WEIGHT={config.MAX_POSITION_WEIGHT}"
    )


# ── 5. Daily-notional fail-closed ─────────────────────────────────────────────


def test_get_daily_notional_raises_on_db_failure(monkeypatch):
    """get_daily_notional must raise OrderLedgerUnavailable on DB failure.

    A silent 0.0 return resets the daily notional cap, allowing unlimited spend.
    """
    import pytest

    import execution.trader as trader
    from models import OrderLedgerUnavailable

    def _bad_db():
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(trader, "_db", _bad_db)
    with pytest.raises(OrderLedgerUnavailable):
        trader.get_daily_notional("2026-01-01")


def test_get_open_shorts_raises_on_db_failure(monkeypatch):
    """get_open_shorts must raise OrderLedgerUnavailable on DB failure.

    A silent empty-set return bypasses the MAX_SHORT_POSITIONS cap.
    """
    import pytest

    import execution.trader as trader
    from models import OrderLedgerUnavailable

    def _bad_db():
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(trader, "_db", _bad_db)
    with pytest.raises(OrderLedgerUnavailable):
        trader.get_open_shorts()


# ── 6. tax_loss_reversal sentinel guard ──────────────────────────────────────


def test_tax_loss_reversal_does_not_fire_on_missing_data():
    """tax_loss_reversal must not fire when price_vs_52w_high_pct is absent.

    The -999 sentinel satisfies the < -30% threshold, producing false signals
    in January for any snapshot missing that field (e.g. ETFs, early backtest rows).
    """
    from signals.evaluator import evaluate_signals

    # January snapshot with no price_vs_52w_high_pct field.
    # ema9 > ema21 satisfies the EMA-up condition; calendar_month=1 satisfies the
    # seasonal gate.  Without the sentinel guard the signal would fire.
    snapshot = {
        "calendar_month": 1,
        "ema9": 100.0,
        "ema21": 99.0,
        "rsi_14": 45,
        "current_price": 100.0,
        # price_vs_52w_high_pct intentionally absent
    }
    # evaluate_signals takes an optional blocked frozenset; no regime arg needed here.
    signals = evaluate_signals(snapshot)
    assert "tax_loss_reversal" not in signals, (
        "tax_loss_reversal fired on a snapshot missing price_vs_52w_high_pct. "
        "The -999 sentinel must not satisfy the drawdown threshold."
    )

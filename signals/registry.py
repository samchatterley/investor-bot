"""Single source of truth for signal name sets used across the codebase.

All places that need to know "what signals can fire?", "what can the AI cite?",
or "is this signal valid?" should import from here rather than maintaining
independent copies.

Derives sets from signals.evaluator so there is exactly one place to add or
remove a signal name.
"""

from signals.evaluator import (
    GLOBALLY_DISABLED,
    SHORT_GLOBALLY_DISABLED,
    SHORT_SIGNAL_PRIORITY,
    SIGNAL_PRIORITY,
)

# All long signals that exist in the evaluator, minus globally-disabled ones.
ACTIVE_LONG_SIGNALS: frozenset[str] = frozenset(SIGNAL_PRIORITY.keys()) - GLOBALLY_DISABLED

# Signals the AI may cite as key_signal.  Equals the active set plus the
# "unknown" catch-all (used when no specific signal is the primary driver).
# Disabled signals are intentionally excluded: offering them to the AI wastes
# prompt tokens and can cause the AI to cite a signal that will never fire,
# tainting the empirical win-rate stats that feed back into position sizing.
AI_CITEABLE_SIGNALS: frozenset[str] = ACTIVE_LONG_SIGNALS | {"unknown"}

# ── Short-side mirror (ADR-006 part B / B2) ──────────────────────────────────
# Same derivation as the long side, applied to the short signal universe. Keeps
# the AI short-candidate enum, the ShortCandidate validator, and the short
# wiring tests reading from one source so they cannot drift independently.
ACTIVE_SHORT_SIGNALS: frozenset[str] = (
    frozenset(SHORT_SIGNAL_PRIORITY.keys()) - SHORT_GLOBALLY_DISABLED
)

# Short signals the AI may cite as key_signal, plus the "unknown" catch-all.
AI_CITEABLE_SHORT_SIGNALS: frozenset[str] = ACTIVE_SHORT_SIGNALS | {"unknown"}

# Catalyst (corporate-event) short signals — flag-driven, surfaced regardless of RS rank by the
# scanner's catalyst path. Single source of truth so the scanner and the seam test cannot drift
# from each other: adding a catalyst short means adding it here, which forces both wiring and the
# end-to-end seam test to pick it up. Must be a subset of ACTIVE_SHORT_SIGNALS (enforced in tests).
CATALYST_SHORT_SIGNALS: frozenset[str] = frozenset(
    {
        "accounting_concern_short",
        "insider_selling_short",
        "index_deletion_short",
        "eps_revision_down_short",
        "analyst_downgrade_signal",
        "guidance_downgrade",
        "secondary_offering_short",
    }
)

# ── Backtest vs live classification (experiment integrity) ────────────────────
# Every ACTIVE short signal is classified as either *backtestable* (the engine has historical data to
# evaluate it) or *live-only* (no historical point-in-time feed — catalyst / event / live-feed signals
# that can only accrue forward paper-trading evidence). This makes the backtest↔live divergence
# explicit in code: a live signal that is silently absent from the backtest baseline is an
# experiment-integrity risk (the pre-registered baseline would not match what trades live). The two
# sets must partition ACTIVE_SHORT_SIGNALS — enforced complete + disjoint in tests/test_wiring.py, so
# adding an active short forces a conscious classification here.
#
# NOTE: the backtest engine's own `_ACTIVE_SHORT_SIGNALS` (ablation baseline) still includes the
# live-only signals; they trade 0 in backtest (no data) so results are unaffected, but tightening it
# to exclude LIVE_ONLY_SHORT_SIGNALS is a recommended follow-up (an experiment-baseline decision).
LIVE_ONLY_SHORT_SIGNALS: frozenset[str] = frozenset(
    {
        "high_short_interest",  # live FINRA short interest; no point-in-time history
        "guidance_downgrade",  # 8-K guidance event
        "secondary_offering_short",  # 424B/S-3 filing event
        "lockup_expiry_short",  # IPO lock-up calendar
        "analyst_downgrade_signal",  # live analyst rating feed
        "post_earnings_gapdown_failed_bounce",  # computed live in scan_short_universe (B4: 0 bt trades)
        "accounting_concern_short",  # 8-K restatement/auditor event
        "insider_selling_short",  # Form 4 (live feed)
        "index_deletion_short",  # news-detected index removal
        "eps_revision_down_short",  # live analyst EPS-revision feed
    }
)

# Backtestable active shorts: the engine has earnings (earnings_gap_down) or quality-fundamentals
# (piotroski / accruals) data to evaluate these historically.
BACKTESTABLE_SHORT_SIGNALS: frozenset[str] = frozenset(
    {
        "earnings_gap_down",
        "piotroski_distress_short",
        "accruals_quality_short",
    }
)

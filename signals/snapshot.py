"""The snapshot seam: the field contract shared by the live scanner and the backtest engine.

`evaluate_signals` / `evaluate_short_signals` (signals/evaluator.py) read a plain dict — the
"snapshot" — assembled independently by two producers:

  * live:     data/market_data.py  (per-symbol fetch + cross-sectional + enrichment feeds)
  * backtest: backtest/engine.py    (_row_to_snapshot + _entry_signal's injected fields)

Finding 3 (Fable architecture review) is about that seam. Two failure modes live here, and this module
plus tests/test_snapshot_contract.py turn each into a CI failure instead of a silent divergence:

1. **Dead-wire across paths.** A field an evaluator branch depends on is produced in one path but not
   the other, so a signal fires live but never in the backtest (or vice versa) — passing unit tests
   the whole time. The guard: every evaluator-read field that is NOT declared live-only here must be
   produced by the backtest long path. `LIVE_ONLY_FIELDS` is the explicit, reviewable allowlist of
   fields the daily backtest genuinely cannot reconstruct (live enrichment feeds, the intraday engine,
   the short path). A new core field that goes dead in the backtest fails the parity test.

2. **Fail-open defaults.** A field read with `snapshot.get(field, DEFAULT)` where DEFAULT makes the
   signal FIRE (or a gate NOT suppress) when the field is absent. The declared direction is
   FAIL-CLOSED: an absent field must make the signal not fire / the gate suppress. This was finding 11
   (spread_proxy_20d defaulted 0.0 = fail-open at one site, 1.0 = fail-closed at another).

   Note a subtlety this module encodes: the same field can *correctly* carry different defaults at
   different sites when the comparison direction differs — e.g. a signal requiring an uptrend reads
   `ema9_above_ema21` default False, while one requiring a downtrend reads it `not ...` default True;
   BOTH mean "don't fire when the trend is unknown". Those intentional splits are listed in
   `INTENTIONAL_SPLIT_DEFAULTS`; any *other* field read with two different literal defaults is a
   finding-11-class bug and fails the contract test.

This is deliberately not a typed default-schema that the builders consume: because the correct default
is per-site (point 2), centralising one default per field would be wrong. The contract that matters —
and is enforceable — is the parity classification below plus the two invariants above.
"""

from __future__ import annotations

# Fields the evaluator reads that the DAILY BACKTEST LONG PATH cannot reconstruct, grouped by source.
# A signal depending solely on one of these is live-only (or belongs to another engine path) by design;
# in the daily long backtest the field is absent and the evaluator fails closed on it. Keep this list
# tight — it is the one piece of human judgement the parity guard trusts.
LIVE_ONLY_FIELDS: frozenset[str] = frozenset(
    {
        # ── sentiment / survey feeds ──────────────────────────────────────────
        "aaii_excessive_bulls",
        "aaii_extreme_fear",
        "fear_greed_score",
        "google_trends_spike",
        # ── EDGAR / filings enrichment ────────────────────────────────────────
        "accounting_concern",
        "secondary_offering",
        "guidance_negative",
        "guidance_positive",
        "index_deletion",
        # ── analyst-revision feed ─────────────────────────────────────────────
        "analyst_downgrade",
        "analyst_upgrade",
        "eps_estimate_cut",
        # ── options feed (IV / skew / flow) ───────────────────────────────────
        "call_skew_spike",
        "iv_cheap",
        "iv_rv_spread",
        "panic_put_skew",
        "put_call_oi_ratio",
        "unusual_call_oi",
        # ── earnings-event enrichment ─────────────────────────────────────────
        "earnings_gap_pct",
        "earnings_miss_candidate",
        "faded_earnings_gap_up_pct",
        "gap_failed_bounce",
        # ── insider / lockup / short-interest enrichment ──────────────────────
        "insider_sell_cluster",
        "lockup_expiry_soon",
        "si_peak",
        # ── macro flags the backtest precompute does not build ────────────────
        # (backtest DOES build macro_credit_stress / duration_flight / yield_curve*)
        "macro_10y_yield",
        "macro_claims_deteriorating",
        # ── breadth internals (backtest builds only breadth_thrust) ───────────
        "breadth_symbols_counted",
        "nhl_ratio",
        # ── pairs / cross-sectional correlation ───────────────────────────────
        "pairs_spread_z",
        "sector_correlation_20d",
        # ── premarket ─────────────────────────────────────────────────────────
        "premarket_gap_retrace",
        # ── intraday engine path (not the daily _entry_signal) ────────────────
        "intraday_change_pct",
        "intraday_rsi",
        "orb_breakout_up",
        "price_above_vwap",
        # ── short path cross-sectional (long _entry_signal doesn't inject it) ──
        "rs_rank_pct_10d_ago",
    }
)

# Fields deliberately read with different literal defaults at different sites because the fail-CLOSED
# value depends on the comparison direction (see the module docstring). Any field read with >1 distinct
# default that is NOT here is a finding-11-class fail-open regression.
INTENTIONAL_SPLIT_DEFAULTS: frozenset[str] = frozenset(
    {
        "vol_ratio",  # `>= vol_min` default 0.0 vs `<= vol_max` default 1.0 — both won't-fire on absent
        "rs_rank_pct",  # `< max` default 50.0 vs `>= min` default 0.0 — both won't-fire on absent
        "ema9_above_ema21",  # uptrend default False vs `not ...` downtrend default True — both fail-closed
    }
)

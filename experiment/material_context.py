"""Deterministic material-context detector (gates the v1.1 primary sample).

Per docs/EXPERIMENT.md section 15.1, the primary and secondary tests run only on candidates that
carry at least one pre-registered *material-context* category. This classifier is deterministic
(no LLM) and point-in-time: it reads only fields already present on a candidate snapshot at decision
time, so the primary sample is reproducible and lookahead-safe.

Categories already wired through the live signal book (earnings, insider, guidance, analyst upgrade,
short-squeeze) fire today. Event categories without a current data feed (secondary offering,
regulatory, index change, M&A, accounting concern) are matched from clearly-named boolean flags and
simply return False until those feeds are wired, so the detector is forward-compatible.
"""

from __future__ import annotations

# Frozen category list (order matches the detection chain below and EXPERIMENT.md section 15.1).
MATERIAL_CONTEXT_CATEGORIES: tuple[str, ...] = (
    "earnings_surprise_or_drift",
    "guidance_change",
    "insider_cluster_buying",
    "secondary_offering_dilution",
    "regulatory_event",
    "index_change",
    "analyst_action",
    "ma_event",
    "accounting_concern",
    "short_squeeze_setup",
)

_EARNINGS_GAP_PCT_MIN = 5.0  # abs earnings gap that counts as a material earnings event
_SQUEEZE_SIGNALS = frozenset(
    {"squeeze_setup_long", "squeeze_momentum_long", "short_interest_trend_long"}
)


def _flag(candidate: dict, key: str) -> bool:
    """Truthiness of an event flag on the candidate (absent or falsy -> False)."""
    return bool(candidate.get(key))


def detect_material_context(candidate: dict) -> list[str]:
    """Return the pre-registered material-context categories that apply, in registry order."""
    sigs = set(candidate.get("matched_signals") or candidate.get("signals") or [])
    gap = candidate.get("earnings_gap_pct")
    matched: list[str] = []

    if (
        "pead" in sigs
        or (gap is not None and abs(gap) >= _EARNINGS_GAP_PCT_MIN)
        or _flag(candidate, "earnings_event")
    ):
        matched.append("earnings_surprise_or_drift")
    if "guidance_raise_signal" in sigs or _flag(candidate, "guidance_event"):
        matched.append("guidance_change")
    if "insider_buying" in sigs or _flag(candidate, "insider_cluster"):
        matched.append("insider_cluster_buying")
    if _flag(candidate, "secondary_offering"):
        matched.append("secondary_offering_dilution")
    if _flag(candidate, "regulatory_event"):
        matched.append("regulatory_event")
    if _flag(candidate, "index_change"):
        matched.append("index_change")
    if "analyst_upgrade_signal" in sigs or _flag(candidate, "analyst_action"):
        matched.append("analyst_action")
    if _flag(candidate, "ma_event"):
        matched.append("ma_event")
    if _flag(candidate, "accounting_concern"):
        matched.append("accounting_concern")
    if (sigs & _SQUEEZE_SIGNALS) or _flag(candidate, "short_squeeze_setup"):
        matched.append("short_squeeze_setup")

    return matched


def is_material_context(candidate: dict) -> bool:
    """True if the candidate carries at least one material-context category."""
    return bool(detect_material_context(candidate))

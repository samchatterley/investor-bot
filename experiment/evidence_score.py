"""evidence_score_v1: the transparent, hand-weighted deterministic baseline (the floor).

Implements the pre-registered formula in docs/EXPERIMENT.md section 5. Per the v1.2 amendment, v1
is the interpretability *floor*, not the headline benchmark: the standable claim is measured against
the fitted `evidence_score_v2` (built later on the same point-in-time dataset). v1 exists so the
score is legible and so we have a transparent reference.

Rigour rule (v1.2): the expectancy components must be supplied point-in-time by the caller. This
function never fabricates them. If the required expectancy components are absent, the result is
flagged ``expectancy_present = False`` and the candidate is not scoreable for the experiment, rather
than silently scored as if expectancy were zero (which would be a leaking strawman).

Pure logic, no I/O. The component VALUES are produced upstream (point-in-time); this only combines
them with frozen, documented weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EVIDENCE_SCORE_VERSION = "v1"

# Positive contributors and penalties (subtracted), per EXPERIMENT.md section 5. concentration_penalty
# is intentionally excluded from v1 (portfolio/path-dependent -> later portfolio layer).
_POSITIVE: tuple[str, ...] = (
    "signal_edge_score",
    "signal_regime_score",
    "confluence_score",
    "liquidity_score",
    "trend_quality_score",
)
_PENALTY: tuple[str, ...] = (
    "volatility_penalty",
    "spread_penalty",
    "decay_penalty",
    "event_risk_penalty",
)
_ALL: tuple[str, ...] = _POSITIVE + _PENALTY

# The expectancy backbone: without these computed point-in-time, the candidate is not scoreable
# (scoring it would be a strawman). decay is expectancy-derived but optional (absent -> no penalty).
_EXPECTANCY_REQUIRED: tuple[str, ...] = ("signal_edge_score", "signal_regime_score")

# Frozen v1 weights: transparent unit weights (components are assumed pre-normalised by their
# providers); penalties subtract. v2 replaces these with fitted weights on a holdout.
_WEIGHTS: dict[str, float] = dict.fromkeys(_POSITIVE, 1.0) | dict.fromkeys(_PENALTY, -1.0)


@dataclass
class EvidenceScore:
    version: str
    score: float
    components: dict[str, float]
    missing: list[str] = field(default_factory=list)
    expectancy_present: bool = False


def evidence_score_v1(components: dict[str, float]) -> EvidenceScore:
    """Combine the pre-registered components into the transparent v1 score.

    Absent or None components contribute 0.0 but are recorded in ``missing``. ``expectancy_present``
    is True only when both required expectancy components are supplied; the experiment must filter on
    it so a candidate is never scored with fabricated expectancy.
    """
    present: dict[str, float] = {
        c: float(components[c]) for c in _ALL if components.get(c) is not None
    }
    missing = [c for c in _ALL if c not in present]
    score = sum(_WEIGHTS[c] * present.get(c, 0.0) for c in _ALL)
    expectancy_present = all(c in present for c in _EXPECTANCY_REQUIRED)
    return EvidenceScore(
        version=EVIDENCE_SCORE_VERSION,
        score=score,
        components=present,
        missing=missing,
        expectancy_present=expectancy_present,
    )

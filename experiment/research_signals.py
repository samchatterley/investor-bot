"""Research-candidate signal tier: authored-but-unproven signals that shadow-accrue forward evidence
and are NEVER traded.

A production signal lives in signals/evaluator.py and can open positions. A research signal lives here
as a declarative spec (feature / comparison / threshold / direction) that is only ever *replayed* over
the observation log to measure the forward return it WOULD have earned — it never touches the live
decision path. This is the safe landing zone for candidates the miner (experiment.candidate_miner) or
the LLM proposes: authoring is autonomous, but a research signal can only become a production signal
through the human approval gate (candidate_registry).

`score_research_signal` is forward-honest: it only scores observations dated on/after the signal was
registered, so a signal mined from history is never "validated" on the very data it was fit to.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date as _date

logger = logging.getLogger(__name__)

RESEARCH_SIGNALS_PATH = os.path.join("logs", "research_signals.json")


@dataclass
class ResearchSignal:
    """A declarative, shadow-only candidate signal: fires when ``feature op threshold`` holds.
    ``direction`` is how it would trade (long = buy the fired names; short = sell them)."""

    id: str
    feature: str
    op: str  # ">=" or "<="
    threshold: float
    direction: str  # "long" | "short"
    source: str = "mined"  # mined | llm | manual
    created: str = ""


def fires(sig: ResearchSignal, obs: dict) -> bool:
    """Whether the signal fires on one observation. Reads obs["features"][feature] (falling back to a
    flat snapshot dict). False when the feature is missing or non-numeric — a signal never fires on
    unknown data."""
    feats = obs.get("features")
    v = (feats if isinstance(feats, dict) else obs).get(sig.feature)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    return v >= sig.threshold if sig.op == ">=" else v <= sig.threshold


def _fwd(obs: dict, horizon: int) -> float | None:
    o = obs.get("outcomes") or {}
    return o.get(f"forward_r_{horizon}d")


def score_research_signal(
    sig: ResearchSignal,
    observations: list[dict],
    *,
    horizon: int = 5,
) -> tuple[int, float | None]:
    """Forward evidence for a research signal: replay it over observations dated on/after it was
    registered and return (n_fired, direction-adjusted excess forward_r vs the field), or (0, None).

    Long: excess = mean(fired forward_r) - field mean. Short: negated (a short profits when the name
    falls), so a good short signal fires on names that go on to LAG the field. Only observations with a
    closed horizon and date >= sig.created count, so a mined signal is judged on genuinely unseen data.
    """
    field = [
        o
        for o in observations
        if str(o.get("date", "")) >= sig.created and _fwd(o, horizon) is not None
    ]
    if not field:
        return 0, None
    field_mean = sum(_fwd(o, horizon) for o in field) / len(field)  # type: ignore[misc]
    fired = [o for o in field if fires(sig, o)]
    if not fired:
        return 0, None
    fired_mean = sum(_fwd(o, horizon) for o in fired) / len(fired)  # type: ignore[misc]
    excess = fired_mean - field_mean
    return len(fired), excess if sig.direction == "long" else -excess


def load_research_signals(path: str | None = None) -> list[ResearchSignal]:
    """Fail-safe read of the research-signal tier (empty list on any error / missing file)."""
    path = path or RESEARCH_SIGNALS_PATH
    try:
        with open(path) as fh:
            raw = json.load(fh)
        return [ResearchSignal(**s) for s in raw["signals"]]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning(f"research_signals: could not read {path} ({exc})")
        return []


def save_research_signals(signals: list[ResearchSignal], path: str | None = None) -> None:
    """Persist the research-signal tier. Fail-safe (logged, swallowed)."""
    path = path or RESEARCH_SIGNALS_PATH
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"updated": _date.today().isoformat(), "signals": [asdict(s) for s in signals]}
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:  # pragma: no cover - defensive; disk errors must not break the run
        logger.warning(f"research_signals: could not write {path}: {exc}")

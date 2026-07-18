"""Improvement-candidate registry: the backbone of the author -> evidence -> human-approval loop.

Every proposed change to the bot (a gate flip, a new signal, a param) is a Candidate carrying a
PRE-REGISTERED bar (a sample floor + an effect floor) recorded *before* the data judges it. Given a
candidate's current forward evidence, the registry renders one of three uniform verdicts:

  ACCUMULATING  -- not enough matured sample yet (the common state on thin data)
  READY         -- cleared both floors -> surfaced in the weekly-review approval queue for a human
  NOT-SUPPORTED -- enough sample, but the effect is below the floor (the hypothesis failed)

Authoring and evaluation are autonomous; promotion to live is ALWAYS a human decision. This generalises
the hand-wired triggers (MIN_CONFIDENCE 7->8, the guidance_downgrade short un-gate) into one
data-driven engine and is where autonomously-mined candidates will land once that stage is built.

Pure core (evaluate / rendering); the evidence for each candidate is supplied by the weekly review from
its own source, so the registry stays decoupled from where the numbers come from.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import date as _date

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.path.join("logs", "candidate_registry.json")

READY = "READY"
ACCUMULATING = "ACCUMULATING"
NOT_SUPPORTED = "NOT-SUPPORTED"


@dataclass
class Candidate:
    """A pre-registered improvement candidate. ``min_n`` / ``min_effect`` are the bar, fixed at
    registration; ``metric`` names what ``effect`` measures (with units) so the dossier is legible."""

    id: str
    hypothesis: str
    action: str  # what promotion does — must be reversible
    metric: str
    min_n: int
    min_effect: float
    source: str = "manual"  # manual | mined | llm
    created: str = ""


def evaluate(cand: Candidate, n: int | None, effect: float | None) -> tuple[str, float]:
    """Return (verdict, fraction-of-sample-floor-reached) from current forward evidence.

    A candidate is only READY when it clears BOTH the sample floor (guards against a lucky small
    sample) and the effect floor (guards against a real-but-worthless edge). Missing evidence reads
    ACCUMULATING, never READY -- the bar can only be cleared with data, never by its absence."""
    if n is None or effect is None:
        return ACCUMULATING, 0.0
    pct = min(n / cand.min_n, 1.0) if cand.min_n > 0 else 1.0
    if n < cand.min_n:
        return ACCUMULATING, pct
    return (READY, pct) if effect > cand.min_effect else (NOT_SUPPORTED, pct)


def build_candidate_lines(
    evaluated: list[tuple[Candidate, int | None, float | None]],
) -> list[str]:
    """Render the candidate pipeline for the weekly review: a PENDING-APPROVAL block (a dossier per
    READY candidate) followed by an in-progress block. ``evaluated`` is [(candidate, n, effect), ...]
    where n/effect are the candidate's current forward evidence (None until any has matured)."""
    if not evaluated:
        return ["Candidate pipeline: empty (no improvement candidates registered)."]

    ranked = [(c, n, e, *evaluate(c, n, e)) for c, n, e in evaluated]
    ready = [row for row in ranked if row[3] == READY]
    other = [row for row in ranked if row[3] != READY]

    lines: list[str] = []
    if ready:
        lines.append(
            f"PENDING APPROVAL -- {len(ready)} candidate(s) cleared their pre-registered bar "
            f"(promotion is a human decision):"
        )
        for cand, n, effect, _verdict, _pct in ready:
            lines.append(f"  [{cand.id}] {cand.hypothesis}")
            lines.append(
                f"     evidence: {cand.metric} = {effect:+.2f} at n={n} "
                f"(bar: n>={cand.min_n} and > {cand.min_effect:+.2f})"
            )
            lines.append(f"     if approved: {cand.action}")
    lines.append(f"Candidate pipeline ({len(ranked)} registered, monitoring only):")
    for cand, n, effect, verdict, pct in other:
        if n is None or effect is None:
            ev = "no matured evidence yet"
        else:
            ev = f"{cand.metric}={effect:+.2f} at n={n} ({pct * 100:.0f}% of n>={cand.min_n})"
        lines.append(f"  [{cand.id}] {verdict} -- {ev}")
    return lines


def default_candidates() -> list[Candidate]:
    """Seed the registry with the real in-flight, single-threshold decisions so it tracks them from
    day one (and demonstrates the shape autonomously-mined candidates will follow)."""
    return [
        Candidate(
            id="min_confidence_7_to_8",
            source="manual",
            created="2026-07-15",
            hypothesis="AI conf>=8 picks beat the field while conf<=7 do not -- raise MIN_CONFIDENCE 7->8.",
            action="Set config.MIN_CONFIDENCE = 8 (reversible).",
            metric="conf=8 edge vs field (R, 5d)",
            min_n=50,
            min_effect=0.15,
        ),
        Candidate(
            id="ungate_guidance_downgrade_shorts",
            source="manual",
            created="2026-07-16",
            hypothesis="guidance_downgrade catalyst shorts pay net-of-borrow in non-bear regimes.",
            action="Allow guidance_downgrade shorts outside bear regimes (reversible gate).",
            metric="net short edge % (25%/yr borrow)",
            min_n=200,
            min_effect=1.0,
        ),
    ]


def load_registry(path: str | None = None) -> list[Candidate]:
    """Load candidates from the registry file, seeding + persisting the defaults if it is missing.
    Fail-safe: on a read/parse error, fall back to the in-memory defaults (never raise)."""
    path = path or REGISTRY_PATH
    if not os.path.exists(path):
        cands = default_candidates()
        save_registry(cands, path)
        return cands
    try:
        with open(path) as fh:
            raw = json.load(fh)
        return [Candidate(**c) for c in raw["candidates"]]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning(f"candidate_registry: could not read {path} ({exc}); using defaults")
        return default_candidates()


def save_registry(candidates: list[Candidate], path: str | None = None) -> None:
    """Persist candidates to the registry file. Fail-safe (logged, swallowed)."""
    path = path or REGISTRY_PATH
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "updated": _date.today().isoformat(),
            "candidates": [asdict(c) for c in candidates],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:  # pragma: no cover - defensive; disk errors must not break the run
        logger.warning(f"candidate_registry: could not write {path}: {exc}")

"""Counterfactual replay: what would a different action have returned, honestly?

First capability on the validation substrate. Learning rate -- how fast we discover and confirm edge --
is the binding constraint: live outcomes trickle in one decision at a time. Counterfactual replay
manufactures effective sample by asking, for every decision the bot already made, what a *different*
action would have returned from the same forward price path.

This module is the cheapest honest tier: the **hold-horizon**. The observation log carries the forward
return at several closed horizons (1/3/5/10d) plus a round-trip cost, all point-in-time via the backfill --
so comparing horizons needs no simulator and is *not* exposed to replay-fidelity risk. (The sim-counterfactual
tier is blocked; see docs/CHANGELOG 1.170.)

The comparison is **paired on the same decisions**: a horizon is compared to the baseline only over
decisions where *both* horizons have a closed forward return. A shorter horizon closes for far more
decisions than a longer one, so averaging each over its own population would flatter the shorter hold with
a different, larger sample -- the paired difference removes that confound. Significance is a one-sample test
on the per-decision differences.

Falsification guardrails, per the governing principle: point-in-time (forward_r from the backfill, closed
horizons only); paired + cost-honest (same decisions, both netted of the same round-trip cost);
multiplicity (a "switch horizon" discovery is charged against the DOF ledger and forward-validated in the
registry, never an in-sample conclusion).
"""

from __future__ import annotations

import math

from experiment.candidate_registry import Candidate
from experiment.dof_ledger import LedgerState, record_batch

HORIZONS = (1, 3, 5, 10)
# the bot's default swing hold; the action counterfactuals are measured against it
BASELINE_HORIZON = 5


def _net_r(o: dict, h: int) -> float | None:
    """Net forward return at horizon h in R units: gross forward_r_{h}d minus the round-trip cost.

    None when the horizon has not closed yet (backfill leaves it None) -- never counted, never guessed."""
    outcomes = o.get("outcomes") or {}
    gross = outcomes.get(f"forward_r_{h}d")
    if gross is None:
        return None
    return float(gross) - float(outcomes.get("cost_r_estimate", 0.0) or 0.0)


def _paired_diffs(observations: list[dict], h: int, baseline: int) -> list[float]:
    """Per-decision (net_r at h) - (net_r at baseline), over the decisions where BOTH horizons closed."""
    diffs: list[float] = []
    for o in observations:
        rh = _net_r(o, h)
        rb = _net_r(o, baseline)
        if rh is not None and rb is not None:
            diffs.append(rh - rb)
    return diffs


def _ttest_1samp(xs: list[float]) -> tuple[float, float]:
    """(mean, two-sided p) for H0: mean(xs) = 0 (normal approx via erfc). Zero standard error reads as
    perfect separation (p=0) when the mean is non-zero, and no signal (p=1) when it is zero."""
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    if se == 0:
        return m, (1.0 if m == 0 else 0.0)
    return m, math.erfc(abs(m / se) / math.sqrt(2))


def horizon_counterfactuals(
    observations: list[dict],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    baseline: int = BASELINE_HORIZON,
) -> dict:
    """Paired uplift of each alternative hold horizon over ``baseline``, measured on the SAME decisions.

    ``per_horizon[h] = {n, uplift}`` where uplift is the mean per-decision (net_r_h - net_r_baseline) over
    decisions where both closed. A horizon appears only when it has such shared sample; ``uplift`` is None
    when nothing is comparable (e.g. the baseline horizon has not matured)."""
    per_horizon: dict[int, dict[str, float]] = {}
    for h in horizons:
        if h == baseline:
            continue
        diffs = _paired_diffs(observations, h, baseline)
        if diffs:
            per_horizon[h] = {"n": len(diffs), "uplift": round(sum(diffs) / len(diffs), 4)}
    best: int | None = (
        max(per_horizon, key=lambda h: per_horizon[h]["uplift"]) if per_horizon else None
    )
    uplift: float | None = per_horizon[best]["uplift"] if best is not None else None
    return {"per_horizon": per_horizon, "baseline": baseline, "best": best, "uplift": uplift}


def build_counterfactual_lines(result: dict, *, min_uplift: float = 0.1) -> list[str]:
    """Weekly-review telemetry: the paired hold-horizon uplift, and whether a different hold would pay."""
    per = result["per_horizon"]
    baseline = result["baseline"]
    if not per:
        return [
            "Hold-horizon counterfactual: no decisions with both a matured baseline and alternative "
            "horizon yet."
        ]
    table = ", ".join(f"{h}d {per[h]['uplift']:+.2f}R(n={per[h]['n']})" for h in sorted(per))
    lines = [
        f"Hold-horizon counterfactual (paired uplift vs {baseline}d, same decisions): {table}."
    ]
    best, uplift = result["best"], result["uplift"]
    if uplift is not None and uplift >= min_uplift:
        lines.append(
            f"  candidate: a {best}d hold beats the {baseline}d baseline by {uplift:+.2f}R net on the "
            f"same {per[best]['n']} decisions -- register + forward-validate before changing the hold."
        )
    return lines


def to_candidate(
    result: dict, created: str, *, min_n: int = 60, min_effect: float = 0.1
) -> Candidate | None:
    """Author a pre-registered candidate when a non-baseline horizon's paired uplift clears the floor."""
    per, baseline, best, uplift = (
        result["per_horizon"],
        result["baseline"],
        result["best"],
        result["uplift"],
    )
    if best is None or uplift is None or uplift < min_effect:
        return None
    return Candidate(
        id=f"hold_horizon_{baseline}_to_{best}",
        hypothesis=f"A {best}d hold nets {uplift:+.2f}R more than the {baseline}d baseline on the same "
        f"decisions (paired, n={per[best]['n']}).",
        action=f"Set the default hold horizon {baseline}d -> {best}d (reversible).",
        metric="paired net-R uplift vs baseline hold (R)",
        min_n=min_n,
        min_effect=min_effect,
        source="counterfactual",
        created=created,
    )


def author_online(
    observations: list[dict],
    ledger: LedgerState,
    created: str,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    baseline: int = BASELINE_HORIZON,
    min_effect: float = 0.1,
    now: str | None = None,
) -> tuple[LedgerState, list[Candidate]]:
    """Charge the best paired horizon-uplift against the DOF ledger, then author a Candidate only if the
    ledger rejects the null (paired one-sample test, lifetime-multiplicity corrected) AND the effect floor
    clears. The paired comparison is what keeps a shorter hold from being flattered by a larger sample."""
    diffs_by_h: dict[int, list[float]] = {}
    for h in horizons:
        if h == baseline:
            continue
        diffs = _paired_diffs(observations, h, baseline)
        if diffs:
            diffs_by_h[h] = diffs
    if not diffs_by_h:
        return ledger, []
    best = max(diffs_by_h, key=lambda h: sum(diffs_by_h[h]) / len(diffs_by_h[h]))
    _mean, p = _ttest_1samp(diffs_by_h[best])
    ledger, looks = record_batch(
        ledger,
        [
            (
                f"cf_hold_{baseline}_to_{best}",
                "counterfactual",
                f"{best}d vs {baseline}d paired uplift",
                p,
            )
        ],
        now=now,
    )
    if not looks[0].rejected:  # ledger did not clear the lifetime-multiplicity bar
        return ledger, []
    uplift = round(sum(diffs_by_h[best]) / len(diffs_by_h[best]), 4)
    result = {
        "per_horizon": {best: {"n": len(diffs_by_h[best]), "uplift": uplift}},
        "baseline": baseline,
        "best": best,
        "uplift": uplift,
    }
    c = to_candidate(result, created, min_effect=min_effect)  # gates the effect floor
    return ledger, ([c] if c else [])

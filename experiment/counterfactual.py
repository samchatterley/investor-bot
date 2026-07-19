"""Counterfactual replay: what would a different action have returned, honestly?

First capability on the validation substrate. Learning rate -- how fast we discover and confirm edge --
is the binding constraint: live outcomes trickle in one decision at a time. Counterfactual replay
manufactures effective sample by asking, for every decision the bot already made, what a *different*
action would have returned from the same forward price path.

This module is the cheapest honest tier: the **hold-horizon**. The observation log already carries the
forward return at several closed horizons (1/3/5/10d) plus a round-trip cost, all point-in-time via the
backfill -- so comparing horizons needs no simulator and is *not* exposed to replay-fidelity risk. (The
next tier, sim-counterfactuals that reconstruct alternative snapshots, IS gated on 1.165 fidelity.)

Its falsification tests, per the governing principle:
  * point-in-time -- forward_r comes from the backfill, which only fills closed horizons (no lookahead);
  * multiplicity -- comparing H horizons is H looks, so a "switch horizon" discovery is a Candidate with
    a pre-registered bar, forward-validated in the registry and charged against the DOF ledger;
  * cost-honest -- every horizon is compared net of the same round-trip cost, so a shorter hold is not
    flattered by ignoring the cost it still pays.
"""

from __future__ import annotations

from experiment.candidate_miner import _welch_p
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


def horizon_counterfactuals(
    observations: list[dict],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    baseline: int = BASELINE_HORIZON,
) -> dict:
    """Compare the net forward return of each alternative hold horizon across the logged decisions.

    Returns per-horizon (n, mean_net_r), the best horizon, and the uplift of the best over ``baseline``
    -- the counterfactual gain from changing nothing but the hold length. Horizons with no closed sample
    are omitted; uplift is None unless both the best and the baseline have sample."""
    per_horizon: dict[int, dict[str, float]] = {}
    for h in horizons:
        nets = [r for o in observations if (r := _net_r(o, h)) is not None]
        if nets:
            per_horizon[h] = {"n": len(nets), "mean_net_r": round(sum(nets) / len(nets), 4)}

    best: int | None = (
        max(per_horizon, key=lambda h: per_horizon[h]["mean_net_r"]) if per_horizon else None
    )
    uplift: float | None = None
    if best is not None and baseline in per_horizon:
        uplift = round(per_horizon[best]["mean_net_r"] - per_horizon[baseline]["mean_net_r"], 4)

    return {"per_horizon": per_horizon, "baseline": baseline, "best": best, "uplift": uplift}


def build_counterfactual_lines(result: dict, *, min_uplift: float = 0.1) -> list[str]:
    """Weekly-review telemetry: the horizon counterfactual, and whether a shorter/longer hold would pay."""
    per = result["per_horizon"]
    if not per:
        return ["Hold-horizon counterfactual: no closed-horizon sample yet."]
    baseline, best, uplift = result["baseline"], result["best"], result["uplift"]
    table = ", ".join(f"{h}d={per[h]['mean_net_r']:+.2f}R(n={per[h]['n']})" for h in sorted(per))
    lines = [f"Hold-horizon counterfactual (net R): {table}."]
    if uplift is not None and best != baseline and uplift >= min_uplift:
        lines.append(
            f"  candidate: {best}d hold beats the {baseline}d baseline by {uplift:+.2f}R net "
            f"-- register + forward-validate before changing the hold."
        )
    return lines


def to_candidate(
    result: dict, created: str, *, min_n: int = 60, min_effect: float = 0.1
) -> Candidate | None:
    """Author a pre-registered candidate when a non-baseline horizon wins by the effect floor (else None).

    The action is reversible (a config hold-length change); promotion stays a human decision after the
    registry forward-validates the uplift on data unseen when this was authored."""
    per, baseline, best, uplift = (
        result["per_horizon"],
        result["baseline"],
        result["best"],
        result["uplift"],
    )
    if best is None or best == baseline or uplift is None or uplift < min_effect:
        return None
    return Candidate(
        id=f"hold_horizon_{baseline}_to_{best}",
        hypothesis=f"A {best}d hold nets {uplift:+.2f}R more than the {baseline}d baseline "
        f"(counterfactual over {per[best]['n']} decisions).",
        action=f"Set the default hold horizon {baseline}d -> {best}d (reversible).",
        metric="net-R uplift vs baseline hold (R)",
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
    """Charge the best-vs-baseline horizon comparison against the DOF ledger, then author a Candidate only
    if the ledger rejects the null (lifetime-multiplicity-corrected) AND the uplift clears the effect floor.

    Returns the advanced ledger and the (zero or one) candidate -- the honesty upgrade over
    ``horizon_counterfactuals`` alone, which relies on forward-validation without lifetime multiplicity."""
    lists: dict[int, list[float]] = {}
    for h in horizons:
        nets = [r for o in observations if (r := _net_r(o, h)) is not None]
        if nets:
            lists[h] = nets
    means = {h: sum(v) / len(v) for h, v in lists.items()}
    if baseline not in lists or len(lists) < 2:
        return ledger, []
    best = max(means, key=lambda h: means[h])
    if best == baseline:
        return ledger, []
    _diff, p = _welch_p(lists[best], lists[baseline])
    ledger, looks = record_batch(
        ledger,
        [
            (
                f"cf_hold_{baseline}_to_{best}",
                "counterfactual",
                f"{best}d vs {baseline}d hold net R",
                p,
            )
        ],
        now=now,
    )
    if not looks[0].rejected:  # ledger did not clear the lifetime-multiplicity bar
        return ledger, []
    result = {
        "per_horizon": {h: {"n": len(lists[h]), "mean_net_r": round(means[h], 4)} for h in lists},
        "baseline": baseline,
        "best": best,
        "uplift": round(means[best] - means[baseline], 4),
    }
    c = to_candidate(result, created, min_effect=min_effect)  # gates the effect floor
    return ledger, ([c] if c else [])

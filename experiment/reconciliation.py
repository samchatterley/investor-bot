"""Live-vs-sim reconciliation: does the replay simulator reproduce what actually happened live?

Substrate brick 3 -- the immune system's immune system. The DOF ledger (1.162) and the lookahead guard
(1.163) make our *claims* honest, but every backtest and counterfactual replay still rests on one
unproven assumption: that the simulator reproduces reality. If replay reconstructs a past decision
differently from what the bot actually logged live that day, its counterfactuals are fiction -- and
self-specialization / case-memory built on replay inherit the lie.

So reality audits the validator. For each date the bot logged a live observation, the replay engine
reconstructs the same snapshot; we compare the two. **Fidelity** = the fraction of reconstructed
snapshots that match live within tolerance. Below a floor, replay-derived capabilities are flagged
not-yet-trustworthy -- this is the gate counterfactual replay must clear before we believe it.

Pure core (comparison + aggregation + rendering); the network-bound replay + observation-log IO live in
scripts/reconcile_replay.py, which persists a summary the weekly review renders fail-safe.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date as _date

logger = logging.getLogger(__name__)

RECONCILIATION_PATH = os.path.join("logs", "reconciliation_summary.json")

# Snapshot fields worth reconciling (the numeric features the evaluator actually keys on).
DEFAULT_FIELDS = (
    "rsi_14",
    "bb_pct",
    "ret_5d_pct",
    "ret_10d_pct",
    "vol_ratio",
    "current_price",
)


@dataclass
class Divergence:
    field: str  # a feature name, or "signal:<name>" for a fired-signal mismatch
    live: float | None
    sim: float | None
    delta: float | None  # live - sim for numerics; None for presence/signal mismatches


def _num(v: object) -> float | None:
    """Coerce a snapshot value to float, or None if it is absent/non-numeric (bools excluded)."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def reconcile_snapshot(
    live: dict,
    sim: dict,
    *,
    numeric_fields: tuple[str, ...] = DEFAULT_FIELDS,
    tol: float = 0.01,
    signal_key: str = "fired_signals",
) -> list[Divergence]:
    """Return the ways a replay-reconstructed snapshot (``sim``) differs from the logged-live one.

    A numeric field diverges when both sides are present and differ by more than ``tol``, or when exactly
    one side has it. Fired-signal set differences are reported as ``signal:<name>`` with presence encoded
    1.0/0.0 -- a signal that fired live but not in replay (or vice versa) is the divergence that matters
    most, because it means replay would have *traded differently*."""
    out: list[Divergence] = []
    for f in numeric_fields:
        lv = _num(live.get(f))
        sv = _num(sim.get(f))
        if lv is not None and sv is not None:
            if abs(lv - sv) > tol:
                out.append(Divergence(f, lv, sv, lv - sv))
        elif (lv is None) != (sv is None):  # present on exactly one side
            out.append(Divergence(f, lv, sv, None))
    live_sig = set(live.get(signal_key) or [])
    sim_sig = set(sim.get(signal_key) or [])
    for name in sorted(live_sig ^ sim_sig):
        out.append(
            Divergence(
                f"signal:{name}",
                1.0 if name in live_sig else 0.0,
                1.0 if name in sim_sig else 0.0,
                None,
            )
        )
    return out


def summarise(
    pairs: list[tuple[dict, dict]],
    *,
    numeric_fields: tuple[str, ...] = DEFAULT_FIELDS,
    tol: float = 0.01,
) -> dict:
    """Aggregate (live, sim) snapshot pairs into a persistable fidelity summary.

    ``fidelity`` is the fraction of pairs that reconciled with zero divergences. ``field_drift`` records,
    per field, how often it mismatched and its mean absolute delta (numeric fields only)."""
    n_pairs = len(pairs)
    n_matched = 0
    drift: dict[str, dict[str, float]] = {}
    for live, sim in pairs:
        divs = reconcile_snapshot(live, sim, numeric_fields=numeric_fields, tol=tol)
        if not divs:
            n_matched += 1
        for d in divs:
            rec = drift.setdefault(d.field, {"n_mismatch": 0.0, "abs_delta_sum": 0.0})
            rec["n_mismatch"] += 1
            if d.delta is not None:
                rec["abs_delta_sum"] += abs(d.delta)
    field_drift = {
        # a field is only in `drift` because it mismatched, so n_mismatch is always >= 1
        f: {
            "n_mismatch": int(r["n_mismatch"]),
            "mean_abs_delta": round(r["abs_delta_sum"] / r["n_mismatch"], 4),
        }
        for f, r in drift.items()
    }
    worst = max(field_drift, key=lambda f: field_drift[f]["n_mismatch"], default=None)
    return {
        "updated": _date.today().isoformat(),
        "n_pairs": n_pairs,
        "n_matched": n_matched,
        "fidelity": round(n_matched / n_pairs, 4) if n_pairs else 0.0,
        "field_drift": field_drift,
        "worst_field": worst,
    }


def build_reconciliation_lines(summary: dict, *, floor: float = 0.9) -> list[str]:
    """Render replay-fidelity telemetry for the weekly review (reality auditing the validator)."""
    n = summary.get("n_pairs", 0)
    if not n:
        return [
            "Replay fidelity: no reconstructed snapshots yet (run scripts/reconcile_replay.py)."
        ]
    fidelity = summary.get("fidelity", 0.0)
    verdict = (
        "OK" if fidelity >= floor else "FLAG -- replay-derived capabilities not yet trustworthy"
    )
    lines = [
        f"Replay fidelity (does the sim reproduce live?): {fidelity:.1%} of {n} reconstructed "
        f"snapshots matched live [{verdict}; floor {floor:.0%}].",
    ]
    worst = summary.get("worst_field")
    if worst:
        fd = summary["field_drift"][worst]
        lines.append(
            f"  worst-drifting field: {worst} ({fd['n_mismatch']} mismatch(es), "
            f"mean |Δ| {fd['mean_abs_delta']})."
        )
    return lines


def load_reconciliation_summary(path: str | None = None) -> dict:
    """Load the persisted fidelity summary. Fail-safe: {} on missing/unreadable (never raises)."""
    path = path or RECONCILIATION_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            result: dict = json.load(fh)
        return result
    except (OSError, ValueError) as exc:
        logger.warning(f"reconciliation: could not read {path} ({exc}); treating as empty")
        return {}


def save_reconciliation_summary(summary: dict, path: str | None = None) -> None:
    """Persist the fidelity summary. Fail-safe (logged, swallowed)."""
    path = path or RECONCILIATION_PATH
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2)
    except OSError as exc:  # pragma: no cover - defensive; disk errors must not break the run
        logger.warning(f"reconciliation: could not write {path}: {exc}")

"""Self-specialization: measure where the AI beats the deterministic baseline, and concentrate there.

Second capability on the substrate. This operationalises the bot's core falsifiable thesis -- does the
LLM's selection add value over the frozen deterministic Champion, and *where* -- turning it from a report
into a steering signal.

The observation log's three-arm ablation already records, per candidate, ``arm1_deterministic_rank`` (what
the baseline would pick) and ``arm3_ai_selected`` (what the AI picked), plus the forward outcome and the
market regime. So ΔR -- the AI-picked minus deterministic-top-K net forward return -- is measurable per
slice, honestly, with no simulator and no replay-fidelity dependency.

Where ΔR is positive and clears the bar the AI has demonstrated edge -> a candidate to concentrate AI
budget/conviction there; where ΔR <= 0 the AI is noise on top of the baseline -> a candidate to defer to
the deterministic pick. Every steering change is a pre-registered, forward-validated Candidate; promotion
stays a human decision.

Falsification guardrails, per the governing principle: point-in-time (net_r comes from the backfill, which
only fills closed horizons); like-for-like (both arms netted of the same round-trip cost, so the AI is not
flattered by comparing gross picks to net baselines); pre-registered + forward-validated (a slice edge is a
Candidate, never an in-sample conclusion; slice-search multiplicity is charged against the DOF ledger in
the online candidate-authoring runner, mirroring the miner).
"""

from __future__ import annotations

from experiment.candidate_registry import Candidate
from experiment.counterfactual import _net_r  # shared point-in-time net-forward-return reader

DEFAULT_HORIZON = 5  # the decision horizon the bot actually trades
DEFAULT_TOP_K = 5  # the deterministic baseline's pick count per run (its "would-have-bought" set)
SLICE_KEY = "regime"


def _slice_returns(
    observations: list[dict],
    *,
    horizon: int,
    top_k: int,
    slice_key: str,
) -> dict[str, dict[str, list[float]]]:
    """Per-slice raw net-forward-return lists for the two arms: AI-selected vs deterministic-top-K.

    Only buy_candidates with a closed horizon AND both ablation fields present are counted -- a decision
    the AI actually weighed in on and the baseline actually ranked."""
    out: dict[str, dict[str, list[float]]] = {}
    for o in observations:
        ex = o.get("extra") or {}
        if ex.get("decision_type") != "buy_candidate":
            continue
        r = _net_r(o, horizon)
        ai = ex.get("arm3_ai_selected")
        rank = ex.get("arm1_deterministic_rank")
        if r is None or ai is None or rank is None:
            continue
        sl = (ex.get("market_context") or {}).get(slice_key, "UNKNOWN")
        bucket = out.setdefault(sl, {"ai": [], "det": []})
        if ai:
            bucket["ai"].append(r)
        if rank <= top_k:
            bucket["det"].append(r)
    return out


def ai_edge_by_slice(
    observations: list[dict],
    *,
    horizon: int = DEFAULT_HORIZON,
    top_k: int = DEFAULT_TOP_K,
    slice_key: str = SLICE_KEY,
) -> dict[str, dict[str, float]]:
    """ΔR per slice: mean net forward return of AI picks minus the deterministic top-K.

    A slice appears only when both arms have sample in it. ``delta_r`` > 0 means the AI's selection
    out-returned what the frozen baseline would have bought in that slice."""
    result: dict[str, dict[str, float]] = {}
    for sl, d in _slice_returns(
        observations, horizon=horizon, top_k=top_k, slice_key=slice_key
    ).items():
        if d["ai"] and d["det"]:
            ai_mean = sum(d["ai"]) / len(d["ai"])
            det_mean = sum(d["det"]) / len(d["det"])
            result[sl] = {
                "n_ai": len(d["ai"]),
                "n_det": len(d["det"]),
                "ai_mean": round(ai_mean, 4),
                "det_mean": round(det_mean, 4),
                "delta_r": round(ai_mean - det_mean, 4),
            }
    return result


def build_specialization_lines(
    edge: dict[str, dict[str, float]], *, min_n: int = 30, min_edge: float = 0.1
) -> list[str]:
    """Weekly-review telemetry: per-slice AI-vs-baseline ΔR, flagging where the AI has (or lacks) edge."""
    if not edge:
        return ["AI-vs-baseline ΔR by regime: no matured, ablation-tagged sample yet."]
    lines = ["AI-vs-baseline ΔR by regime (net R; AI picks minus deterministic top-K):"]
    for sl in sorted(edge, key=lambda s: edge[s]["delta_r"], reverse=True):
        e = edge[sl]
        verdict = ""
        if e["n_ai"] >= min_n and e["n_det"] >= min_n:
            if e["delta_r"] >= min_edge:
                verdict = " <- EDGE: candidate to concentrate AI here"
            elif e["delta_r"] <= 0:
                verdict = " <- NOISE: candidate to defer to the deterministic pick"
        lines.append(
            f"  {sl}: ΔR={e['delta_r']:+.2f} (AI {e['ai_mean']:+.2f} n={e['n_ai']} vs "
            f"det {e['det_mean']:+.2f} n={e['n_det']}){verdict}"
        )
    return lines


def to_candidate(
    slice_name: str,
    stats: dict[str, float],
    created: str,
    *,
    min_n: int = 60,
    min_effect: float = 0.1,
) -> Candidate | None:
    """Author a pre-registered "concentrate AI in this regime" candidate when a slice clears the bar."""
    if stats["n_ai"] < min_n or stats["n_det"] < min_n or stats["delta_r"] < min_effect:
        return None
    return Candidate(
        id=f"specialize_ai_{slice_name}",
        hypothesis=f"In {slice_name}, AI selection out-returns the deterministic baseline by "
        f"{stats['delta_r']:+.2f}R (n_ai={stats['n_ai']}).",
        action=f"Concentrate AI budget/conviction in the {slice_name} regime (reversible).",
        metric="ΔR AI vs deterministic top-K (net R)",
        min_n=min_n,
        min_effect=min_effect,
        source="specialization",
        created=created,
    )

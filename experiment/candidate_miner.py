"""Autonomous candidate identification: mine feature -> forward-return edges from the observation log.

For each numeric feature, this tests whether its extreme (top- or bottom-quintile) subset has a forward
return that differs from the rest of the field, then applies **Holm-Bonferroni across every test** so the
correction reflects the whole search, not a single lucky comparison. This is the discipline that keeps
autonomous mining from surfacing curve-fit noise -- but it is still only in-sample HYPOTHESIS
GENERATION: a survivor becomes a `ResearchSignal` and must then forward-validate through the registry
before it can ever be approved for live trading.

Pure; the observation-log IO + registry filing live in scripts/mine_candidates.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from experiment.candidate_registry import Candidate
from experiment.research_signals import ResearchSignal

# Numeric snapshot features worth scanning (the ones the observation log carries).
DEFAULT_FEATURES = (
    "rsi_14",
    "ret_5d_pct",
    "ret_10d_pct",
    "bb_pct",
    "rs_rank_pct",
    "vol_ratio",
)


@dataclass
class MinedEdge:
    feature: str
    op: str  # ">=" (top-quantile subset) or "<=" (bottom-quantile subset)
    threshold: float
    direction: str  # "long" if the fired subset out-returned the rest, else "short"
    n: int
    excess: float  # mean(fired forward_r) - mean(rest) — the effect the test measures
    p_value: float
    p_corrected: float


def _pct(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending list (q in [0, 1])."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _mean_var(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, var


def _welch_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch two-sample test of a vs b: return (mean_a - mean_b, two-sided p) via a normal approx of
    the t-statistic (erfc). p=1.0 when the standard error is zero (degenerate)."""
    ma, va = _mean_var(a)
    mb, vb = _mean_var(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    diff = ma - mb
    if se == 0:
        # Zero within-group variance: identical means => no difference (p=1); different means =>
        # perfect separation, an infinitely significant split (p=0).
        return diff, (1.0 if diff == 0 else 0.0)
    t = diff / se
    return diff, math.erfc(abs(t) / math.sqrt(2))


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, aligned to the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((m - rank) * pvals[idx], 1.0))  # monotone non-decreasing
        adjusted[idx] = running
    return adjusted


def mine_feature_edges(
    observations: list[dict],
    features: tuple[str, ...] = DEFAULT_FEATURES,
    *,
    horizon: int = 5,
    top_q: float = 0.8,
    bottom_q: float = 0.2,
    min_n: int = 30,
    alpha: float = 0.05,
    min_abs_excess: float = 0.15,
) -> list[MinedEdge]:
    """Return survivor feature edges, most significant first.

    A survivor clears three bars: Holm-corrected p < ``alpha`` (real after correcting for the search),
    |excess| >= ``min_abs_excess`` (worth acting on), and the fired subset has >= ``min_n`` members
    (not a lucky handful). ``observations`` are scored rows with a ``features`` dict and a closed
    forward_r horizon. Survivors are hypotheses to forward-validate, never conclusions.
    """
    field = [
        o
        for o in observations
        if (o.get("outcomes") or {}).get(f"forward_r_{horizon}d") is not None
    ]

    def _fwd(o: dict) -> float:
        return (o["outcomes"])[f"forward_r_{horizon}d"]

    def _val(o: dict, f: str) -> float | None:
        v = (o.get("features") or {}).get(f)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    tests: list[tuple[str, str, float, float, int, float]] = []  # feature, op, thr, excess, n, p
    for feat in features:
        pairs = [(v, _fwd(o)) for o in field if (v := _val(o, feat)) is not None]
        if len(pairs) < 2 * min_n:  # need enough for a subset AND a rest
            continue
        vals = sorted(v for v, _ in pairs)
        for op, thr in ((">=", _pct(vals, top_q)), ("<=", _pct(vals, bottom_q))):
            fired = [r for v, r in pairs if (v >= thr if op == ">=" else v <= thr)]
            rest = [r for v, r in pairs if not (v >= thr if op == ">=" else v <= thr)]
            if len(fired) < min_n or len(rest) < min_n:
                continue
            excess, p = _welch_p(fired, rest)
            tests.append((feat, op, thr, excess, len(fired), p))

    if not tests:
        return []
    corrected = _holm([t[5] for t in tests])
    edges = [
        MinedEdge(
            feature=feat,
            op=op,
            threshold=round(thr, 4),
            direction="long" if excess > 0 else "short",
            n=n,
            excess=excess,
            p_value=p,
            p_corrected=pc,
        )
        for (feat, op, thr, excess, n, p), pc in zip(tests, corrected, strict=True)
        if pc < alpha and abs(excess) >= min_abs_excess
    ]
    return sorted(edges, key=lambda e: e.p_corrected)


def to_research_signal(edge: MinedEdge, created: str) -> ResearchSignal:
    """Author a research-tier signal from a mined edge (shadow-only until human-approved)."""
    return ResearchSignal(
        id=f"mined_{edge.feature}_{'ge' if edge.op == '>=' else 'le'}",
        feature=edge.feature,
        op=edge.op,
        threshold=edge.threshold,
        direction=edge.direction,
        source="mined",
        created=created,
    )


def to_candidate(
    edge: MinedEdge, created: str, *, min_n: int = 60, min_effect: float = 0.15
) -> Candidate:
    """Register a mined edge as a pre-registered improvement candidate for the approval queue."""
    sig = to_research_signal(edge, created)
    arrow = ">=" if edge.op == ">=" else "<="
    return Candidate(
        id=sig.id,
        hypothesis=f"{edge.feature} {arrow} {edge.threshold} predicts a {edge.direction} forward edge "
        f"(mined: excess {edge.excess:+.2f}R, Holm p={edge.p_corrected:.3f}).",
        action=f"Promote research signal {sig.id} to production ({edge.direction}).",
        metric="forward_r excess vs field (R, 5d)",
        min_n=min_n,
        min_effect=min_effect,
        source="mined",
        created=created,
    )

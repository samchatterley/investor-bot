"""Causal case-memory: retrieve similar past situations, and prove that consulting them helps.

Third capability on the substrate. The bot has statistical memory (confidence + outcome) but no
*instance* memory -- it cannot recall "situations like this one went badly last time." This is the honest
first tier: non-parametric, retrieval-based. A **case** is a resolved decision's feature-situation plus its
realised net forward return. For a new situation we retrieve the k nearest past cases and read their mean
outcome -- the "what happened last time in situations like this" signal the miner's univariate splits
cannot capture (it keys on the full multivariate situation, interactions included).

The whole capability lives or dies on one falsification test, and it ships with it: **does consulting the
case base actually help, out-of-sample?** ``evaluate_case_memory`` splits the log by time, builds cases
from the earlier part, and asks whether, on the held-out later part, decisions whose neighbours were
positive actually out-returned those whose neighbours were negative. If that edge is not there, the memory
is noise and must not be consulted -- so nothing is wired into the live decision path here. This build
measures and validates; live consultation is a separate, human-approved step *after* the edge clears the
bar.

Point-in-time by construction: cases strictly precede the held-out decisions they are tested on.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiment.candidate_miner import _welch_p
from experiment.candidate_registry import Candidate
from experiment.counterfactual import _net_r  # shared point-in-time net-forward-return reader
from experiment.dof_ledger import LedgerState, record_batch

CASE_FEATURES = ("rsi_14", "ret_5d_pct", "ret_10d_pct", "bb_pct", "rs_rank_pct", "vol_ratio")


@dataclass
class Case:
    situation: dict[str, float]
    outcome: float  # net forward return (R) at the evaluated horizon
    date: str


def _situation(o: dict, features: tuple[str, ...]) -> dict[str, float] | None:
    """Extract the full numeric feature vector for a decision, or None if any feature is missing.

    A case must be fully specified -- a partial situation has no well-defined distance to others."""
    feats = o.get("features") or {}
    out: dict[str, float] = {}
    for f in features:
        v = feats.get(f)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        out[f] = float(v)
    return out


def _feature_scales(cases: list[Case], features: tuple[str, ...]) -> dict[str, float]:
    """Per-feature spread (population std, floored) so distance is scale-invariant across features."""
    scales: dict[str, float] = {}
    for f in features:
        xs = [c.situation[f] for c in cases]
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        scales[f] = max(var**0.5, 1e-9)
    return scales


def _distance(
    a: dict[str, float], b: dict[str, float], scales: dict[str, float], features: tuple[str, ...]
) -> float:
    """Squared, scale-normalised Euclidean distance (monotonic in true distance -- fine for ranking)."""
    return sum(((a[f] - b[f]) / scales[f]) ** 2 for f in features)


def retrieve_neighbors(
    query: dict[str, float],
    cases: list[Case],
    *,
    k: int,
    scales: dict[str, float],
    features: tuple[str, ...],
) -> list[Case]:
    """The k cases whose situation is nearest the query (ascending distance)."""
    return sorted(cases, key=lambda c: _distance(query, c.situation, scales, features))[:k]


def neighbor_outcome(
    query: dict[str, float],
    cases: list[Case],
    *,
    k: int,
    scales: dict[str, float],
    features: tuple[str, ...],
) -> float | None:
    """Mean realised outcome of the k nearest past cases (None if there are no cases)."""
    nb = retrieve_neighbors(query, cases, k=k, scales=scales, features=features)
    return sum(c.outcome for c in nb) / len(nb) if nb else None


def _held_out_groups(
    observations: list[dict],
    *,
    k: int,
    split_frac: float,
    horizon: int,
    features: tuple[str, ...],
    max_test: int,
) -> tuple[int, list[float], list[float]]:
    """Temporal held-out split: build cases from the earlier ``split_frac`` of dated closed decisions, then
    on the (most-recent ``max_test`` of the) rest return the actual outcomes grouped by whether the k
    nearest cases were on average positive. Returns (n_cases, positive-neighbour actuals, negative ones)."""
    rows: list[tuple[str, dict[str, float], float]] = []
    for o in observations:
        s = _situation(o, features)
        r = _net_r(o, horizon)
        if s is not None and r is not None and o.get("date"):
            rows.append((o["date"], s, r))
    rows.sort(key=lambda x: x[0])

    n = len(rows)
    split = int(n * split_frac)
    if split < 1 or split >= n:  # need at least one case AND one held-out decision
        return split, [], []

    cases = [Case(s, r, d) for d, s, r in rows[:split]]
    scales = _feature_scales(cases, features)
    pos: list[float] = []
    neg: list[float] = []
    for _d, s, actual in rows[split:][-max_test:]:
        pred = neighbor_outcome(s, cases, k=k, scales=scales, features=features)
        (pos if pred > 0 else neg).append(actual)  # type: ignore[operator]  # cases non-empty -> float
    return len(cases), pos, neg


def evaluate_case_memory(
    observations: list[dict],
    *,
    k: int = 10,
    split_frac: float = 0.7,
    horizon: int = 5,
    features: tuple[str, ...] = CASE_FEATURES,
    max_test: int = 500,
    min_edge: float = 0.1,
) -> dict:
    """Held-out falsification: does consulting the case base improve the decision, out-of-sample?

    ``edge`` = mean(actual | neighbours positive) - mean(actual | neighbours negative). Positive edge means
    the memory's retrieval carries out-of-sample signal; ``helps`` gates on the effect floor."""
    n_cases, pos, neg = _held_out_groups(
        observations,
        k=k,
        split_frac=split_frac,
        horizon=horizon,
        features=features,
        max_test=max_test,
    )
    edge = None
    helps = False
    if pos and neg:
        edge = round(sum(pos) / len(pos) - sum(neg) / len(neg), 4)
        helps = edge >= min_edge
    return {
        "n_cases": n_cases,
        "n_test": len(pos) + len(neg),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "edge": edge,
        "helps": helps,
        "k": k,
    }


def build_case_memory_lines(result: dict) -> list[str]:
    """Weekly-review telemetry: whether consulting the case base helps out-of-sample."""
    if result.get("edge") is None:
        return [
            "Case-memory (does recalling similar situations help?): not enough matured sample yet."
        ]
    edge, helps = result["edge"], result["helps"]
    verdict = (
        "HELPS: candidate to consult the case base at decision time"
        if helps
        else "no out-of-sample edge yet -- do not consult"
    )
    return [
        f"Case-memory held-out edge (k={result['k']}): decisions with positive neighbours out-returned "
        f"negative-neighbour ones by {edge:+.2f}R "
        f"(n={result['n_test']}: {result['n_pos']}+/{result['n_neg']}-) [{verdict}]."
    ]


def to_candidate(
    result: dict, created: str, *, min_n: int = 100, min_effect: float = 0.1
) -> Candidate | None:
    """Author a pre-registered "consult the case base" candidate when the held-out edge clears the bar."""
    if result.get("edge") is None or result["n_test"] < min_n or result["edge"] < min_effect:
        return None
    return Candidate(
        id="consult_case_memory",
        hypothesis=f"Recalling the k={result['k']} nearest past situations carries out-of-sample signal: "
        f"positive-neighbour decisions out-returned negative-neighbour ones by {result['edge']:+.2f}R "
        f"(held-out n={result['n_test']}).",
        action="Consult the case base at decision time (surface nearest-neighbour outcomes to the analyst).",
        metric="held-out neighbour-signal edge (net R)",
        min_n=min_n,
        min_effect=min_effect,
        source="case_memory",
        created=created,
    )


def author_online(
    observations: list[dict],
    ledger: LedgerState,
    created: str,
    *,
    k: int = 10,
    split_frac: float = 0.7,
    horizon: int = 5,
    features: tuple[str, ...] = CASE_FEATURES,
    max_test: int = 500,
    min_effect: float = 0.1,
    now: str | None = None,
) -> tuple[LedgerState, list[Candidate]]:
    """Charge the held-out neighbour-signal test against the DOF ledger; author a "consult the case base"
    Candidate only if the ledger rejects the null AND the edge clears ``to_candidate``'s bar."""
    _n_cases, pos, neg = _held_out_groups(
        observations,
        k=k,
        split_frac=split_frac,
        horizon=horizon,
        features=features,
        max_test=max_test,
    )
    if not (pos and neg):
        return ledger, []
    _diff, p = _welch_p(pos, neg)
    ledger, looks = record_batch(
        ledger,
        [("consult_case_memory", "case_memory", f"held-out neighbour edge k={k}", p)],
        now=now,
    )
    if not looks[0].rejected:  # ledger did not clear the lifetime-multiplicity bar
        return ledger, []
    edge = round(sum(pos) / len(pos) - sum(neg) / len(neg), 4)
    c = to_candidate(  # gates the effect + sample floors
        {"edge": edge, "n_test": len(pos) + len(neg), "k": k}, created, min_effect=min_effect
    )
    return ledger, ([c] if c else [])

"""Phase 0 Gate A: instrument noise audit for the coarse context_adjustment.

Tests whether the LLM reproduces its own coarse context_adjustment under repeated calls on the
same candidate and context (fixed model, temperature, and prompt). If the instrument is not
stable at the coarse 3-level scale, a conviction-IC experiment is not worth running
(docs/EXPERIMENT.md section 2.6, Gate A): drop granularity (veto/no-veto) or pivot to qualitative
case studies.

Pure metrics plus an injected caller, so the logic is testable without the API. The live LLM
caller is supplied by scripts/phase0_noise_audit.py.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CandidateStability:
    symbol: str
    runs: list[str]
    mode: str
    flip_rate: float  # fraction of repeated calls that differ from the modal answer
    fully_stable: bool


@dataclass
class NoiseAuditResult:
    n_candidates: int
    n_runs: int
    pass_flip_threshold: float
    mean_flip_rate: float
    frac_fully_stable: float
    passed: bool
    per_candidate: list[CandidateStability] = field(default_factory=list)


def _candidate_stability(symbol: str, runs: list[str]) -> CandidateStability:
    counts = Counter(runs)
    mode, mode_count = counts.most_common(1)[0]
    flip_rate = (len(runs) - mode_count) / len(runs)
    return CandidateStability(
        symbol=symbol,
        runs=runs,
        mode=mode,
        flip_rate=flip_rate,
        fully_stable=flip_rate == 0.0,
    )


def summarise_stability(
    per_candidate_runs: dict[str, list[str]],
    pass_flip_threshold: float = 0.2,
) -> NoiseAuditResult:
    """Aggregate per-candidate repeated answers into a pass/fail stability verdict.

    ``passed`` is True when the mean per-candidate flip rate is at or below the threshold.
    """
    if not per_candidate_runs:
        raise ValueError("per_candidate_runs must be non-empty")
    stats = [_candidate_stability(sym, runs) for sym, runs in per_candidate_runs.items() if runs]
    if not stats:
        raise ValueError("every candidate had zero runs")
    mean_flip = sum(s.flip_rate for s in stats) / len(stats)
    frac_stable = sum(1 for s in stats if s.fully_stable) / len(stats)
    n_runs = max(len(s.runs) for s in stats)
    return NoiseAuditResult(
        n_candidates=len(stats),
        n_runs=n_runs,
        pass_flip_threshold=pass_flip_threshold,
        mean_flip_rate=mean_flip,
        frac_fully_stable=frac_stable,
        passed=mean_flip <= pass_flip_threshold,
        per_candidate=stats,
    )


def run_noise_audit(
    caller: Callable[[dict], str],
    candidates: list[dict],
    runs: int = 5,
    pass_flip_threshold: float = 0.2,
) -> NoiseAuditResult:
    """Call ``caller`` ``runs`` times per candidate and summarise coarse-answer stability.

    ``caller`` maps a candidate dict to a coarse context_adjustment string; it is injected so the
    audit logic is testable without the API.
    """
    if runs < 2:
        raise ValueError("runs must be >= 2 to measure stability")
    if not candidates:
        raise ValueError("candidates must be non-empty")
    per_candidate_runs: dict[str, list[str]] = {}
    for i, candidate in enumerate(candidates):
        symbol = candidate.get("symbol", f"candidate_{i}")
        per_candidate_runs[symbol] = [caller(candidate) for _ in range(runs)]
    return summarise_stability(per_candidate_runs, pass_flip_threshold)


def format_report(result: NoiseAuditResult) -> str:
    """Human-readable noise-audit summary (plain prose, no symbol tells)."""
    verdict = (
        "PASS: the coarse instrument is reproducible enough to proceed."
        if result.passed
        else "FAIL: coarse answers are not reproducible; drop granularity or pivot to case studies."
    )
    lines = [
        "Phase 0 Gate A: context_adjustment noise audit",
        f"  candidates ............. {result.n_candidates}",
        f"  runs per candidate ..... {result.n_runs}",
        f"  mean flip rate ......... {result.mean_flip_rate:.3f} "
        f"(threshold {result.pass_flip_threshold:.3f})",
        f"  fully-stable fraction .. {result.frac_fully_stable:.3f}",
        f"  VERDICT: {verdict}",
    ]
    return "\n".join(lines)

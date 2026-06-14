"""Phase 0 — Gate B: power analysis for the AI-contextual-value experiment.

Answers the go/no-go question from docs/EXPERIMENT.md §9 before any live apparatus is built:

    Given the candidate flow we can realistically expect, can the live contextual track
    detect a *plausible* incremental information coefficient (IC) inside the target window?

The test statistic is the incremental IC of ``context_adjustment`` over ``evidence_score_v1``.
For a correlation, SE(r) ≈ 1/√N, so the minimum detectable IC ≈ z / √N_eff. Clustering
(many candidates share a single day's market shock) deflates the raw count to an effective
sample N_eff via a design effect.

This module is pure arithmetic — no live data, no network. Replace the ASSUMPTION defaults
with values measured from a week of real candidate logging before trusting the verdict.

Run:  python scripts/phase0_power_analysis.py --help
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# ── ASSUMPTION defaults (REPLACE with measured candidate flow before trusting output) ──
DEFAULT_UNIVERSE = 507  # len(config.STOCK_UNIVERSE)
DEFAULT_ELIGIBLE_RATE = 0.05  # fraction of universe passing all filters on a given day
DEFAULT_CONTEXT_PRESENT_RATE = 0.15  # fraction of eligible the context-presence gate flags
DEFAULT_MATERIAL_RATE = 0.30  # fraction of context-present where context is materially non-neutral
DEFAULT_TRADING_DAYS = 126  # ~6 months
DEFAULT_DAILY_ARM3_CAP = 30  # cost cap on Arm-3 (contextual) calls per day
DEFAULT_ICC = 0.10  # intra-day intracluster correlation (clustering of same-day candidates)
DEFAULT_Z = 1.96  # two-sided 5% significance
DEFAULT_PLAUSIBLE_IC = 0.05  # a plausible single-feature incremental edge to power against


def min_detectable_ic(n_eff: float, z: float = DEFAULT_Z) -> float:
    """Smallest IC distinguishable from zero at significance ``z``, given effective N.

    SE(r) ≈ 1/√N, so the detectable correlation ≈ z / √N_eff.
    """
    if n_eff <= 0:
        raise ValueError("n_eff must be positive")
    return z / math.sqrt(n_eff)


def required_n_for_ic(ic: float, z: float = DEFAULT_Z) -> float:
    """Effective N needed to detect an IC of magnitude ``ic`` at significance ``z``."""
    if ic <= 0:
        raise ValueError("ic must be positive")
    return (z / ic) ** 2


def design_effect(cluster_size: float, icc: float) -> float:
    """Variance-inflation factor for clustered observations: 1 + (m - 1)·ρ."""
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    if not 0.0 <= icc <= 1.0:
        raise ValueError("icc must be in [0, 1]")
    return 1.0 + (cluster_size - 1.0) * icc


def effective_n(raw_n: float, cluster_size: float, icc: float) -> float:
    """Deflate a raw observation count to an effective sample size under clustering."""
    if raw_n < 0:
        raise ValueError("raw_n must be non-negative")
    return raw_n / design_effect(cluster_size, icc)


@dataclass(frozen=True)
class Assumptions:
    universe: int = DEFAULT_UNIVERSE
    eligible_rate: float = DEFAULT_ELIGIBLE_RATE
    context_present_rate: float = DEFAULT_CONTEXT_PRESENT_RATE
    material_rate: float = DEFAULT_MATERIAL_RATE
    trading_days: int = DEFAULT_TRADING_DAYS
    daily_arm3_cap: int = DEFAULT_DAILY_ARM3_CAP
    icc: float = DEFAULT_ICC
    z: float = DEFAULT_Z
    plausible_ic: float = DEFAULT_PLAUSIBLE_IC


@dataclass(frozen=True)
class PowerResult:
    eligible_per_day: float
    context_present_per_day: float  # after the daily Arm-3 cap
    material_per_day: float
    raw_context_present_n: float
    raw_material_n: float
    n_eff_context_present: float
    n_eff_material: float
    mdi_context_present: float  # min detectable IC (optimistic: all context-present obs)
    mdi_material: float  # min detectable IC (realistic: material obs carry the signal)
    powered: bool  # realistic MDI <= plausible_ic


def project(a: Assumptions) -> PowerResult:
    """Project effective sample sizes and minimum detectable IC from the assumptions.

    Two N's are reported: context-present (every flagged candidate is a regression row —
    optimistic) and material (neutral adjustments add rows but little predictor variance,
    so the binding signal is carried by the material subset — realistic). The verdict uses
    the realistic one.
    """
    eligible_per_day = a.universe * a.eligible_rate
    context_present_per_day = min(eligible_per_day * a.context_present_rate, a.daily_arm3_cap)
    material_per_day = context_present_per_day * a.material_rate

    raw_cp = context_present_per_day * a.trading_days
    raw_mat = material_per_day * a.trading_days

    n_eff_cp = effective_n(raw_cp, max(context_present_per_day, 1.0), a.icc)
    n_eff_mat = effective_n(raw_mat, max(material_per_day, 1.0), a.icc)

    mdi_cp = min_detectable_ic(n_eff_cp, a.z) if n_eff_cp > 0 else float("inf")
    mdi_mat = min_detectable_ic(n_eff_mat, a.z) if n_eff_mat > 0 else float("inf")

    return PowerResult(
        eligible_per_day=eligible_per_day,
        context_present_per_day=context_present_per_day,
        material_per_day=material_per_day,
        raw_context_present_n=raw_cp,
        raw_material_n=raw_mat,
        n_eff_context_present=n_eff_cp,
        n_eff_material=n_eff_mat,
        mdi_context_present=mdi_cp,
        mdi_material=mdi_mat,
        powered=mdi_mat <= a.plausible_ic,
    )


def _format_report(a: Assumptions, r: PowerResult) -> str:
    verdict = (
        "POWERED — the window can detect the target effect."
        if r.powered
        else "UNDERPOWERED — downgrade the live track to a trend + qualitative evidence layer."
    )
    return "\n".join(
        [
            "=" * 70,
            "  Phase 0 — Gate B: Power analysis (AI contextual value)",
            "=" * 70,
            "  Assumptions (REPLACE with measured candidate flow):",
            f"    universe ................. {a.universe}",
            f"    eligible_rate ........... {a.eligible_rate:.3f}  -> {r.eligible_per_day:.1f}/day",
            f"    context_present_rate .... {a.context_present_rate:.3f}"
            f"  -> {r.context_present_per_day:.1f}/day (cap {a.daily_arm3_cap})",
            f"    material_rate ........... {a.material_rate:.3f}  -> {r.material_per_day:.2f}/day",
            f"    trading_days ............ {a.trading_days}",
            f"    icc (clustering) ........ {a.icc:.3f}",
            f"    z ....................... {a.z:.2f}",
            f"    plausible_ic (target) ... {a.plausible_ic:.3f}",
            "-" * 70,
            f"  Raw N  (context-present) . {r.raw_context_present_n:.0f}",
            f"  Raw N  (material) ........ {r.raw_material_n:.0f}",
            f"  N_eff  (context-present) . {r.n_eff_context_present:.0f}",
            f"  N_eff  (material) ........ {r.n_eff_material:.0f}",
            "-" * 70,
            f"  Min detectable IC (context-present, optimistic) : {r.mdi_context_present:.3f}",
            f"  Min detectable IC (material, realistic) ........ {r.mdi_material:.3f}",
            f"  Required N_eff for IC={a.plausible_ic:.3f} ............... "
            f"{required_n_for_ic(a.plausible_ic, a.z):.0f}",
            "=" * 70,
            f"  VERDICT: {verdict}",
            "=" * 70,
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--universe", type=int, default=DEFAULT_UNIVERSE)
    p.add_argument("--eligible-rate", type=float, default=DEFAULT_ELIGIBLE_RATE)
    p.add_argument("--context-present-rate", type=float, default=DEFAULT_CONTEXT_PRESENT_RATE)
    p.add_argument("--material-rate", type=float, default=DEFAULT_MATERIAL_RATE)
    p.add_argument("--trading-days", type=int, default=DEFAULT_TRADING_DAYS)
    p.add_argument("--daily-arm3-cap", type=int, default=DEFAULT_DAILY_ARM3_CAP)
    p.add_argument("--icc", type=float, default=DEFAULT_ICC)
    p.add_argument("--z", type=float, default=DEFAULT_Z)
    p.add_argument("--plausible-ic", type=float, default=DEFAULT_PLAUSIBLE_IC)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = _parse_args(argv)
    a = Assumptions(
        universe=ns.universe,
        eligible_rate=ns.eligible_rate,
        context_present_rate=ns.context_present_rate,
        material_rate=ns.material_rate,
        trading_days=ns.trading_days,
        daily_arm3_cap=ns.daily_arm3_cap,
        icc=ns.icc,
        z=ns.z,
        plausible_ic=ns.plausible_ic,
    )
    print(_format_report(a, project(a)))


if __name__ == "__main__":  # pragma: no cover
    main()

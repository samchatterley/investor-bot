"""Global degrees-of-freedom ledger: lifetime multiple-testing control for the whole research loop.

The miner's Holm correction controls error *within one run*. But the bot runs the search again every
week -- and every slice, threshold, and horizon it will ever examine is another look. The family-wise
error that actually matters is across *everything the system has ever tested*, not per run. This ledger
is that accountant.

It implements **alpha-investing** (Foster & Stine, 2008) for online FDR control: a bounded pool of
"alpha-wealth" that each formal test spends. A test is conducted at level ``alpha_j = min(gamma*wealth,
alpha)``; a rejection (a discovery) refunds a payout, while a non-discovery pays a penalty. So the
effective bar *tightens* as unsuccessful looks pile up and *relaxes* after a genuine discovery -- the
"research budget that depletes as you test" made literal. Under this scheme the marginal FDR is
controlled at the target ``alpha`` (Foster & Stine, 2008).

This is a substrate primitive: the miner (and every future capability that makes discovery claims --
self-specialization, information expansion) submits its tests here, so multiplicity is accounted for
*once, globally, and auditably*. Pure core (the wealth process); the only IO is the persisted state,
fail-safe like the candidate registry.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

LEDGER_PATH = os.path.join("logs", "dof_ledger.json")

DEFAULT_ALPHA = 0.10  # target marginal FDR across the entire lifetime search
DEFAULT_GAMMA = 0.5  # fraction of current alpha-wealth invested per test


@dataclass
class Look:
    """One formally-tested hypothesis charged against the ledger -- the auditable unit of multiplicity."""

    id: str  # stable id of the hypothesis (e.g. "mined:rsi_14>=70:5d")
    family: str  # coarse group for reporting ("miner", "specialization", ...)
    description: str
    p_value: float
    alpha_level: float  # the level it was tested at (the invested budget)
    rejected: bool  # True => counted as a discovery
    wealth_after: float
    created: str  # ISO-8601 timestamp
    source: str = "system"


@dataclass
class LedgerState:
    alpha: float
    gamma: float
    wealth: float
    n_tests: int
    n_rejections: int
    looks: list[Look] = field(default_factory=list)


def new_state(alpha: float = DEFAULT_ALPHA, gamma: float = DEFAULT_GAMMA) -> LedgerState:
    """A fresh ledger. Initial wealth is ``alpha/2`` (a conservative Foster-Stine starting endowment)."""
    return LedgerState(alpha=alpha, gamma=gamma, wealth=alpha / 2, n_tests=0, n_rejections=0)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def invest_level(state: LedgerState) -> float:
    """The level the next test will be conducted at: ``min(gamma*wealth, alpha)``.

    The cap at ``alpha`` keeps the level a sane probability (so the wealth process can never go
    negative) and never tests looser than the global target; ``gamma*wealth`` is what makes the bar
    tighten as wealth is spent down by unsuccessful looks."""
    return min(state.gamma * state.wealth, state.alpha)


def record_test(
    state: LedgerState,
    look_id: str,
    family: str,
    description: str,
    p_value: float,
    *,
    source: str = "system",
    now: str | None = None,
) -> tuple[LedgerState, Look]:
    """Charge one formal test against the ledger; return the advanced state and the recorded Look.

    Reject (discovery) when ``p <= alpha_j``: wealth earns a payout of ``alpha/2``. Otherwise wealth
    pays ``alpha_j/(1-alpha_j)``. Both keep wealth strictly positive, so the ledger never dead-ends."""
    alpha_j = invest_level(state)
    rejected = p_value <= alpha_j
    if rejected:
        wealth = state.wealth + state.alpha / 2  # payout omega refunds the search budget
    else:
        wealth = state.wealth - alpha_j / (1 - alpha_j)
    look = Look(
        id=look_id,
        family=family,
        description=description,
        p_value=p_value,
        alpha_level=alpha_j,
        rejected=rejected,
        wealth_after=wealth,
        created=now or _now_iso(),
        source=source,
    )
    new = replace(
        state,
        wealth=wealth,
        n_tests=state.n_tests + 1,
        n_rejections=state.n_rejections + (1 if rejected else 0),
        looks=[*state.looks, look],
    )
    return new, look


def record_batch(
    state: LedgerState,
    tests: list[tuple[str, str, str, float]],
    *,
    source: str = "system",
    now: str | None = None,
) -> tuple[LedgerState, list[Look]]:
    """Submit a batch of ``(look_id, family, description, p_value)`` sequentially.

    Processed in ascending p-value order so the strongest candidate is tested first, on the freshest
    budget -- deterministic, and the natural step-down convention for a simultaneous batch."""
    looks: list[Look] = []
    for look_id, family, description, p_value in sorted(tests, key=lambda t: t[3]):
        state, look = record_test(
            state, look_id, family, description, p_value, source=source, now=now
        )
        looks.append(look)
    return state, looks


def build_ledger_lines(state: LedgerState) -> list[str]:
    """Research-budget telemetry for the weekly review (the system's own multiplicity audit)."""
    w0 = state.alpha / 2
    next_bar = invest_level(state)
    if state.n_tests == 0:
        return [
            f"Research-budget ledger (online FDR, alpha-investing @ mFDR<={state.alpha:.2f}): "
            f"no formal tests recorded yet; alpha-wealth {state.wealth:.4f} "
            f"(next-test bar p<={next_bar:.4f})."
        ]
    fams = sorted({lk.family for lk in state.looks})
    return [
        f"Research-budget ledger (online FDR, alpha-investing @ mFDR<={state.alpha:.2f}):",
        f"  looks: {state.n_tests} formal test(s) across {len(fams)} family(ies) "
        f"[{', '.join(fams)}]; {state.n_rejections} discovery(ies).",
        f"  alpha-wealth: {state.wealth:.4f} remaining (start {w0:.4f}); "
        f"next-test bar p<={next_bar:.4f}.",
    ]


def load_ledger(path: str | None = None) -> LedgerState:
    """Load ledger state, seeding + persisting a fresh ledger if the file is missing.
    Fail-safe: on any read/parse error, fall back to a fresh in-memory ledger (never raise)."""
    path = path or LEDGER_PATH
    if not os.path.exists(path):
        state = new_state()
        save_ledger(state, path)
        return state
    try:
        with open(path) as fh:
            raw = json.load(fh)
        return LedgerState(
            alpha=raw["alpha"],
            gamma=raw["gamma"],
            wealth=raw["wealth"],
            n_tests=raw["n_tests"],
            n_rejections=raw["n_rejections"],
            looks=[Look(**lk) for lk in raw.get("looks", [])],
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning(f"dof_ledger: could not read {path} ({exc}); using a fresh ledger")
        return new_state()


def save_ledger(state: LedgerState, path: str | None = None) -> None:
    """Persist ledger state. Fail-safe (logged, swallowed)."""
    path = path or LEDGER_PATH
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(state), fh, indent=2)
    except OSError as exc:  # pragma: no cover - defensive; disk errors must not break the run
        logger.warning(f"dof_ledger: could not write {path}: {exc}")

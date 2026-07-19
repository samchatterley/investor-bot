"""Ledger-charged candidate-authoring runner for the substrate capabilities.

Runs each capability's discovery search against the ONE global DOF ledger, so their multi-comparisons
(hold-horizons, regime slices, the case-memory neighbour test) are charged against the same lifetime
alpha-wealth -- multiplicity accounted for across capabilities, not per capability. A survivor (the ledger
rejects its null AND it clears the effect floor) is registered idempotently in the candidate registry for
forward-validation and human approval. Nothing goes live.

(The miner has its own ledger-charged runner, scripts/mine_candidates.py; both write the same ledger, so
the lifetime accounting stays unified.)

Run:  python scripts/author_candidates.py
"""

from __future__ import annotations  # pragma: no cover

import os  # pragma: no cover
import sys  # pragma: no cover
from datetime import date  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from experiment import case_memory, counterfactual, specialization  # noqa: E402  # pragma: no cover
from experiment.candidate_registry import (  # noqa: E402  # pragma: no cover
    load_registry,
    save_registry,
)
from experiment.dof_ledger import load_ledger, save_ledger  # noqa: E402  # pragma: no cover
from experiment.monitoring import load_scored_observations  # noqa: E402  # pragma: no cover


def main() -> None:  # pragma: no cover
    obs = load_scored_observations()
    today = date.today().isoformat()
    ledger = load_ledger()
    registry = load_registry()
    existing = {c.id for c in registry}

    candidates = []
    for module in (counterfactual, specialization, case_memory):
        ledger, cands = module.author_online(obs, ledger, today, now=today)
        candidates.extend(cands)

    added = 0
    for c in candidates:
        if c.id not in existing:
            registry.append(c)
            existing.add(c.id)
            added += 1

    save_ledger(ledger)
    save_registry(registry)
    print(
        f"Charged capabilities against the DOF ledger (alpha-wealth {ledger.wealth:.4f}, "
        f"{ledger.n_tests} lifetime looks, {ledger.n_rejections} discoveries); "
        f"registered {added} new candidate(s)."
    )
    print("These are hypotheses -- they forward-validate in the weekly review before any approval.")


if __name__ == "__main__":  # pragma: no cover
    main()

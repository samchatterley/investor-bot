"""Autonomous candidate authoring — the single entry point run weekly from the self-review.

Runs every discovery search (the miner + the three capabilities) against the ONE global DOF ledger,
registers survivors idempotently for forward-validation, and persists the ledger, the candidate registry,
and the shadow research-signal tier. Multiplicity is charged across all of them against the same lifetime
alpha-wealth. Nothing goes live: authoring and evaluation are autonomous, promotion is a human decision.

Called from ``analysis.weekly_review.run_weekly_review`` (so the Sunday scheduler runs it automatically)
and by ``scripts/author_candidates.py`` for a manual run. Pure orchestration over already-tested pieces;
the IO paths it touches are isolated in tests.
"""

from __future__ import annotations

from datetime import date

from experiment import case_memory, counterfactual, specialization
from experiment.candidate_miner import mine_edges_online, to_candidate, to_research_signal
from experiment.candidate_registry import Candidate, load_registry, save_registry
from experiment.dof_ledger import load_ledger, save_ledger
from experiment.monitoring import load_scored_observations
from experiment.research_signals import load_research_signals, save_research_signals


def run_authoring(observations: list[dict] | None = None, *, created: str | None = None) -> dict:
    """Charge all discovery searches against the ledger, register survivors, and persist. Idempotent
    (dedup by candidate id / research-signal id), so re-running never double-registers. Returns a summary."""
    obs = load_scored_observations() if observations is None else observations
    today = created or date.today().isoformat()
    ledger = load_ledger()
    registry = load_registry()
    existing = {c.id for c in registry}
    signals = load_research_signals()
    sig_ids = {s.id for s in signals}

    added = 0

    def _register(cands: list[Candidate]) -> None:
        nonlocal added
        for c in cands:
            if c.id not in existing:
                registry.append(c)
                existing.add(c.id)
                added += 1

    # Miner: open-mode buy candidates -> feature-edge signals (charged against the ledger).
    mined_obs = [
        o
        for o in obs
        if (o.get("extra") or {}).get("decision_type") == "buy_candidate"
        and (o.get("extra") or {}).get("mode") == "open"
    ]
    ledger, edges = mine_edges_online(mined_obs, ledger, now=today)
    for e in edges:
        sig = to_research_signal(e, today)
        if sig.id not in sig_ids:
            signals.append(sig)
            sig_ids.add(sig.id)
    _register([to_candidate(e, today) for e in edges])

    # Capabilities: hold-horizon, self-specialization, case-memory (each charges its own tests).
    for module in (counterfactual, specialization, case_memory):
        ledger, cands = module.author_online(obs, ledger, today, now=today)
        _register(cands)

    save_ledger(ledger)
    save_registry(registry)
    save_research_signals(signals)
    return {
        "registered": added,
        "ledger_tests": ledger.n_tests,
        "ledger_discoveries": ledger.n_rejections,
        "research_signals": len(signals),
    }

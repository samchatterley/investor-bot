"""Autonomous candidate identification runner.

Mines the observation log for feature -> forward-return edges (Holm-Bonferroni corrected for the
search), authors survivors into the research-signal tier (shadow-only), and registers them in the
candidate registry so they forward-validate and surface for human approval. Idempotent — dedup by id,
so re-running never double-registers.

These are in-sample hypotheses; nothing goes live. The registry's forward evaluation (weekly review) is
the real test, and promotion to a production signal is always a human decision.

Run:  python scripts/mine_candidates.py
"""

from __future__ import annotations  # pragma: no cover

import os  # pragma: no cover
import sys  # pragma: no cover
from datetime import date  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from experiment.candidate_miner import (  # noqa: E402  # pragma: no cover
    mine_edges_online,
    to_candidate,
    to_research_signal,
)
from experiment.candidate_registry import (  # noqa: E402  # pragma: no cover
    load_registry,
    save_registry,
)
from experiment.dof_ledger import load_ledger, save_ledger  # noqa: E402  # pragma: no cover
from experiment.monitoring import load_scored_observations  # noqa: E402  # pragma: no cover
from experiment.research_signals import (  # noqa: E402  # pragma: no cover
    load_research_signals,
    save_research_signals,
)


def main() -> None:  # pragma: no cover
    obs = [
        o
        for o in load_scored_observations()
        if (o.get("extra") or {}).get("decision_type") == "buy_candidate"
        and (o.get("extra") or {}).get("mode") == "open"
    ]
    ledger = load_ledger()
    today = date.today().isoformat()
    ledger, edges = mine_edges_online(obs, ledger, now=today)
    save_ledger(ledger)  # every look charged against the lifetime budget, discovery or not
    print(
        f"Mined {len(obs)} observations -> {len(edges)} survivor edge(s) "
        f"(online FDR; alpha-wealth now {ledger.wealth:.4f} after {ledger.n_tests} lifetime looks)."
    )
    if not edges:
        print("No edge cleared the ledger + effect floor. (Expected on thin/single-regime data.)")
        return

    registry = load_registry()
    existing = {c.id for c in registry}
    signals = load_research_signals()
    sig_ids = {s.id for s in signals}
    added = 0
    for e in edges:
        cand = to_candidate(e, today)
        print(f"  [{cand.id}] {cand.hypothesis}")
        if to_research_signal(e, today).id not in sig_ids:
            signals.append(to_research_signal(e, today))
            sig_ids.add(to_research_signal(e, today).id)
        if cand.id not in existing:
            registry.append(cand)
            existing.add(cand.id)
            added += 1
    save_research_signals(signals)
    save_registry(registry)
    print(f"Registered {added} new candidate(s); research tier now holds {len(signals)} signal(s).")
    print("These are hypotheses -- they forward-validate in the weekly review before any approval.")


if __name__ == "__main__":  # pragma: no cover
    main()

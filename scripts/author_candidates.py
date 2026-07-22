"""Manual candidate-authoring run (the weekly scheduler does this automatically via the self-review).

Thin wrapper over ``experiment.authoring.run_authoring``: charges every discovery search (miner + the
three capabilities) against the one global DOF ledger and registers survivors for forward-validation.
Nothing goes live -- promotion stays a human decision.

Run:  python scripts/author_candidates.py
"""

from __future__ import annotations  # pragma: no cover

import os  # pragma: no cover
import sys  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from experiment.authoring import run_authoring  # noqa: E402  # pragma: no cover


def main() -> None:  # pragma: no cover
    summary = run_authoring()
    print(
        f"Authoring run: registered {summary['registered']} new candidate(s); "
        f"DOF ledger now {summary['ledger_tests']} lifetime looks, "
        f"{summary['ledger_discoveries']} discovery(ies); "
        f"research tier holds {summary['research_signals']} signal(s)."
    )
    print("These are hypotheses -- they forward-validate in the weekly review before any approval.")


if __name__ == "__main__":  # pragma: no cover
    main()

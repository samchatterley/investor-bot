"""On-demand view of the improvement-candidate pipeline — "what is the bot working on?"

Prints every registered candidate, its current forward evidence, and which (if any) have cleared their
pre-registered bar and await human approval. Same data the weekly review emails; run it anytime.

Read-only glue around the registry + evidence sources; the engine lives in
experiment/candidate_registry.py.

Run:  python scripts/pipeline_status.py
"""

from __future__ import annotations  # pragma: no cover

import os  # pragma: no cover
import sys  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from analysis.weekly_review import _candidate_evidence  # noqa: E402  # pragma: no cover
from experiment.candidate_registry import build_candidate_lines  # noqa: E402  # pragma: no cover


def main() -> None:  # pragma: no cover
    print("=== Improvement-candidate pipeline ===")
    for line in build_candidate_lines(_candidate_evidence()):
        print(line)


if __name__ == "__main__":  # pragma: no cover
    main()

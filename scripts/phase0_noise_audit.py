"""Phase 0 Gate A runner: measure context_adjustment stability against the live LLM.

Reads a JSON fixtures file (a stratified set of ~25-50 candidate snapshots, each with structured
features plus a fixed ``context_block``), calls Arm 2 then Arm 3 for each candidate ``--runs``
times at a fixed temperature, and prints the noise-audit verdict.

This is operational glue around the LLM API and is therefore excluded from unit-test coverage
(see pyproject ``[tool.coverage.run] omit``); the testable logic lives in
``experiment/arms.py`` and ``experiment/noise_audit.py``.

Fixtures format (list of objects):
    [{"symbol": "AAA", "evidence_score": 1.2, "rsi_14": 31, ..., "context_block": "..."}]

Run:  python scripts/phase0_noise_audit.py fixtures.json --runs 5 --temperature 0.0
"""

from __future__ import annotations  # pragma: no cover

import argparse  # pragma: no cover
import json  # pragma: no cover
import os  # pragma: no cover
import sys  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

import anthropic  # noqa: E402  # pragma: no cover

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL  # noqa: E402  # pragma: no cover
from experiment.arms import (  # noqa: E402  # pragma: no cover
    ARM2_TOOL,
    ARM3_TOOL,
    build_arm2_prompt,
    build_arm3_prompt,
    build_structured_prose,
)
from experiment.noise_audit import format_report, run_noise_audit  # noqa: E402  # pragma: no cover


def _tool_call(client, model, prompt, tool, temperature):  # pragma: no cover
    """Single structured tool call; returns the tool input dict."""
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": prompt}],
    )
    block = next((b for b in resp.content if hasattr(b, "input")), None)
    return block.input if block is not None else {}


def _make_caller(client, model, temperature):  # pragma: no cover
    """Return caller(candidate) -> coarse context_adjustment via Arm 2 then Arm 3."""

    def caller(candidate: dict) -> str:
        prose = build_structured_prose(candidate)
        arm2 = _tool_call(client, model, build_arm2_prompt(prose), ARM2_TOOL, temperature)
        arm3_prompt = build_arm3_prompt(prose, arm2, candidate.get("context_block", ""))
        arm3 = _tool_call(client, model, arm3_prompt, ARM3_TOOL, temperature)
        return str(arm3.get("context_adjustment", "neutral"))

    return caller


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", help="path to JSON fixtures file")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flip-threshold", type=float, default=0.2)
    ns = parser.parse_args(argv)

    with open(ns.fixtures) as f:
        candidates = json.load(f)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    caller = _make_caller(client, CLAUDE_MODEL, ns.temperature)
    result = run_noise_audit(
        caller, candidates, runs=ns.runs, pass_flip_threshold=ns.flip_threshold
    )
    print(format_report(result))


if __name__ == "__main__":  # pragma: no cover
    main()

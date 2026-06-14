"""Weekly experiment-monitoring report (descriptive telemetry, not a hypothesis test).

Produces the monitoring block shown in the weekly review email and appended to
``docs/EXPERIMENT_LOG.md``. Per ``docs/EXPERIMENT.md`` section 2.6, weekly outputs are
MONITORING ONLY: they describe sample accumulation, Phase-0 gate status, and operational
health. They are never a hypothesis test, and they never touch the paper's Results section
(formal tests fire only at the pre-registered N_eff milestones).

The block self-populates as the measurement pipeline comes online: when an ``ExperimentState``
with accumulated samples and per-arm metrics is supplied, those fields are reported; until then
it reports the Phase-0 status honestly (pipeline not yet live, N_eff = 0).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

# User-facing strings deliberately avoid em dashes and unicode symbols (plain prose only).
MONITORING_BANNER = "Monitoring only, not a hypothesis test. Formal tests fire at the pre-registered N_eff milestones."

_DEFAULT_LOG_PATH = os.path.join("docs", "EXPERIMENT_LOG.md")


@dataclass
class ExperimentState:
    """Snapshot of experiment progress for the weekly monitoring block.

    All fields default to the pre-data / Phase-0 state so the block renders honestly before the
    measurement pipeline exists. As the pipeline comes online, callers populate the real values.
    """

    phase: str = "Phase 0 (pre-data)"
    noise_audit_status: str = "not run"
    power_gate_verdict: str = (
        "projected underpowered for the live track; scoped as a trend and qualitative layer"
    )
    n_eff_accumulated: float = 0.0
    next_milestone: int = 200
    arm_metrics: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def build_monitoring_lines(state: ExperimentState | None = None) -> list[str]:
    """Return the monitoring bullet lines for the weekly email and the experiment log."""
    s = state or ExperimentState()
    lines = [
        MONITORING_BANNER,
        f"Phase: {s.phase}.",
        f"Noise audit (Gate A): {s.noise_audit_status}.",
        f"Power analysis (Gate B): {s.power_gate_verdict}.",
        f"Effective sample accumulated: N_eff = {s.n_eff_accumulated:.0f} "
        f"(next formal test at N_eff >= {s.next_milestone}).",
    ]
    for arm, metric in s.arm_metrics.items():
        lines.append(f"{arm}: {metric}")
    lines.extend(s.notes)
    return lines


def append_log_entry(
    lines: list[str],
    entry_date: str | None = None,
    log_path: str = _DEFAULT_LOG_PATH,
) -> str:
    """Append a dated monitoring entry to the experiment log; return the block written.

    Fail-safe: directory creation and write errors are logged and swallowed (an unwritable log must
    never break the weekly review). Returns the block on success, or an empty string on failure.
    """
    entry_date = entry_date or date.today().isoformat()
    block = f"\n## {entry_date}\n\n" + "\n".join(f"- {line}" for line in lines) + "\n"
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(block)
    except OSError as exc:
        logger.warning(f"experiment monitoring: could not append to {log_path}: {exc}")
        return ""
    return block

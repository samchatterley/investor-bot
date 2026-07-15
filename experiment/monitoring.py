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

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

# User-facing strings deliberately avoid em dashes and unicode symbols (plain prose only).
MONITORING_BANNER = "Monitoring only, not a hypothesis test. Formal tests fire at the pre-registered N_eff milestones."

_DEFAULT_LOG_PATH = os.path.join("docs", "EXPERIMENT_LOG.md")
_SCORED_PATH = os.path.join("logs", "experiment_scored.jsonl")
_SHORT_GATE_SUMMARY_PATH = os.path.join("logs", "short_gate_summary.json")

# Pre-registered MIN_CONFIDENCE 7->8 gate trigger (see build_edge_anatomy_lines). Both the low and high
# confidence buckets must clear this many picks before the edge pattern is trusted enough to act on.
_TRIGGER_MIN_N = 50
_TRIGGER_LO_EDGE_MAX = 0.10  # conf<=7 must show ~no edge over the field (|edge| below this)
_TRIGGER_HI_EDGE_MIN = 0.15  # conf=8 must show a real edge above this

# Pre-registered trigger to un-gate catalyst shorts in non-bear regimes (see build_short_gate_lines).
# guidance_downgrade is the strongest catalyst-short signal in the shadow log; require a real, cost-
# surviving edge (net of a conservative hard-to-borrow assumption) over enough matured observations
# before relaxing the bear-regime gate. A flagged decision, never automatic.
_SHORT_GATE_TRIGGER_SIGNAL = "guidance_downgrade"
_SHORT_GATE_MIN_N = 200
_SHORT_GATE_MIN_NET = 1.0  # net short return % (after realistic borrow + slippage) to consider it


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


_ARM_LABELS = {
    "arm1": "Arm 1 (Champion, deterministic)",
    "arm2": "Arm 2 (structured-only LLM)",
    "arm3": "Arm 3 (contextual LLM)",
}


def build_three_arm_summary(arm_stats: dict[str, str] | None = None) -> list[str]:
    """Lines summarising Arm 1 / Arm 2 / Arm 3 performance for the email.

    ``arm_stats`` maps "arm1"/"arm2"/"arm3" to a one-line metric string, with an optional "headline"
    key for the standable comparison. Empty or None renders a pre-data scaffold line, so the section
    is honest before the measurement pipeline is collecting.
    """
    if not arm_stats:
        return [
            "No matched decisions yet. Three-arm performance (Arm 1 Champion vs Arm 2 "
            "structured-only vs Arm 3 contextual) populates once the measurement pipeline is live."
        ]
    lines = [
        f"{_ARM_LABELS[k]}: {arm_stats[k]}" for k in ("arm1", "arm2", "arm3") if arm_stats.get(k)
    ]
    if arm_stats.get("headline"):
        lines.append(arm_stats["headline"])
    return lines or ["No arm metrics available yet."]


def load_scored_observations(path: str = _SCORED_PATH) -> list[dict]:
    """Fail-safe read of the scored-observations JSONL. Returns [] on any error or missing file, so a
    bad data file can never break the weekly review."""
    rows: list[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, ValueError) as exc:
        logger.warning(f"experiment monitoring: could not read {path}: {exc}")
        return []
    return rows


def _net_r(row: dict, horizon: int) -> float | None:
    """Net (cost-adjusted) forward return at ``horizon`` days, or None if that horizon has not closed."""
    o = row.get("outcomes") or {}
    g = o.get(f"forward_r_{horizon}d")
    return None if g is None else g - (o.get("cost_r_estimate") or 0.0)


def _nets(rows: list[dict], horizon: int) -> list[float]:
    return [v for r in rows if (v := _net_r(r, horizon)) is not None]


def _selected(row: dict) -> bool:
    return bool((row.get("extra") or {}).get("arm3_ai_selected"))


def build_edge_anatomy_lines(rows: list[dict], horizon: int = 5) -> list[str]:
    """Descriptive telemetry for the AI's selection edge (MONITORING ONLY; never changes behaviour).

    From the scored open-mode buy candidates, at the primary horizon, it reports the three levers for
    improving the edge over the field so the evidence to act on any of them accumulates honestly:
      * confidence calibration -- net forward_r by arm3 confidence bucket vs the field, plus the
        PRE-REGISTERED MIN_CONFIDENCE 7->8 trigger status (only flags once each bucket clears n>=50,
        so the gate is never raised off a 12-day rally);
      * extension tilt -- AI pick-rate and edge among extended (rsi>=60) vs not;
      * per-primary-signal pick quality for families with enough picks.
    """
    candidates = [
        r
        for r in rows
        if (r.get("extra") or {}).get("decision_type") == "buy_candidate"
        and (r.get("extra") or {}).get("mode") == "open"
        and _net_r(r, horizon) is not None
    ]
    # Dedup to one row per (symbol, date): a single day can carry many run_ids (e.g. a 68-run replay
    # burst on 2026-06-21) which would otherwise multiply-count that day's candidates and skew every
    # figure below. Outcomes are deterministic per (symbol, date), so keeping the last row is safe.
    obs = list({(r.get("symbol"), r.get("date")): r for r in candidates}.values())
    if not obs:
        return ["Edge anatomy: no scored open-mode buy candidates yet."]

    field_mean = sum(_nets(obs, horizon)) / len(obs)
    picks = [r for r in obs if _selected(r)]
    lines = [
        f"Edge anatomy ({horizon}d net R, monitoring only): field n={len(obs)} "
        f"mean={field_mean:+.3f}R; AI picks n={len(picks)}."
    ]

    # -- confidence calibration + the pre-registered gate trigger --
    def _conf(r: dict) -> int | None:
        c = (r.get("extra") or {}).get("arm3_ai_confidence")
        return c if isinstance(c, int) else None

    edge: dict[str, tuple[int, float]] = {}
    for label, keep in (
        ("conf<=7", lambda c: c is not None and c <= 7),
        ("conf=8", lambda c: c == 8),
        ("conf>=9", lambda c: c is not None and c >= 9),
    ):
        vals = _nets([r for r in picks if keep(_conf(r))], horizon)
        if vals:
            m = sum(vals) / len(vals)
            edge[label] = (len(vals), m - field_mean)
            lines.append(f"  {label}: n={len(vals)} net={m:+.3f}R edge={m - field_mean:+.3f}R")
    lo, hi = edge.get("conf<=7"), edge.get("conf=8")
    if lo and hi and lo[0] >= _TRIGGER_MIN_N and hi[0] >= _TRIGGER_MIN_N:
        if abs(lo[1]) < _TRIGGER_LO_EDGE_MAX and hi[1] > _TRIGGER_HI_EDGE_MIN:
            lines.append(
                "  PRE-REGISTERED TRIGGER MET: conf<=7 shows ~no edge while conf=8 does (both "
                "n>=50) -> consider raising MIN_CONFIDENCE 7->8 (a decision to make, not automatic)."
            )
        else:
            lines.append(
                "  Pre-registered 7->8 trigger: pattern did NOT hold at n>=50 (do not raise)."
            )
    else:
        lines.append(
            f"  Pre-registered 7->8 trigger: accumulating (need n>={_TRIGGER_MIN_N}/bucket; have "
            f"conf<=7 n={lo[0] if lo else 0}, conf=8 n={hi[0] if hi else 0})."
        )

    # -- extension tilt (is the AI fishing in the mean-reverting extended pond?) --
    def _rsi(r: dict) -> float:
        v = (r.get("features") or {}).get("rsi_14")
        return float(v) if isinstance(v, (int, float)) else 0.0

    for lbl, keep_ext in (("extended(rsi>=60)", True), ("not-extended", False)):
        grp = [r for r in obs if (_rsi(r) >= 60) == keep_ext]
        gp = [r for r in grp if _selected(r)]
        if grp and gp:
            fm = sum(_nets(grp, horizon)) / len(grp)
            pm = sum(_nets(gp, horizon)) / len(gp)
            lines.append(
                f"  {lbl}: field n={len(grp)} {fm:+.3f}R | AI pick-rate "
                f"{100 * len(gp) / len(grp):.1f}% edge {pm - fm:+.3f}R"
            )

    # -- per-primary-signal pick quality (families with >=4 AI picks) --
    fam: dict[str, list[dict]] = {}
    for r in obs:
        sigs = r.get("fired_signals") or []
        fam.setdefault(sigs[0] if sigs else "(none)", []).append(r)
    for key, grp in sorted(fam.items(), key=lambda kv: -len(kv[1])):
        gp = [r for r in grp if _selected(r)]
        if len(gp) >= 4:
            fm = sum(_nets(grp, horizon)) / len(grp)
            pm = sum(_nets(gp, horizon)) / len(gp)
            lines.append(
                f"  signal {key}: field {fm:+.3f}R | AI picks n={len(gp)} {pm:+.3f}R edge {pm - fm:+.3f}R"
            )

    return lines


def load_short_gate_edges(path: str = _SHORT_GATE_SUMMARY_PATH) -> dict:
    """Fail-safe read of the short-gate summary written by scripts/eval_shadow_catalyst_shorts.py.
    Returns the per-signal edges map {signal: [n, net, hit]} (empty on any error / missing file)."""
    try:
        with open(path) as fh:
            return (json.load(fh) or {}).get("edges", {})
    except (OSError, ValueError) as exc:
        logger.warning(f"experiment monitoring: could not read {path}: {exc}")
        return {}


def build_short_gate_lines(
    edges: dict[str, tuple[int, float, float]],
    *,
    trigger_signal: str = _SHORT_GATE_TRIGGER_SIGNAL,
    min_n: int = _SHORT_GATE_MIN_N,
    min_net: float = _SHORT_GATE_MIN_NET,
) -> list[str]:
    """Short-gate efficacy telemetry (MONITORING ONLY; never changes behaviour).

    Catalyst shorts are gated to bear regimes; the shadow log measures whether they'd pay in NON-bear
    regimes too. ``edges`` maps catalyst signal -> (n, net_avg_pct, hit_pct) from
    ``analysis.shadow_catalyst_shorts.score_short_edge`` under a REALISTIC borrow+slippage haircut (the
    tradeable edge, not the gross). Reports each signal and the PRE-REGISTERED trigger for un-gating the
    top signal — which only flags once it clears both a sample floor and a net-edge floor, so the gate
    is never relaxed off a small optimistic-cost sample.
    """
    ranked = sorted(((k, v) for k, v in edges.items() if k != "__all__"), key=lambda kv: -kv[1][0])
    if not ranked:
        return ["Short-gate efficacy: no matured shadow catalyst-short observations yet."]
    lines = ["Short-gate efficacy (non-bear, net of realistic borrow+slippage, monitoring only):"]
    for sig, (n, net, hit) in ranked:
        lines.append(f"  {sig}: n={n} net={net:+.2f}% hit={hit:.0f}%")
    t = edges.get(trigger_signal)
    if t and t[0] >= min_n and t[1] > min_net:
        lines.append(
            f"  PRE-REGISTERED TRIGGER MET: {trigger_signal} net short edge {t[1]:+.2f}% at "
            f"n={t[0]} (>= {min_n}) -> consider un-gating {trigger_signal} shorts in non-bear "
            f"regimes (a decision to make, not automatic)."
        )
    elif t:
        lines.append(
            f"  Un-gate trigger ({trigger_signal}): accumulating (need n>={min_n} and "
            f"net>{min_net:.1f}%; have n={t[0]}, net={t[1]:+.2f}%)."
        )
    else:
        lines.append(f"  Un-gate trigger ({trigger_signal}): no matured observations yet.")
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

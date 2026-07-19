"""
Weekly self-review: asks Claude to analyse the past week's closed trades,
identify what worked by signal and regime, produce lessons for next week,
and propose bounded config changes.

Config changes are never applied automatically — they are validated,
recorded, and surfaced for manual operator review.

Runs Sunday evenings via run_scheduler.py.
Output saved to logs/weekly_review_YYYY-MM-DD.json.
"""

import contextlib
import json
import logging
import os
from datetime import date, timedelta

import anthropic

import config as cfg
from analysis.performance import compute_metrics, get_attribution, get_win_rates
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, LOG_DIR
from experiment.candidate_registry import build_candidate_lines, load_registry
from experiment.case_memory import build_case_memory_lines, evaluate_case_memory
from experiment.counterfactual import build_counterfactual_lines, horizon_counterfactuals
from experiment.dof_ledger import build_ledger_lines, load_ledger
from experiment.monitoring import (
    append_log_entry,
    build_edge_anatomy_lines,
    build_monitoring_lines,
    build_short_gate_lines,
    confidence_edges,
    load_scored_observations,
    load_short_gate_edges,
)
from experiment.reconciliation import build_reconciliation_lines, load_reconciliation_summary
from experiment.research_signals import load_research_signals, score_research_signal
from experiment.specialization import ai_edge_by_slice, build_specialization_lines
from utils.portfolio_tracker import load_history

logger = logging.getLogger(__name__)

_RUNTIME_CONFIG_PATH = os.path.join(LOG_DIR, "runtime_config.json")
EXPERIMENT_LOG_PATH = os.path.join("docs", "EXPERIMENT_LOG.md")


def _map_candidate_evidence(cands, conf_edges, short_edges, research_by_id, observations):
    """Attach each registered candidate's current forward evidence (n, effect) from its source. Pure;
    absent evidence maps to (None, None) so the candidate reads ACCUMULATING.

    Manually-registered candidates read their dedicated telemetry; autonomously-mined candidates
    (source="mined") are scored forward by replaying their research signal over the observation log."""
    gd = short_edges.get("guidance_downgrade")
    fixed = {
        "min_confidence_7_to_8": conf_edges.get("conf=8"),
        "ungate_guidance_downgrade_shorts": (gd[0], gd[1]) if gd else None,
    }
    out = []
    for c in cands:
        if c.id in fixed:
            ev = fixed[c.id]
        elif c.source == "mined" and c.id in research_by_id:
            n, effect = score_research_signal(research_by_id[c.id], observations)
            ev = (n, effect) if effect is not None else None
        else:
            ev = None
        out.append((c, *(ev or (None, None))))
    return out


def _candidate_evidence():
    """Load the registry + evidence sources and map them (glue around _map_candidate_evidence)."""
    obs = load_scored_observations()
    research_by_id = {s.id: s for s in load_research_signals()}
    return _map_candidate_evidence(
        load_registry(), confidence_edges(obs), load_short_gate_edges(), research_by_id, obs
    )


# Only these parameters may be auto-adjusted, within these bounds.
_SAFE_PARAMS: dict[str, dict] = {
    "MIN_CONFIDENCE": {
        "min": 6,
        "max": 9,
        "type": int,
        "desc": "minimum confidence score (1-10) to open a position",
    },
    "TRAILING_STOP_PCT": {
        "min": 2.0,
        "max": 8.0,
        "type": float,
        "desc": "% trail below highest price for the stop order",
    },
    "PARTIAL_PROFIT_PCT": {
        "min": 5.0,
        "max": 20.0,
        "type": float,
        "desc": "take half-position profit at this % unrealised gain",
    },
    "MAX_HOLD_DAYS": {
        "min": 2,
        "max": 7,
        "type": int,
        "desc": "force-exit positions held longer than this many trading days",
    },
}


def _current_param_values() -> dict[str, float | int]:
    return {
        "MIN_CONFIDENCE": cfg.MIN_CONFIDENCE,
        "TRAILING_STOP_PCT": cfg.TRAILING_STOP_PCT,
        "PARTIAL_PROFIT_PCT": cfg.PARTIAL_PROFIT_PCT,
        "MAX_HOLD_DAYS": cfg.MAX_HOLD_DAYS,
    }


def _apply_config_changes(proposed: list[dict]) -> list[dict]:
    """
    Validate proposed config changes and return a report of what would be applied.
    Changes are intentionally NOT written to disk — auto-parameter modification
    from small weekly trade samples is disabled to prevent strategy drift.
    """
    if not proposed:
        return []

    try:
        with open(_RUNTIME_CONFIG_PATH) as f:
            current_overrides: dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        current_overrides = {}

    results = []

    for change in proposed:
        param = change.get("parameter", "")
        spec = _SAFE_PARAMS.get(param)

        if not spec:
            results.append(
                {
                    **change,
                    "status": "rejected",
                    "rejection_reason": "not in the safe-to-modify list",
                }
            )
            continue

        old_val = spec["type"](current_overrides.get(param, getattr(cfg, param)))
        raw_proposed = spec["type"](change.get("proposed_value", old_val))
        new_val = max(spec["min"], min(spec["max"], raw_proposed))

        if new_val == old_val:
            results.append(
                {**change, "old_value": old_val, "new_value": new_val, "status": "unchanged"}
            )
            continue

        status = "applied" if new_val == raw_proposed else "clamped"
        results.append(
            {
                "parameter": param,
                "old_value": old_val,
                "new_value": new_val,
                "reason": change.get("reason", ""),
                "status": status,
            }
        )
        logger.info(f"Config proposal (not applied): {param} {old_val} → {new_val} ({status})")

    return results


def get_latest_review(regime: str | None = None) -> list[dict]:
    """Return structured lessons from the most recent weekly review, filtered by expiry and regime.

    Each lesson is {"lesson": str, "applies_when": str, "expiry": "YYYY-MM-DD"}.
    Plain-string lessons from older reviews are wrapped for backward compatibility.
    Expired lessons (expiry < today) are silently dropped.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    review_files = sorted(
        f for f in os.listdir(LOG_DIR) if f.startswith("weekly_review_") and f.endswith(".json")
    )
    if not review_files:
        return []
    path = os.path.join(LOG_DIR, review_files[-1])
    try:
        with open(path) as f:
            raw_lessons = json.load(f).get("lessons", [])
    except Exception:
        return []

    today = date.today().isoformat()
    result = []
    for item in raw_lessons:
        if isinstance(item, str):
            result.append({"lesson": item, "applies_when": "ANY", "expiry": None})
            continue
        expiry = item.get("expiry")
        if expiry and expiry < today:
            continue
        applies_when = item.get("applies_when", "ANY")
        if regime and applies_when not in ("ANY", regime):
            continue
        result.append(item)
    return result


def _parse_detail(detail: str) -> tuple[str | None, int | None]:
    """Extract (signal, confidence) from a pipe-separated BUY detail string.

    Format: "$500.00 | momentum | confidence=8"
    Returns (None, None) when the format is unrecognised (sell / exit details).
    """
    signal: str | None = None
    confidence: int | None = None
    for part in detail.split("|"):
        part = part.strip()
        if part.startswith("confidence="):
            with contextlib.suppress(ValueError, IndexError):
                confidence = int(part.split("=")[1])
        elif part and not part.startswith("$") and "Kelly" not in part and part != "dry run":
            signal = part
    return signal, confidence


def _fallback_review(metrics: dict, week_attribution: dict | None, reason: str) -> dict:
    """Data-backed review used when the AI narrative can't be generated.

    Returned (instead of None) so the weekly email still reports the week's REAL activity rather
    than the stub's false "no trade history available". A genuinely empty week is handled earlier
    by the no-records short-circuit; this path means trades may well have happened but the AI
    summary failed (e.g. a truncated or unparseable response).
    """
    n_trades = metrics.get("total_trades", 0)
    total_return = metrics.get("total_return_pct", 0.0)
    return {
        "week_summary": (
            f"Automated narrative unavailable ({reason}). This week: {n_trades} trade(s) "
            f"executed, net {total_return:+.2f}%. See the attribution section for detail."
        ),
        "what_worked": [],
        "what_didnt": [],
        "lessons": [],
        "proposed_changes": [],
        "week_attribution": week_attribution or {},
        "review_degraded": True,
    }


def run_weekly_review() -> dict | None:
    """
    Ask Claude to review the past 7 days and generate lessons for next week.
    Config changes proposed by the review are recorded as proposed_changes but
    never written to disk — parameter modification requires a manual decision.
    Returns the full review dict (including proposed_changes), or None if skipped/failed.
    """
    all_records = load_history()
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    week_records = [r for r in all_records if r.get("date", "") >= cutoff]

    if not week_records:
        logger.info("Weekly review: no records from the past 7 days — skipping.")
        return None

    metrics = compute_metrics(week_records)
    win_rates = get_win_rates()
    week_attribution = get_attribution(7)
    current_params = _current_param_values()

    trade_summary = []
    for r in week_records:
        for t in r.get("trades_executed", []):
            entry: dict = {
                "date": r["date"],
                "symbol": t["symbol"],
                "action": t["action"],
            }
            if t.get("action", "").upper() == "BUY":
                sig, conf = _parse_detail(t.get("detail", ""))
                if sig:
                    entry["signal"] = sig
                if conf is not None:
                    entry["confidence"] = conf
            else:
                entry["exit_reason"] = t.get("detail", "")
            trade_summary.append(entry)

    daily_breakdown = [
        {
            "date": r["date"],
            "pnl_usd": round(r.get("daily_pnl", 0), 2),
            "market": r.get("market_summary", ""),
        }
        for r in week_records
    ]

    param_block = "\n".join(
        f"  {p} = {current_params[p]}  (allowed range: {spec['min']}–{spec['max']})  — {spec['desc']}"
        for p, spec in _SAFE_PARAMS.items()
    )

    # Build compact 7-day attribution block for Claude
    attr_lines: list[str] = []
    total_attr = week_attribution.get("total_trades", 0) if week_attribution else 0
    if total_attr:
        attr_lines.append(f"7-DAY TRADE ATTRIBUTION ({total_attr} closed trades from DB):")
        for dim_name, dim_key in [
            ("By signal", "by_signal"),
            ("By regime", "by_regime"),
            ("By sector", "by_sector"),
        ]:
            dim = week_attribution.get(dim_key, {})
            if dim:
                parts = [
                    f"{k}: {v['trades']}t {v['win_rate']:.0f}%wr {v['avg_return_pct']:+.2f}%"
                    for k, v in dim.items()
                ]
                attr_lines.append(f"  {dim_name}: {' | '.join(parts)}")
        best_sig = week_attribution.get("best_signal")
        worst_sig = week_attribution.get("worst_signal")
        if best_sig:
            attr_lines.append(f"  Best performing: {best_sig}  |  Worst: {worst_sig or 'n/a'}")
    attribution_block = (
        "\n".join(attr_lines) if attr_lines else "No closed-trade DB records for this week."
    )

    prompt = f"""You are reviewing the past week of automated trading to identify what worked,
what didn't, and what changes to make for next week.

WEEK METRICS:
- Total return: {metrics.get("total_return_pct", 0):+.2f}%
- Win rate (profitable days): {metrics.get("win_rate_pct", 0):.0f}%
- Sharpe ratio: {metrics.get("sharpe", 0):.2f}
- Total trades executed: {metrics.get("total_trades", 0)}

DAILY BREAKDOWN:
{json.dumps(daily_breakdown, indent=2)}

TRADES THIS WEEK:
{json.dumps(trade_summary, indent=2)}

{attribution_block}

ALL-TIME SIGNAL WIN RATES (with regime and confidence breakdowns):
{json.dumps(win_rates, indent=2)}

CURRENT TUNABLE PARAMETERS (you may propose changes to any of these):
{param_block}

INSTRUCTIONS:
1. Analyse what drove this week's results.
2. Write "lessons" as probabilistic tendencies to guide the AI trader — they will be injected
   into every daily trading prompt next week. Lessons must express a lean or preference, never
   a hard ban. The confidence score is the sole mechanical gate; lessons must not override it.
   FORBIDDEN phrases: "never", "only", "always", "reject all", "regardless of confidence",
   "mechanically rejected", "disqualified". Use instead: "prefer", "favour", "be cautious with",
   "requires stronger evidence", "historically underperforms in".
   Base lessons only on patterns with at least 5 trades — smaller samples are noise.
3. In "config_changes", only propose a change if you have clear evidence from the data.
   Use an empty array if no changes are warranted. Each change must name one of the
   parameters listed above and give a concrete proposed_value within its allowed range.

Respond with ONLY this JSON:
{{
  "week_summary": "2-3 sentence summary of the week's performance",
  "what_worked": ["specific observation backed by the data"],
  "what_didnt": ["specific observation backed by the data"],
  "lessons": [
    {{
      "lesson": "Specific tendency backed by data (e.g. 'Mean-reversion in CHOPPY outperformed momentum this week — favour it when regime is choppy but do not exclude other signals with strong confidence.')",
      "applies_when": "BULL_TRENDING | CHOPPY | HIGH_VOL | BEAR_DAY | ANY",
      "expiry": "YYYY-MM-DD (one week from today — lessons expire unless renewed)"
    }}
  ],
  "config_changes": [
    {{
      "parameter": "MIN_CONFIDENCE",
      "proposed_value": 8,
      "reason": "confidence=7 trades returned -0.8% avg vs +3.1% at confidence=8 — raise the bar"
    }}
  ]
}}"""

    # Experiment monitoring is descriptive telemetry independent of the AI narrative — record it
    # BEFORE the AI call so a failed/truncated/timed-out review never drops the weekly telemetry entry
    # (the 2026-06-28 review failed and the EXPERIMENT_LOG.md entry was silently skipped). Fail-safe.
    monitoring_lines = build_monitoring_lines()
    # Edge-anatomy telemetry (AI-vs-field selection edge) is fail-safe around its own data read, but
    # wrap it too so a malformed scored file can never drop the core monitoring entry.
    try:
        monitoring_lines = monitoring_lines + build_edge_anatomy_lines(load_scored_observations())
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the weekly review
        logger.warning(f"edge-anatomy telemetry skipped: {exc}")
    try:
        monitoring_lines = monitoring_lines + build_short_gate_lines(load_short_gate_edges())
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the weekly review
        logger.warning(f"short-gate telemetry skipped: {exc}")
    try:
        monitoring_lines = monitoring_lines + build_candidate_lines(_candidate_evidence())
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the weekly review
        logger.warning(f"candidate pipeline telemetry skipped: {exc}")
    try:
        monitoring_lines = monitoring_lines + build_ledger_lines(load_ledger())
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the weekly review
        logger.warning(f"research-budget ledger telemetry skipped: {exc}")
    try:
        monitoring_lines = monitoring_lines + build_reconciliation_lines(
            load_reconciliation_summary()
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the weekly review
        logger.warning(f"replay-fidelity telemetry skipped: {exc}")
    try:
        monitoring_lines = monitoring_lines + build_counterfactual_lines(
            horizon_counterfactuals(load_scored_observations())
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the weekly review
        logger.warning(f"counterfactual telemetry skipped: {exc}")
    try:
        monitoring_lines = monitoring_lines + build_specialization_lines(
            ai_edge_by_slice(load_scored_observations())
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the weekly review
        logger.warning(f"specialization telemetry skipped: {exc}")
    try:
        monitoring_lines = monitoring_lines + build_case_memory_lines(
            evaluate_case_memory(load_scored_observations())
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break the weekly review
        logger.warning(f"case-memory telemetry skipped: {exc}")
    append_log_entry(monitoring_lines, log_path=EXPERIMENT_LOG_PATH)

    try:
        # Bounded timeout — the weekly review runs in the sequential scheduler, so a hung call would
        # freeze all later jobs (cf. the 1.124 incident in analysis/ai_analyst.py).
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=240.0, max_retries=1)
        response = ai_client.messages.create(
            # 8192, not 2000: the structured review (summary + worked/didn't + lessons +
            # config_changes) overran 2000 on an active week, truncating the JSON mid-string
            # ("Unterminated string") → parse failure → the email fell back to a stub that
            # falsely reported "no trades this week".
            model=CLAUDE_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()  # type: ignore[union-attr]
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        review = json.loads(raw.strip())

        # Validate proposed changes and record for logging — not applied to disk
        proposed_changes = _apply_config_changes(review.get("config_changes", []))
        review["proposed_changes"] = proposed_changes
        review["week_attribution"] = week_attribution or {}

        # Monitoring already recorded (above) so it survives an AI failure; just attach to the review.
        review["experiment_monitoring"] = monitoring_lines

        path = os.path.join(LOG_DIR, f"weekly_review_{date.today().isoformat()}.json")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(review, f, indent=2)

        logger.info(f"Weekly review saved: {path}")
        logger.info(f"Summary: {review.get('week_summary', '')}")
        for lesson in review.get("lessons", []):
            logger.info(f"  Lesson: {lesson}")
        for change in proposed_changes:
            if change["status"] in ("applied", "clamped"):
                logger.info(
                    f"  Proposed (not applied): {change['parameter']} {change['old_value']} → {change['new_value']}"
                )

        return review

    except json.JSONDecodeError as e:
        logger.error(f"Weekly review: failed to parse Claude response as JSON: {e}")
        fb = _fallback_review(metrics, week_attribution, "AI response was not valid JSON")
        fb["experiment_monitoring"] = monitoring_lines
        return fb
    except Exception as e:
        logger.error(f"Weekly review failed: {e}", exc_info=True)
        fb = _fallback_review(metrics, week_attribution, "AI call failed")
        fb["experiment_monitoring"] = monitoring_lines
        return fb

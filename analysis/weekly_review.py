"""
Weekly self-review: asks Claude to analyse the past week's closed trades,
identify what worked by signal and regime, produce lessons for next week,
and propose bounded config changes. Applied changes are written directly
to config.py so they take effect from Monday's first run.

Runs Sunday evenings via run_scheduler.py.
Output saved to logs/weekly_review_YYYY-MM-DD.json.
"""
import json
import logging
import os
from datetime import date, timedelta

import anthropic
import config as cfg
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, LOG_DIR
from analysis.performance import get_win_rates, compute_metrics
from utils.portfolio_tracker import load_history

logger = logging.getLogger(__name__)

_RUNTIME_CONFIG_PATH = os.path.join(LOG_DIR, "runtime_config.json")

# Only these parameters may be auto-adjusted, within these bounds.
_SAFE_PARAMS: dict[str, dict] = {
    "MIN_CONFIDENCE":     {"min": 6,   "max": 9,    "type": int,   "desc": "minimum confidence score (1-10) to open a position"},
    "TRAILING_STOP_PCT":  {"min": 2.0, "max": 8.0,  "type": float, "desc": "% trail below highest price for the stop order"},
    "PARTIAL_PROFIT_PCT": {"min": 5.0, "max": 20.0, "type": float, "desc": "take half-position profit at this % unrealised gain"},
    "MAX_HOLD_DAYS":      {"min": 2,   "max": 7,    "type": int,   "desc": "force-exit positions held longer than this many trading days"},
}


def _current_param_values() -> dict[str, float | int]:
    return {
        "MIN_CONFIDENCE":     cfg.MIN_CONFIDENCE,
        "TRAILING_STOP_PCT":  cfg.TRAILING_STOP_PCT,
        "PARTIAL_PROFIT_PCT": cfg.PARTIAL_PROFIT_PCT,
        "MAX_HOLD_DAYS":      cfg.MAX_HOLD_DAYS,
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
            results.append({**change, "status": "rejected",
                             "rejection_reason": "not in the safe-to-modify list"})
            continue

        old_val = spec["type"](current_overrides.get(param, getattr(cfg, param)))
        raw_proposed = spec["type"](change.get("proposed_value", old_val))
        new_val = max(spec["min"], min(spec["max"], raw_proposed))

        if new_val == old_val:
            results.append({**change, "old_value": old_val, "new_value": new_val,
                             "status": "unchanged"})
            continue

        status = "applied" if new_val == raw_proposed else "clamped"
        results.append({
            "parameter": param,
            "old_value": old_val,
            "new_value": new_val,
            "reason": change.get("reason", ""),
            "status": status,
        })
        logger.info(f"Config proposal (not applied): {param} {old_val} → {new_val} ({status})")

    return results


def get_latest_review() -> list[str]:
    """Return the lessons list from the most recent weekly review, or [] if none exists."""
    os.makedirs(LOG_DIR, exist_ok=True)
    review_files = sorted(
        f for f in os.listdir(LOG_DIR)
        if f.startswith("weekly_review_") and f.endswith(".json")
    )
    if not review_files:
        return []
    path = os.path.join(LOG_DIR, review_files[-1])
    try:
        with open(path) as f:
            return json.load(f).get("lessons", [])
    except Exception:
        return []


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
    current_params = _current_param_values()

    trade_summary = [
        {"date": r["date"], "symbol": t["symbol"],
         "action": t["action"], "detail": t.get("detail", "")}
        for r in week_records
        for t in r.get("trades_executed", [])
    ]
    daily_breakdown = [
        {"date": r["date"], "pnl_usd": round(r.get("daily_pnl", 0), 2),
         "market": r.get("market_summary", "")}
        for r in week_records
    ]

    param_block = "\n".join(
        f"  {p} = {current_params[p]}  (allowed range: {spec['min']}–{spec['max']})  — {spec['desc']}"
        for p, spec in _SAFE_PARAMS.items()
    )

    prompt = f"""You are reviewing the past week of automated trading to identify what worked,
what didn't, and what changes to make for next week.

WEEK METRICS:
- Total return: {metrics.get('total_return_pct', 0):+.2f}%
- Win rate (profitable days): {metrics.get('win_rate_pct', 0):.0f}%
- Sharpe ratio: {metrics.get('sharpe', 0):.2f}
- Total trades executed: {metrics.get('total_trades', 0)}

DAILY BREAKDOWN:
{json.dumps(daily_breakdown, indent=2)}

TRADES THIS WEEK:
{json.dumps(trade_summary, indent=2)}

ALL-TIME SIGNAL WIN RATES (with regime and confidence breakdowns):
{json.dumps(win_rates, indent=2)}

CURRENT TUNABLE PARAMETERS (you may propose changes to any of these):
{param_block}

INSTRUCTIONS:
1. Analyse what drove this week's results.
2. Write "lessons" as direct instructions to the AI trader — they will be injected into
   every daily trading prompt next week, so be specific and actionable.
3. In "config_changes", only propose a change if you have clear evidence from the data.
   Use an empty array if no changes are warranted. Each change must name one of the
   parameters listed above and give a concrete proposed_value within its allowed range.

Respond with ONLY this JSON:
{{
  "week_summary": "2-3 sentence summary of the week's performance",
  "what_worked": ["specific observation backed by the data"],
  "what_didnt": ["specific observation backed by the data"],
  "lessons": [
    "Specific instruction for the AI trader (e.g. 'Avoid mean_reversion in CHOPPY regime — 33% win rate this week. Only take momentum setups with vol_ratio > 1.5.')"
  ],
  "config_changes": [
    {{
      "parameter": "MIN_CONFIDENCE",
      "proposed_value": 8,
      "reason": "confidence=7 trades returned -0.8% avg vs +3.1% at confidence=8 — raise the bar"
    }}
  ]
}}"""

    try:
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = ai_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        review = json.loads(raw.strip())

        # Validate proposed changes and record for logging — not applied to disk
        proposed_changes = _apply_config_changes(review.get("config_changes", []))
        review["proposed_changes"] = proposed_changes

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
                logger.info(f"  Proposed (not applied): {change['parameter']} {change['old_value']} → {change['new_value']}")

        return review

    except json.JSONDecodeError as e:
        logger.error(f"Weekly review: failed to parse Claude response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Weekly review failed: {e}", exc_info=True)
        return None

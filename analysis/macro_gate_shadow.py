"""Macro-gate efficacy shadow log (read-only measurement; never affects trading).

When the macro-event gate blocks new buys (``main._execute_buy_phase``), the AI's would-be buy
candidates are recorded here so we can measure — PER EVENT — whether skipping actually helped: the
forward market-excess return those blocked names went on to make.

Reading the score: aggregate excess < 0 ⇒ the gate SAVED us (the names we didn't buy underperformed
the market); > 0 ⇒ the gate COST us (they beat the market and we sat out). The gate's real purpose is
variance reduction around events, so a single event's sign is an anecdote — the value is the
distribution across many events, which this log accumulates.

Live capture appends here from the skip point; ``scripts/macro_gate_report.py`` backfills past events
from the scheduler log + daily run records and scores them. Both write the same schema.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date as _date

logger = logging.getLogger(__name__)

SHADOW_LOG_PATH = os.path.join("logs", "macro_gate_shadow.jsonl")


def capture(
    macro_event: str | None,
    candidates: list[dict] | None,
    *,
    today: str | None = None,
    mode: str | None = None,
    regime: str | None = None,
    vix: float | None = None,
    log_path: str | None = None,
) -> None:
    """Append one macro-skip event and its would-be buy candidates. Fail-safe: any error is logged and
    swallowed so instrumentation can never break a trading run."""
    path = log_path or SHADOW_LOG_PATH
    try:
        rows = [
            {
                "symbol": c["symbol"],
                "confidence": c.get("confidence"),
                "key_signal": c.get("key_signal"),
            }
            for c in (candidates or [])
            if isinstance(c, dict) and c.get("symbol")
        ]
        rec = {
            "date": today or _date.today().isoformat(),
            "mode": mode,
            "macro_event": macro_event or "(unspecified)",
            "regime": regime,
            "vix": vix,
            "candidates": rows,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as exc:  # noqa: BLE001 - telemetry must never break trading
        logger.warning(f"macro_gate_shadow.capture failed (non-fatal): {exc}")


def load(log_path: str | None = None) -> list[dict]:
    """Fail-safe read of the shadow log (empty list on any error / missing file)."""
    path = log_path or SHADOW_LOG_PATH
    rows: list[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, ValueError) as exc:
        logger.warning(f"macro_gate_shadow.load failed: {exc}")
        return []
    return rows


def score_event(
    candidates: list[dict],
    prices: dict[str, tuple[float | None, float | None]],
    *,
    min_confidence: int = 7,
) -> tuple[int, float | None]:
    """Equal-weight market-excess return of one event's would-be buys.

    ``candidates`` are the recorded rows; ``prices`` maps symbol -> (entry_px, exit_px) and MUST include
    ``"SPY"``. Only candidates with confidence >= ``min_confidence`` and usable prices are counted (the
    gate only blocks what would actually have been bought). Returns ``(n, mean_excess_pct)`` — the
    equal-weight mean of (name return - SPY return) in percent — or ``(0, None)`` if nothing is scorable.
    Negative mean ⇒ the blocked names lagged the market ⇒ the gate saved us.
    """
    spy = prices.get("SPY")
    if not spy or not spy[0] or spy[1] is None:
        return 0, None
    spy_ret = (spy[1] / spy[0] - 1.0) * 100.0
    excess: list[float] = []
    for c in candidates:
        if (c.get("confidence") or 0) < min_confidence:
            continue
        p = prices.get(c.get("symbol", ""))
        if not p or not p[0] or p[1] is None:
            continue
        excess.append((p[1] / p[0] - 1.0) * 100.0 - spy_ret)
    if not excess:
        return 0, None
    return len(excess), sum(excess) / len(excess)

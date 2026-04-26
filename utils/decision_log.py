"""
Per-trade AI decision log — captures Claude's full reasoning for every buy/sell candidate.
Written to logs/decisions.jsonl (append-only JSONL).

Fields per entry:
  ts             UTC timestamp
  date           trading date (YYYY-MM-DD)
  mode           open | midday | close
  symbol         ticker
  action         BUY | SELL | HOLD
  confidence     1-10
  reasoning      Claude's plain-English reasoning
  key_signal     signal type (buy candidates only)
  executed       True if the trade was actually placed
  market_summary overall market tone for that run
"""
from __future__ import annotations
import json
import os
import logging
from datetime import datetime, timezone
from config import LOG_DIR

logger = logging.getLogger(__name__)

_DECISIONS_PATH = os.path.join(LOG_DIR, "decisions.jsonl")


def _write(entry: dict):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_DECISIONS_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Decision log write failed: {e}")


def log_decisions(decisions: dict, mode: str, executed_symbols: set[str]):
    """
    Log every buy candidate and position decision from an AI analysis run.
    executed_symbols — symbols where an order was actually placed.
    """
    now = datetime.now(timezone.utc).isoformat()
    date = decisions.get("date", now[:10])
    market_summary = decisions.get("market_summary", "")

    for candidate in decisions.get("buy_candidates", []):
        sym = candidate.get("symbol", "")
        _write({
            "ts": now,
            "date": date,
            "mode": mode,
            "symbol": sym,
            "action": "BUY",
            "confidence": candidate.get("confidence"),
            "reasoning": candidate.get("reasoning", ""),
            "key_signal": candidate.get("key_signal", ""),
            "executed": sym in executed_symbols,
            "market_summary": market_summary,
        })

    for decision in decisions.get("position_decisions", []):
        sym = decision.get("symbol", "")
        _write({
            "ts": now,
            "date": date,
            "mode": mode,
            "symbol": sym,
            "action": decision.get("action", ""),
            "confidence": decision.get("confidence"),
            "reasoning": decision.get("reasoning", ""),
            "key_signal": None,
            "executed": sym in executed_symbols,
            "market_summary": market_summary,
        })


def load_decisions(n: int = 200) -> list[dict]:
    """Return the last n decision log entries."""
    if not os.path.exists(_DECISIONS_PATH):
        return []
    entries = []
    try:
        with open(_DECISIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return entries[-n:]

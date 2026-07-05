"""Shadow measurement: crowded-popper shorts (workshop v2 #1, shadow-gated).

The lottery_pop_short study (scripts/lottery_pop_short_backtest.py) found the short edge after a
>=+10% single-day pop lives ONLY in the FINRA-crowded names (top-quartile short-volume-ratio:
+0.86%/3d gross, t=3.7, survives 30%/yr borrow on the mean) — but year-consistency degrades under
borrow (4/9 at 15%/yr) and the live squeeze tail is unbounded, so it ships as a SHADOW logger, not
a live short. `capture()` records every crowded popper each run — read-only, no trades — so forward
evidence can accumulate; scoring reuses the eval pattern of the catalyst shadow. Output:
logs/shadow_popper_shorts.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime, timedelta

from config import LOG_DIR, STOCK_UNIVERSE

logger = logging.getLogger(__name__)

SHADOW_LOG_PATH = os.path.join(LOG_DIR, "shadow_popper_shorts.jsonl")
_POP_MIN_PCT = 10.0  # single-day return that defines a lottery pop


def _latest_svr() -> tuple[str, dict[str, float]]:
    """Most recent available FINRA SVR map for the universe (files lag ~1 session)."""
    from data.short_flow import get_day

    d = date.today()
    for _ in range(5):
        svr = get_day(d, symbols=set(STOCK_UNIVERSE))
        if svr:
            return d.isoformat(), svr
        d -= timedelta(days=1)
    return "", {}


def _symbols_logged_on(today: str) -> set[str]:
    """Symbols already captured for `today` so multiple daily runs don't double-log a name."""
    seen: set[str] = set()
    try:
        with open(SHADOW_LOG_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("date") == today:
                    seen.add(rec.get("symbol", ""))
    except FileNotFoundError:
        pass
    return seen


def capture(snapshots: list[dict], mode: str, run_id: str, today: str | None = None) -> int:
    """Log every >=+10% popper whose FINRA short-flow is in the universe top quartile.

    Read-only instrumentation (no trades, no experiment writes); returns rows written.
    """
    today = today or datetime.now(UTC).date().isoformat()
    try:
        svr_day, svr = _latest_svr()
        if not svr:
            return 0
        vals = sorted(svr.values())
        q3 = vals[int(len(vals) * 0.75)] if len(vals) >= 100 else None
        if q3 is None:
            return 0
        seen = _symbols_logged_on(today)
        rows = []
        for s in snapshots:
            sym = s.get("symbol", "")
            ret1 = s.get("ret_1d_pct")
            if not sym or sym in seen or ret1 is None or float(ret1) < _POP_MIN_PCT:
                continue
            sv = svr.get(sym)
            if sv is None or sv < q3:
                continue
            rows.append(
                {
                    "date": today,
                    "run_id": run_id,
                    "mode": mode,
                    "symbol": sym,
                    "ret_1d_pct": round(float(ret1), 2),
                    "svr": round(float(sv), 4),
                    "svr_q3": round(float(q3), 4),
                    "svr_day": svr_day,
                }
            )
            seen.add(sym)
        if rows:
            os.makedirs(os.path.dirname(SHADOW_LOG_PATH), exist_ok=True)
            with open(SHADOW_LOG_PATH, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        return len(rows)
    except Exception as exc:  # instrumentation must never block trading
        logger.warning(f"shadow_popper capture failed (non-fatal): {exc}")
        return 0

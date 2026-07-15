"""Shadow measurement: would catalyst shorts have edge OUTSIDE bear regimes?

The live bot enriches catalyst short signals (eps_estimate_cut, accounting_concern, insider selling,
analyst downgrade, …) onto the short universe on every cycle — but `scan_short_candidates` is
regime-gated (ADR-006 B1), so those names are never shorted outside bear regimes. We detect the
deteriorating fundamentals and act on them only defensively (exiting longs), never offensively.

`capture()` records every catalyst-flagged name each run, REGARDLESS of regime — read-only, no
trades, no experiment-observation writes — so we can later score the forward short return (net of
market beta + borrow) and decide whether to relax the regime gate for catalyst-class shorts before
the experiment PNR. It is called fail-safe from the run; the scoring lives in
scripts/eval_shadow_catalyst_shorts.py. Output: logs/shadow_catalyst_shorts.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from config import LOG_DIR

logger = logging.getLogger(__name__)

SHADOW_LOG_PATH = os.path.join(LOG_DIR, "shadow_catalyst_shorts.jsonl")

# Snapshot flag -> canonical catalyst short signal. The values must stay equal to
# signals.registry.CATALYST_SHORT_SIGNALS (a test enforces this), so the shadow measurement tracks
# exactly the catalyst signals the live scanner would route once the regime gate is lifted.
_FLAG_TO_SIGNAL: dict[str, str] = {
    "accounting_concern": "accounting_concern_short",
    "insider_sell_cluster": "insider_selling_short",
    "index_deletion": "index_deletion_short",
    "eps_estimate_cut": "eps_revision_down_short",
    "analyst_downgrade": "analyst_downgrade_signal",
    "guidance_negative": "guidance_downgrade",
    "secondary_offering": "secondary_offering_short",
}


def _catalyst_signals(snapshot: dict) -> list[str]:
    """Canonical catalyst short signals whose flag is set on this snapshot."""
    return sorted(sig for flag, sig in _FLAG_TO_SIGNAL.items() if snapshot.get(flag))


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


def capture(
    snapshots: list[dict],
    regime: str,
    mode: str,
    run_id: str,
    today: str | None = None,
) -> int:
    """Append a shadow record for each catalyst-flagged name not yet captured today; return the count.

    Records `entry_price` (the snapshot's current price) so the forward short return can be scored
    later. Read-only with respect to trading. Idempotent within a day (deduped by symbol).
    """
    today = today or datetime.now(UTC).date().isoformat()
    seen = _symbols_logged_on(today)
    rows: list[dict] = []
    for s in snapshots:
        sym = s.get("symbol")
        if not sym or sym in seen:
            continue
        signals = _catalyst_signals(s)
        price = s.get("current_price")
        if not signals or not price:
            continue
        seen.add(sym)
        rows.append(
            {
                "date": today,
                "ts": datetime.now(UTC).isoformat(),
                "mode": mode,
                "run_id": run_id,
                "regime": regime,
                "symbol": sym,
                "catalyst_signals": signals,
                "entry_price": float(price),
            }
        )

    if rows:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(SHADOW_LOG_PATH, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Shadow catalyst shorts: captured {len(rows)} name(s) (regime={regime})")
    return len(rows)


# ── forward-edge scoring (net of a REALISTIC borrow + slippage haircut) ──────────────────────────
# Shared by scripts/eval_shadow_catalyst_shorts.py and the weekly short-gate telemetry so both apply
# the same cost model. Borrow is a flat annualised assumption (no point-in-time fee feed): the names a
# catalyst short flags are disproportionately hard-to-borrow, so the honest read is a SENSITIVITY over
# several borrow rates, not a single number — the gross 3%/yr figure is an optimistic upper bound.

BEAR_REGIMES = frozenset({"DEFENSIVE_DOWNTREND", "STRESS_RISK_OFF", "HIGH_VOL_DOWNTREND"})


def net_short_return(
    stock_ret_pct: float,
    spy_ret_pct: float,
    *,
    borrow_annual_pct: float,
    hold_days: int,
    slippage_bps: float,
) -> float:
    """Net market-neutral short return (%) for one position over ``hold_days``.

    A short profits when the stock falls, so the raw short return is ``-stock_ret``; add SPY to make it
    ~market-neutral (isolate the stock-specific move), then subtract the borrow accrued over the hold
    and a round-trip execution slippage (``slippage_bps`` basis points, entry + exit)."""
    raw_short = -stock_ret_pct
    market_excess = raw_short + spy_ret_pct
    borrow = borrow_annual_pct * hold_days / 252.0
    return market_excess - borrow - slippage_bps / 100.0


def score_short_edge(
    observations: list[dict],
    *,
    borrow_annual_pct: float = 3.0,
    hold_days: int = 5,
    slippage_bps: float = 0.0,
) -> dict[str, tuple[int, float, float]]:
    """Aggregate net short returns per catalyst signal under one cost assumption.

    ``observations`` is a list of ``{"stock_ret", "spy_ret", "signals": [...]}`` (returns are raw long
    % over the hold; None-valued returns are skipped). Returns ``{key: (n, net_avg_pct, hit_pct)}`` for
    each catalyst signal plus the pooled key ``"__all__"``. A name flagged by several signals counts
    under each. Positive net_avg ⇒ shorting the flagged names paid after costs."""
    per: dict[str, list[float]] = {}
    for o in observations:
        if o.get("stock_ret") is None or o.get("spy_ret") is None:
            continue
        net = net_short_return(
            o["stock_ret"],
            o["spy_ret"],
            borrow_annual_pct=borrow_annual_pct,
            hold_days=hold_days,
            slippage_bps=slippage_bps,
        )
        per.setdefault("__all__", []).append(net)
        for sig in o.get("signals", []):
            per.setdefault(sig, []).append(net)
    out: dict[str, tuple[int, float, float]] = {}
    for key, vals in per.items():
        hit = sum(1 for v in vals if v > 0) / len(vals) * 100.0
        out[key] = (len(vals), sum(vals) / len(vals), hit)
    return out

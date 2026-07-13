"""Live experiment collection: capture each surfaced candidate as a point-in-time observation.

For every candidate the deterministic engine surfaces (passed the prefilter), record what is needed
for the veto/down-weight primary endpoint (EXPERIMENT.md section 1):

  - the point-in-time features it was decided on,
  - its material-context tags (experiment/material_context.py),
  - Arm 1, the deterministic baseline: the engine's rank score and rank,
  - Arm 3, the contextual AI's verdict: selected vs vetoed, and its confidence (the down-weight).

Forward outcomes and the fitted evidence scores are NOT computed here — they are backfilled offline
under the AsOfExpectancy discipline (known_at < decision_date), so this captures decision-time inputs
only and never looks ahead. Pure row-building (build_observation) is separated from the JSONL append.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable
from dataclasses import asdict

from experiment.dataset import assemble_row
from experiment.material_context import detect_material_context

logger = logging.getLogger(__name__)

# v2: arm3 decision-maker moved from claude-sonnet-4-6 to claude-opus-4-8 (a different model = a
# different arm), so Opus-era observations are stamped v2 to keep them separable from the v1 sonnet
# pilot. The v1 observation/scored logs are archived at the swap; accumulation restarts under v2.
OBSERVATIONS_VERSION = "v2"
OBSERVATIONS_PATH = os.path.join("logs", "experiment_observations.jsonl")

# Decision-time feature view: the point-in-time inputs the engine/AI saw. atr + current_price anchor
# the forward-return backfill; the rest are the technical state and the material-context flags.
_FEATURE_KEYS: tuple[str, ...] = (
    "current_price",
    "atr",
    "rsi_14",
    "bb_pct",
    "vol_ratio",
    "ret_5d_pct",
    "ret_10d_pct",
    "rs_rank_pct",
    "weekly_trend_up",
    "earnings_gap_pct",
    "sector",
    "sector_momentum_rank",  # A3.1: sector's 20d-momentum rank (for measuring the sector gate)
    "sector_gate_pass",  # A3.1: did the (advisory) sector-momentum gate pass this candidate?
    "guidance_positive",
    "guidance_negative",
    "activist_filing",
    "secondary_offering",
    "ma_event",
    "accounting_concern",
    "regulatory_event",
    "index_change",
    "insider_cluster",
    "short_ratio",
    "high_short_interest",
)


def _json_safe(value: object) -> object:
    """Coerce a feature value to something JSON-serialisable (NaN/inf -> None)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool | int | str) or value is None:
        return value
    try:
        f = float(value)  # type: ignore[arg-type]  # duck-typed: numpy scalars, Decimals, etc.
    except (TypeError, ValueError):
        return str(value)
    return f if math.isfinite(f) else None


def _features(candidate: dict) -> dict:
    return {k: _json_safe(candidate[k]) for k in _FEATURE_KEYS if k in candidate}


def _base_record(snapshot: dict, decision_date: str) -> dict:
    """The point-in-time row shared by buy and sell observations (features, fired signals, material
    context, blinded view, split tag; expectancy/forward_r backfilled offline)."""
    fired = list(snapshot.get("matched_signals") or snapshot.get("signals") or [])
    row = assemble_row(
        symbol=snapshot["symbol"],
        date=decision_date,
        features=_features(snapshot),
        fired_signals=fired,
        material_context=detect_material_context(snapshot),
        expectancy={},  # backfilled offline (AsOfExpectancy)
        forward_r_value=None,  # not known at decision time
    )
    return asdict(row)


def build_observation(
    candidate: dict,
    *,
    decision_date: str,
    selected: bool,
    confidence: float | None,
    deterministic_score: float | None,
    deterministic_rank: int | None,
    run_id: str,
    mode: str,
    market_context: dict | None = None,
) -> dict:
    """Build a buy-side observation: an engine-surfaced candidate the AI selected or vetoed."""
    record = _base_record(candidate, decision_date)
    record["extra"] = {
        "version": OBSERVATIONS_VERSION,
        "run_id": run_id,
        "mode": mode,
        "decision_type": "buy_candidate",
        "arm1_deterministic_score": deterministic_score,
        "arm1_deterministic_rank": deterministic_rank,
        "arm3_ai_selected": selected,
        "arm3_ai_confidence": confidence,
        "market_context": market_context or {},
    }
    return record


def build_sell_observation(
    snapshot: dict,
    *,
    decision_date: str,
    action: str | None,
    confidence: float | None,
    run_id: str,
    mode: str,
    market_context: dict | None = None,
) -> dict:
    """Build a sell-side observation: a held position the AI evaluated for exit (HOLD/SELL).

    The forward return (backfilled, long-direction) measures the decision: a good SELL precedes a
    falling position, a good HOLD a rising one. The deterministic exit baseline (Arm 1) is deferred.
    """
    record = _base_record(snapshot, decision_date)
    record["extra"] = {
        "version": OBSERVATIONS_VERSION,
        "run_id": run_id,
        "mode": mode,
        "decision_type": "held_position",
        "arm3_ai_action": action,
        "arm3_ai_confidence": confidence,
        "market_context": market_context or {},
    }
    return record


def append_observations(records: list[dict], path: str | None = None) -> int:
    """Append observation records as JSON lines. Returns the number written.

    ``path`` is resolved at call time (default OBSERVATIONS_PATH) so tests can redirect the live log
    via the module constant — see the autouse fixture in conftest.py.
    """
    if not records:
        return 0
    target = path if path is not None else OBSERVATIONS_PATH
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "a") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return len(records)


def log_run_observations(
    candidates: list[dict],
    *,
    buy_candidates: list[dict],
    ranked: list[dict],
    decision_date: str,
    run_id: str,
    mode: str,
    market_context: dict,
    score_fn: Callable[[dict], float],
    path: str | None = None,
) -> int:
    """Build and append observations for one decision run; return the number logged.

    Fail-safe: this is experiment instrumentation and must never block trading, so any error is logged
    and swallowed (returns 0). ``ranked`` is the candidates ordered by the deterministic score (its
    index gives Arm 1's rank); ``buy_candidates`` are the AI's picks (Arm 3), carrying ``confidence``.
    """
    try:
        confidence_by_sym = {b["symbol"]: b.get("confidence") for b in buy_candidates}
        selected = set(confidence_by_sym)
        rank_by_sym = {c["symbol"]: i + 1 for i, c in enumerate(ranked)}
        records = [
            build_observation(
                c,
                decision_date=decision_date,
                selected=c["symbol"] in selected,
                confidence=confidence_by_sym.get(c["symbol"]),
                deterministic_score=score_fn(c),
                deterministic_rank=rank_by_sym.get(c["symbol"]),
                run_id=run_id,
                mode=mode,
                market_context=market_context,
            )
            for c in candidates
        ]
        return append_observations(records, path)
    except Exception as exc:
        logger.warning("experiment observation logging failed (non-fatal): %s", exc)
        return 0


def log_sell_observations(
    held_snapshots: list[dict],
    position_decisions: list[dict],
    *,
    decision_date: str,
    run_id: str,
    mode: str,
    market_context: dict,
    path: str | None = None,
) -> int:
    """Build and append sell-side observations for held positions the AI evaluated; return the count.

    Fail-safe (must never block trading): records each held position with the AI's HOLD/SELL action
    and confidence (Arm 3). ``position_decisions`` are the AI's per-position calls, keyed by symbol.
    """
    try:
        action_by_sym = {d["symbol"]: d for d in position_decisions}
        records = [
            build_sell_observation(
                snap,
                decision_date=decision_date,
                action=action_by_sym.get(snap["symbol"], {}).get("action"),
                confidence=action_by_sym.get(snap["symbol"], {}).get("confidence"),
                run_id=run_id,
                mode=mode,
                market_context=market_context,
            )
            for snap in held_snapshots
        ]
        return append_observations(records, path)
    except Exception as exc:
        logger.warning("experiment sell-observation logging failed (non-fatal): %s", exc)
        return 0

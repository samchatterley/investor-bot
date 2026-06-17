import logging

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_DRAWDOWN_PCT = -12.0  # halt buys if portfolio drops this much from 5-day peak
MAX_DAILY_LOSS_PCT = -5.0  # close everything if down this much on the day
MAX_SECTOR_POSITIONS = 2  # max open positions in any single sector


def check_circuit_breaker(portfolio_history: list[dict]) -> tuple[bool, float]:
    """
    Returns (is_triggered, drawdown_pct).
    Triggered when the current portfolio value has dropped more than
    CIRCUIT_BREAKER_DRAWDOWN_PCT from its peak over the last 5 days.
    """
    if len(portfolio_history) < 2:
        return False, 0.0

    # Collapse to one value per calendar day (the day's last record) so the lookback is ~5 trading
    # days, not ~5 runs (audit A1.2 — with 4 runs/day, [-5:] records was only ~1.25 days and missed
    # slow multi-day bleeds). Undated records (e.g. unit tests) are kept distinct via a per-index key.
    _by_day: dict[str, float] = {}
    for _i, _r in enumerate(portfolio_history):
        try:
            _day = _r["date"][:10] if _r.get("date") else f"_{_i}"
            _by_day[_day] = _r["account_after"]["portfolio_value"]
        except (KeyError, TypeError):
            continue
    recent_values = [_by_day[k] for k in list(_by_day)[-5:]]
    # Exclude records with implausibly small values — guards against corrupted
    # test/placeholder records that would produce a false -99.9% drawdown signal.
    # Threshold scales with actual account size (half the peak floored at $10)
    # so the circuit breaker works correctly in SMALL_ACCOUNT_MODE (~$150 accounts).
    _peak_raw = max(recent_values) if recent_values else 0.0
    _MIN_PLAUSIBLE = max(10.0, _peak_raw * 0.5)
    values = [v for v in recent_values if v >= _MIN_PLAUSIBLE]
    if len(values) < 2:
        return False, 0.0
    peak = max(values[:-1])
    current = values[-1]
    if peak <= 0:
        return False, 0.0
    drawdown = (current / peak - 1) * 100
    triggered = drawdown <= CIRCUIT_BREAKER_DRAWDOWN_PCT
    if triggered:
        logger.warning(f"Circuit breaker triggered: {drawdown:.1f}% drawdown from {peak:.2f} peak")
    return triggered, round(drawdown, 2)


def check_daily_loss(value_at_open: float, value_now: float) -> tuple[bool, float]:
    """
    Returns (is_triggered, loss_pct).
    Triggered when today's loss exceeds MAX_DAILY_LOSS_PCT.
    """
    if value_at_open <= 0:
        return False, 0.0
    loss_pct = (value_now / value_at_open - 1) * 100
    triggered = loss_pct <= MAX_DAILY_LOSS_PCT
    if triggered:
        logger.warning(f"Daily loss limit triggered: {loss_pct:.1f}% loss today")
    return triggered, round(loss_pct, 2)


def check_vix_stop_adjustment(vix: float | None) -> float:
    """
    Return an adjusted trailing stop percentage based on VIX level.
    Higher VIX = wider stop to avoid noise shake-outs.
    """
    if vix is None:
        return 4.0
    if vix > 35:
        return 7.0  # very high vol — wide stop
    if vix > 25:
        return 5.5  # elevated vol
    if vix > 18:
        return 4.0  # normal
    return 3.0  # low vol — tight stop


def validate_buy_candidates(
    candidates: list[dict],
    held_symbols: set[str],
    sector_map_fn,
    max_per_sector: int = MAX_SECTOR_POSITIONS,
) -> list[dict]:
    """
    Filter buy candidates:
    - Remove symbols already held
    - Enforce sector concentration cap (max_per_sector per sector)
    - Remove ETFs from sector concentration check
    Returns filtered, ordered list.
    """
    sector_counts: dict[str, int] = {}
    for sym in held_symbols:
        s = sector_map_fn(sym)
        if s not in ("ETF", "Unknown"):
            sector_counts[s] = sector_counts.get(s, 0) + 1

    valid = []
    for c in candidates:
        sym = c["symbol"]
        if sym in held_symbols:
            continue
        sector = sector_map_fn(sym)
        if sector not in ("ETF", "Unknown"):
            count = sector_counts.get(sector, 0)
            if count >= max_per_sector:
                logger.info(f"Skipping {sym} — sector '{sector}' already at {count} positions")
                continue
            sector_counts[sector] = count + 1
        valid.append(c)
    return valid

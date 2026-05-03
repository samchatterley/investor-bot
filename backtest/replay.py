"""Historical replay harness.

Downloads data once, then simulates the live pipeline day-by-day using
point-in-time slices so there is no lookahead.  Claude is called for real
on each simulation date; fills are applied at the next day's open.
"""

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

import config
from analysis import ai_analyst
from data import market_data
from execution import stock_scanner

logger = logging.getLogger(__name__)

# Enough history before the replay window to warm up all indicators
_WARMUP_DAYS = 252


def _build_preloaded(
    symbols: list[str], fetch_start: date, end_date: date
) -> dict[str, pd.DataFrame]:
    """Download full OHLCV history for all symbols in one batch call."""
    tickers = list({*symbols, "SPY", "^VIX"})
    raw = yf.download(
        tickers,
        start=fetch_start.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    preloaded: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in tickers:
            try:
                df = raw.xs(sym, axis=1, level=1).dropna(how="all")
                if not df.empty:
                    preloaded[sym] = df
            except KeyError:
                logger.warning(f"replay: no data downloaded for {sym}")
    else:
        # Single ticker — raw has no MultiIndex
        sym = tickers[0]
        if not raw.empty:
            preloaded[sym] = raw

    return preloaded


def _compute_regime(preloaded: dict[str, pd.DataFrame], as_of: str) -> dict:
    """Derive the 4-state market regime from preloaded SPY data up to as_of."""
    spy_df = preloaded.get("SPY")
    if spy_df is None:
        return {"is_bearish": False, "spy_change_pct": 0.0, "spy_5d_pct": 0.0, "regime": "UNKNOWN"}
    try:
        sliced = spy_df[spy_df.index <= pd.Timestamp(as_of)]["Close"].dropna()
        if len(sliced) < 6:
            return {
                "is_bearish": False,
                "spy_change_pct": 0.0,
                "spy_5d_pct": 0.0,
                "regime": "UNKNOWN",
            }

        spy_1d = float((sliced.iloc[-1] / sliced.iloc[-2] - 1) * 100)
        spy_5d = float((sliced.iloc[-1] / sliced.iloc[-6] - 1) * 100)

        vix_df = preloaded.get("^VIX")
        vix: float | None = None
        if vix_df is not None:
            vix_sliced = vix_df[vix_df.index <= pd.Timestamp(as_of)]["Close"].dropna()
            if not vix_sliced.empty:
                vix = float(vix_sliced.iloc[-1])

        is_bearish = spy_1d <= config.BEAR_MARKET_SPY_THRESHOLD
        if is_bearish:
            regime = "BEAR_DAY"
        elif vix is not None and vix > 25 and spy_5d < -3:
            regime = "HIGH_VOL"
        elif spy_5d > 2 and spy_1d > 0:
            regime = "BULL_TRENDING"
        else:
            regime = "CHOPPY"

        return {
            "is_bearish": is_bearish,
            "spy_change_pct": round(spy_1d, 2),
            "spy_5d_pct": round(spy_5d, 2),
            "regime": regime,
            "vix": vix,
        }
    except Exception as e:
        logger.error(f"replay: regime computation failed for {as_of}: {e}")
        return {"is_bearish": False, "spy_change_pct": 0.0, "spy_5d_pct": 0.0, "regime": "UNKNOWN"}


def run_historical_replay(
    symbols: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    max_hold_days: int = 3,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run the full live pipeline historically using pre-downloaded data.

    On each simulation date T:
      1. Slice all preloaded DataFrames to <= T (point-in-time isolation).
      2. Call get_market_snapshots with the preloaded slice.
      3. Call ai_analyst.get_trading_decisions with the resulting snapshots.
      4. Fill buys at T+1 Open price; fill sells at T+1 Open price.

    Returns a summary dict with per-day records and aggregate stats.
    """
    universe = symbols if symbols is not None else config.STOCK_UNIVERSE

    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date() if end_date else date.today()
    fetch_start = start - timedelta(days=_WARMUP_DAYS)

    logger.info(f"replay: downloading data {fetch_start} → {end} for {len(universe)} symbols")
    preloaded = _build_preloaded(universe, fetch_start, end)
    if not preloaded:
        logger.error("replay: no data downloaded — aborting")
        return {"error": "no data downloaded"}

    # Build a sorted list of trading dates within the window
    spy_df = preloaded.get("SPY")
    if spy_df is None:
        logger.error("replay: SPY not in preloaded data — aborting")
        return {"error": "SPY data missing"}

    trading_dates = [d.date() for d in spy_df.index if start <= d.date() <= end]
    if not trading_dates:
        return {"error": "no trading dates in range"}

    # Simulation state
    cash = initial_capital
    positions: dict[str, dict] = {}  # symbol -> {shares, entry_price, entry_date}
    daily_records: list[dict] = []
    all_trades: list[dict] = []

    slippage_factor = 1.0 + (config.SLIPPAGE_BPS + config.SPREAD_BPS / 2) / 10_000
    sell_factor = 1.0 - (config.SLIPPAGE_BPS + config.SPREAD_BPS / 2) / 10_000

    for i, sim_date in enumerate(trading_dates):
        as_of = sim_date.strftime("%Y-%m-%d")

        # ── Market snapshots (point-in-time) ──────────────────────────────────
        snapshots = market_data.get_market_snapshots(
            universe,
            config.LOOKBACK_DAYS,
            preloaded=preloaded,
            as_of=as_of,
        )
        if not snapshots:
            logger.warning(f"replay {as_of}: no snapshots — skipping")
            continue

        # ── Regime ───────────────────────────────────────────────────────────
        regime = _compute_regime(preloaded, as_of)

        # ── Pre-filter ────────────────────────────────────────────────────────
        held_syms = set(positions.keys())
        held_snaps = [s for s in snapshots if s["symbol"] in held_syms]
        candidate_snaps = [s for s in snapshots if s["symbol"] not in held_syms]
        filtered = stock_scanner.prefilter_candidates(candidate_snaps)
        ai_snapshots = held_snaps + filtered

        # ── Claude decisions ──────────────────────────────────────────────────
        open_pos_list = [
            {"symbol": sym, "unrealized_plpc": 0.0, "market_value": p["shares"] * p["entry_price"]}
            for sym, p in positions.items()
        ]
        portfolio_value = cash + sum(
            p["shares"]
            * next((s["current_price"] for s in snapshots if s["symbol"] == sym), p["entry_price"])
            for sym, p in positions.items()
        )

        try:
            decisions = ai_analyst.get_trading_decisions(
                snapshots=ai_snapshots,
                current_positions=open_pos_list,
                available_cash=cash,
                portfolio_value=portfolio_value,
                news_by_symbol={},
                track_record={},
                market_regime=regime,
                position_ages={
                    sym: (sim_date - p["entry_date"]).days for sym, p in positions.items()
                },
                stale_positions=[],
                vix=regime.get("vix"),
                sector_performance={},
                sentiment={},
                earnings_risk={},
                macro_risk={"is_high_risk": False, "event": ""},
                leading_sectors=[],
                options_signals={},
                lessons="",
                run_id=f"replay-{as_of}",
            )
        except Exception as e:
            logger.error(f"replay {as_of}: AI call failed: {e}")
            decisions = {}

        if dry_run:
            # In dry_run mode record what Claude said but don't touch positions
            daily_records.append(
                {
                    "date": as_of,
                    "portfolio_value": round(portfolio_value, 2),
                    "cash": round(cash, 2),
                    "open_positions": len(positions),
                    "regime": regime.get("regime"),
                    "decisions": decisions,
                    "trades": [],
                }
            )
            continue

        # ── Fill decisions at next-day open ───────────────────────────────────
        next_date_idx = i + 1
        if next_date_idx >= len(trading_dates):
            break
        next_date = trading_dates[next_date_idx]
        next_as_of = next_date.strftime("%Y-%m-%d")

        day_trades: list[dict] = []

        # Sells
        sell_symbols = {
            d["symbol"]
            for d in decisions.get("position_decisions", [])
            if d.get("action") == "SELL"
        }
        # Stale exits
        for sym, pos in list(positions.items()):
            if (sim_date - pos["entry_date"]).days >= max_hold_days:
                sell_symbols.add(sym)

        for sym in sell_symbols:
            if sym not in positions:
                continue
            pos = positions[sym]
            sym_df = preloaded.get(sym)
            exit_px = None
            if sym_df is not None:
                next_rows = sym_df[sym_df.index.date == next_date]
                if not next_rows.empty and "Open" in next_rows.columns:
                    exit_px = float(next_rows["Open"].iloc[0]) * sell_factor
            if exit_px is None:
                # fallback: last close as_of sim_date
                snap = next((s for s in snapshots if s["symbol"] == sym), None)
                exit_px = snap["current_price"] * sell_factor if snap else pos["entry_price"]

            pnl_pct = (exit_px / pos["entry_price"] - 1) * 100
            proceeds = pos["shares"] * exit_px
            cash += proceeds
            trade = {
                "date": next_as_of,
                "symbol": sym,
                "action": "SELL",
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(exit_px, 4),
                "shares": round(pos["shares"], 4),
                "pnl_pct": round(pnl_pct, 2),
                "hold_days": (sim_date - pos["entry_date"]).days + 1,
            }
            day_trades.append(trade)
            all_trades.append(trade)
            del positions[sym]

        # Buys
        skip_buys = regime.get("is_bearish") or len(positions) >= max_positions
        if not skip_buys:
            slots = max_positions - len(positions)
            buys = sorted(
                [
                    c
                    for c in decisions.get("buy_candidates", [])
                    if c["confidence"] >= config.MIN_CONFIDENCE
                ],
                key=lambda x: x["confidence"],
                reverse=True,
            )[:slots]

            for candidate in buys:
                sym = candidate["symbol"]
                if sym in positions:
                    continue

                sym_df = preloaded.get(sym)
                entry_px = None
                if sym_df is not None:
                    next_rows = sym_df[sym_df.index.date == next_date]
                    if not next_rows.empty and "Open" in next_rows.columns:
                        entry_px = float(next_rows["Open"].iloc[0]) * slippage_factor

                if entry_px is None or entry_px <= 0:
                    continue

                notional = min(cash * 0.9 / max(slots, 1), cash * config.MAX_POSITION_WEIGHT)
                if notional / entry_px < 1.0:
                    continue  # sub-share guard

                shares = notional / entry_px
                cost = shares * entry_px
                if cost > cash:
                    continue
                cash -= cost
                positions[sym] = {
                    "shares": shares,
                    "entry_price": entry_px,
                    "entry_date": next_date,
                    "signal": candidate.get("key_signal", "unknown"),
                    "confidence": candidate["confidence"],
                }
                trade = {
                    "date": next_as_of,
                    "symbol": sym,
                    "action": "BUY",
                    "entry_price": round(entry_px, 4),
                    "shares": round(shares, 4),
                    "notional": round(cost, 2),
                    "signal": candidate.get("key_signal", "unknown"),
                    "confidence": candidate["confidence"],
                }
                day_trades.append(trade)
                all_trades.append(trade)

        portfolio_value = cash + sum(
            p["shares"]
            * next((s["current_price"] for s in snapshots if s["symbol"] == sym), p["entry_price"])
            for sym, p in positions.items()
        )
        daily_records.append(
            {
                "date": as_of,
                "portfolio_value": round(portfolio_value, 2),
                "cash": round(cash, 2),
                "open_positions": len(positions),
                "regime": regime.get("regime"),
                "trades": day_trades,
            }
        )

    # ── Close all remaining positions at end ──────────────────────────────────
    last_date = trading_dates[-1].strftime("%Y-%m-%d") if trading_dates else end_date
    for sym, pos in list(positions.items()):
        snap_list = market_data.get_market_snapshots([sym], 2, preloaded=preloaded, as_of=last_date)
        exit_px = snap_list[0]["current_price"] * sell_factor if snap_list else pos["entry_price"]
        cash += pos["shares"] * exit_px
        all_trades.append(
            {
                "date": last_date,
                "symbol": sym,
                "action": "SELL",
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(exit_px, 4),
                "shares": round(pos["shares"], 4),
                "pnl_pct": round((exit_px / pos["entry_price"] - 1) * 100, 2),
                "hold_days": (pd.Timestamp(last_date).date() - pos["entry_date"]).days,
            }
        )

    final_value = cash
    total_return_pct = round((final_value / initial_capital - 1) * 100, 2)
    completed_sells = [t for t in all_trades if t["action"] == "SELL" and "pnl_pct" in t]
    win_rate = (
        round(sum(1 for t in completed_sells if t["pnl_pct"] > 0) / len(completed_sells) * 100, 1)
        if completed_sells
        else 0.0
    )

    return {
        "start_date": start_date,
        "end_date": last_date,
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return_pct": total_return_pct,
        "total_trades": len(completed_sells),
        "win_rate_pct": win_rate,
        "daily_records": daily_records,
        "all_trades": all_trades,
    }

"""Short-side universe management.

Provides two things:
1. STATIC_SHORT_UNIVERSE — ~300 sector-diverse stocks biased toward
   laggards, cyclicals, and fundamentally weak names.  Used as the
   backtest short universe and as a live fallback when Alpaca's
   easy-to-borrow list is unavailable.

2. get_short_universe(client) — live path; queries Alpaca for all
   tradable, easy-to-borrow, non-OTC assets and returns their symbols.
   Falls back to STATIC_SHORT_UNIVERSE on any failure.

3. scan_short_universe(symbols) — downloads OHLCV data, computes
   cross-sectional RS ranks (today and 10 days ago), and returns
   enriched snapshot dicts ready for scan_short_candidates().
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Static fallback universe ──────────────────────────────────────────────────
# ~300 sector-diverse stocks biased toward laggards and cyclicals.
# Deliberately EXCLUDES mega-cap tech (NVDA, AAPL, MSFT, etc.) — institutional
# buying support on every dip makes those structurally hard to short.
# Instead: legacy tech, cyclicals at cycle peak, financials, commodity names,
# Chinese ADRs, airlines, cruise lines, retail in secular decline, and
# regional/specialty sectors with known headwinds.
STATIC_SHORT_UNIVERSE: list[str] = list(
    dict.fromkeys(
        [
            # Legacy / mature tech (patent cliff, secular decline)
            "INTC",
            "IBM",
            "HPQ",
            "HPE",
            "WDC",
            "STX",
            "CSCO",
            "DELL",
            "NTAP",
            "NCR",
            "EXPE",
            "EBAY",
            "LYFT",
            "SNAP",
            "PINS",
            "TWTR",  # may be delisted — excluded automatically by Alpaca
            # Semiconductors at cycle peak
            "MU",
            "ON",
            "SWKS",
            "QCOM",
            "TXN",
            "WOLF",
            "MCHP",
            "AMKR",
            # Chinese ADRs (regulatory + delisting risk)
            "BABA",
            "JD",
            "PDD",
            "BIDU",
            "NIO",
            "XPEV",
            "LI",
            "DIDI",
            "BILI",
            "NTES",
            "VIPS",
            "TAL",
            "EDU",
            "LAIX",
            # Airlines (macro-cyclical, high fuel exposure)
            "AAL",
            "UAL",
            "DAL",
            "JBLU",
            "LUV",
            "HA",
            "SKYW",
            "ALK",
            # Cruise lines & travel (leverage + macro sensitivity)
            "CCL",
            "RCL",
            "NCLH",
            "EXPE",
            "TRIP",
            "ABNB",
            # Autos (EV transition stress + cyclical)
            "F",
            "GM",
            "STLA",
            "RIVN",
            "LCID",
            "GOEV",
            "WKHS",
            "FSR",
            # Retail in secular decline
            "M",
            "JWN",
            "KSS",
            "BBY",
            "GPS",
            "ANF",
            "PRGO",
            "SIG",
            "BBWI",
            "ROST",
            "TJX",
            "BJ",
            # Energy — oil & gas cyclicals
            "OXY",
            "DVN",
            "MRO",
            "APA",
            "HAL",
            "SLB",
            "BKR",
            "NOV",
            "HP",
            "RIG",
            "VAL",
            "DO",
            "NE",
            # Clean energy at valuation extremes
            "ENPH",
            "SEDG",
            "FSLR",
            "RUN",
            "PLUG",
            "BE",
            "NOVA",
            "ARRY",
            # Materials & metals (commodity cycle exposure)
            "CLF",
            "X",
            "AA",
            "NUE",
            "STLD",
            "MT",
            "FCX",
            "SCCO",
            "VALE",
            "RIO",
            "BHP",
            "MP",
            # Legacy media & entertainment
            "PARA",
            "WBD",
            "FOXA",
            "DIS",
            "NYT",
            "NWSA",
            "IPG",
            "OMC",
            "AMC",
            # Regional & specialty banks (SVB contagion risk, rate exposure)
            "ZION",
            "CMA",
            "PACW",
            "WAL",
            "HBAN",
            "RF",
            "KEY",
            "FITB",
            "CFG",
            "SBNY",
            "SI",
            "SIVB",  # may be delisted
            # Office REITs (remote work structural headwind)
            "SLG",
            "VNO",
            "HIW",
            "PDM",
            "OFC",
            "CUZ",
            "BXP",
            "KRC",
            # Biotech / pharma with near-term catalysts
            "MRNA",
            "PFE",
            "VTRS",
            "BMY",
            "JNJ",
            "ABBV",
            "MRK",
            "AGN",
            "PRGO",
            "ENDP",
            "CRON",
            "CGC",
            "ACB",
            # Telecom & legacy infrastructure
            "T",
            "VZ",
            "LUMN",
            "FYBR",
            "SHEN",
            "USM",
            "SATS",
            # Gaming & casinos
            "MGM",
            "CZR",
            "WYNN",
            "LVS",
            "PENN",
            "DKNG",
            "RSI",
            # Consumer discretionary / specialty
            "NKE",
            "LULU",
            "RH",
            "W",
            "CHWY",
            "PTON",
            "XRAY",
            "VSCO",
            "REAL",
            "OSTK",
            # Fintech at stretched valuations
            "PYPL",
            "AFRM",
            "SOFI",
            "HOOD",
            "UPST",
            "LC",
            "PROG",
            "OPFI",
            # Crypto-adjacent (high beta, structurally volatile)
            "MSTR",
            "MARA",
            "RIOT",
            "CLSK",
            "BTBT",
            "HUT",
            "CIFR",
            "COIN",
            # SPACs / de-SPACs (secular headwind)
            "NKLA",
            "HYLN",
            "RIDE",
            "SPCE",
            "BARK",
            "PSFE",
            # Industrials at cycle peak / headwinds
            "GE",
            "BA",
            "HON",
            "MMM",
            "DE",
            "CAT",
            "CMI",
            # Healthcare equipment & services
            "HUM",
            "CVS",
            "WBA",
            "RAD",
            "TDOC",
            "AMWL",
            "HIMS",
            # Food & consumer staples under cost pressure
            "MCD",
            "SYY",
            "SBUX",
            "YUM",
            "QSR",
            "DPZ",
            "CAKE",
        ]
    )
)


def detect_failed_gapdown(
    opens: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    gap_pct_max: float = -7.0,
    lookback: int = 7,
) -> dict:
    """Detect a recent earnings/news gap-down whose reflexive bounce has failed.

    This is the live, OHLCV-only detector behind the ``post_earnings_gapdown_failed_bounce``
    short signal. A single-day drop of ≥7% is almost always an earnings/guidance reaction;
    the *failed-bounce* condition (price subsequently breaking below the gap bar's low) is the
    timing filter that removes dead-cat-bounce losses the naive gap-day short suffers.

    Returns a dict with:
      earnings_gap_pct : float | None  — the gap on the detected gap bar, or None if no
                                         qualifying gap-down occurred within ``lookback`` bars.
      gap_failed_bounce: bool          — True when the latest close is below the gap bar's low
                                         AND at least one bar has printed since the gap.
      vol_ratio        : float         — latest volume / trailing-20 average (1.0 if too short).
    """
    n = len(closes)
    result = {"earnings_gap_pct": None, "gap_failed_bounce": False, "vol_ratio": 1.0}
    if n < 2:
        return result

    # vol_ratio on the latest (continuation) bar.
    if n >= 3:
        trailing = volumes[max(0, n - 21) : n - 1]
        avg_vol = sum(trailing) / len(trailing) if trailing else 0.0
        if avg_vol > 0:
            result["vol_ratio"] = volumes[-1] / avg_vol

    # Scan back over the lookback window for the most recent qualifying gap-down bar.
    earliest = max(1, n - lookback)
    for i in range(n - 1, earliest - 1, -1):
        prev_close = closes[i - 1]
        if prev_close <= 0:
            continue
        gap_pct = (opens[i] - prev_close) / prev_close * 100
        if gap_pct <= gap_pct_max:
            result["earnings_gap_pct"] = round(gap_pct, 4)
            # Failed bounce: a later bar broke below the gap bar's low (continuation lower).
            if i < n - 1 and closes[-1] < lows[i]:
                result["gap_failed_bounce"] = True
            return result

    return result


def get_short_universe(client, _retries: int = 2, _retry_delay: float = 3.0) -> list[str]:
    """Return STATIC_SHORT_UNIVERSE symbols that are currently easy-to-borrow on Alpaca.

    Uses Alpaca's asset list to filter STATIC_SHORT_UNIVERSE down to symbols that
    are verified borrowable today. Falls back to the full STATIC_SHORT_UNIVERSE if
    Alpaca is unavailable.

    Retries up to _retries times on connection errors before falling back.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + _retries):
        try:
            assets = client.get_all_assets()
            etb = {
                a.symbol
                for a in assets
                if getattr(a, "tradable", False)
                and getattr(a, "easy_to_borrow", False)
                and getattr(a, "exchange", "") not in ("OTC",)
            }
            verified = [s for s in STATIC_SHORT_UNIVERSE if s in etb]
            logger.info(
                f"Alpaca easy-to-borrow universe: {len(etb)} symbols"
                f" — {len(verified)}/{len(STATIC_SHORT_UNIVERSE)} static candidates verified"
            )
            return verified if verified else STATIC_SHORT_UNIVERSE
        except Exception as e:
            last_exc = e
            if attempt < _retries:
                logger.warning(
                    f"Alpaca asset discovery attempt {attempt + 1} failed, retrying: {e}"
                )
                time.sleep(_retry_delay)
    logger.warning(f"Alpaca asset discovery failed, using static universe: {last_exc}")
    return STATIC_SHORT_UNIVERSE


def scan_short_universe(
    symbols: list[str],
    lookback_days: int = 30,
) -> list[dict]:
    """Download OHLCV data, compute cross-sectional RS ranks, and return short snapshots.

    Each returned snapshot dict includes:
      symbol, current_price, ret_5d_pct, ret_20d_pct, avg_volume,
      rs_rank_pct (percentile rank today), rs_rank_pct_10d_ago.

    Symbols with fewer than 15 trading days of data are excluded.
    """
    if not symbols:
        return []

    period = f"{lookback_days}d"
    try:
        raw = yf.download(
            symbols,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as e:
        logger.error(f"Short universe download failed: {e}")
        return []

    if raw.empty:
        return []

    close_raw = raw["Close"]
    volume_raw = raw["Volume"]
    open_raw = raw["Open"]
    low_raw = raw["Low"]

    # Single-symbol download returns a Series; promote to DataFrame with ticker column
    if isinstance(close_raw, pd.Series):
        sym = symbols[0]
        close_df = close_raw.to_frame(name=sym).dropna(how="all")
        volume_df = volume_raw.to_frame(name=sym).dropna(how="all")
        open_df = open_raw.to_frame(name=sym)
        low_df = low_raw.to_frame(name=sym)
    else:
        close_df = close_raw.dropna(how="all")
        volume_df = volume_raw.dropna(how="all")
        open_df = open_raw
        low_df = low_raw

    if len(close_df) < 15:
        logger.warning("Short universe: fewer than 15 bars — skipping")
        return []

    # Cross-sectional 20d return for RS rank
    ret_20d = (close_df / close_df.shift(20) - 1) * 100

    # Today and 10-trading-days-ago RS ranks (percentile within universe)
    today_ret = ret_20d.iloc[-1]
    rs_rank_today = today_ret.rank(pct=True, na_option="keep") * 100

    # 10 trading days ago: use iloc[-11] as the "10d ago" close row
    lag_index = max(0, len(ret_20d) - 11)
    lag_ret = ret_20d.iloc[lag_index] if len(ret_20d) > 10 else ret_20d.iloc[0]
    rs_rank_lag10 = lag_ret.rank(pct=True, na_option="keep") * 100

    # Per-symbol 5d and 20d returns
    ret_5d = (
        (close_df.iloc[-1] / close_df.iloc[-6] - 1) * 100
        if len(close_df) >= 6
        else close_df.iloc[-1] * 0
    )
    ret_20d_scalar = (
        (close_df.iloc[-1] / close_df.iloc[-21] - 1) * 100
        if len(close_df) >= 21
        else close_df.iloc[-1] * 0
    )

    avg_volume = volume_df.iloc[-20:].mean()

    snapshots: list[dict] = []
    for sym in close_df.columns:
        px = close_df[sym].iloc[-1]
        if pd.isna(px) or px <= 0:
            continue
        valid_bars = close_df[sym].dropna()
        if len(valid_bars) < 10:
            continue

        # Earnings/news gap-down + failed-bounce detection (post_earnings_gapdown_failed_bounce).
        # Align O/L/C/V on the symbol's own valid dates so a recent gap and its continuation
        # bar are read from clean, contiguous OHLCV.
        ohlcv = pd.DataFrame(
            {
                "Open": open_df[sym] if sym in open_df.columns else float("nan"),
                "Low": low_df[sym] if sym in low_df.columns else float("nan"),
                "Close": close_df[sym],
                "Volume": volume_df[sym] if sym in volume_df.columns else 0.0,
            }
        ).dropna(subset=["Open", "Low", "Close"])
        gap_info = detect_failed_gapdown(
            ohlcv["Open"].tolist(),
            ohlcv["Low"].tolist(),
            ohlcv["Close"].tolist(),
            ohlcv["Volume"].fillna(0.0).tolist(),
        )

        snapshots.append(
            {
                "symbol": sym,
                "current_price": round(float(px), 4),
                "ret_5d_pct": round(float(ret_5d.get(sym, 0) or 0), 4),
                "ret_20d_pct": round(float(ret_20d_scalar.get(sym, 0) or 0), 4),
                "avg_volume": float(avg_volume.get(sym, 0) or 0),
                "rs_rank_pct": (
                    round(float(rs_rank_today[sym]), 2)
                    if sym in rs_rank_today.index and not pd.isna(rs_rank_today[sym])
                    else None
                ),
                "rs_rank_pct_10d_ago": (
                    round(float(rs_rank_lag10[sym]), 2)
                    if sym in rs_rank_lag10.index and not pd.isna(rs_rank_lag10[sym])
                    else None
                ),
                "earnings_gap_pct": gap_info["earnings_gap_pct"],
                "gap_failed_bounce": gap_info["gap_failed_bounce"],
                "vol_ratio": round(float(gap_info["vol_ratio"]), 4),
            }
        )

    logger.info(
        f"Short universe scan: {len(snapshots)} valid symbols from {len(symbols)} requested"
    )
    return snapshots

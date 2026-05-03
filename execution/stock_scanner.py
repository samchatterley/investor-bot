import logging

import yfinance as yf

from config import MIN_VOLUME

logger = logging.getLogger(__name__)

# Broader universe used only for the daily top-movers scan.
# Intentionally sector-diverse and mid-cap-heavy so the momentum×volume scorer
# can surface "sleeper" moves beyond mega-caps.
EXTENDED_UNIVERSE = list(
    dict.fromkeys(
        [
            # Mega-cap tech (anchor names)
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "AMD",
            "NFLX",
            "CRM",
            "ADBE",
            "UBER",
            # Semiconductors — mid/large
            "INTC",
            "QCOM",
            "AVGO",
            "TXN",
            "MU",
            "SMCI",
            "MRVL",
            "ON",
            "MPWR",
            "LRCX",
            "KLAC",
            "AMAT",
            "SWKS",
            # Software / SaaS growth
            "PLTR",
            "SNOW",
            "NET",
            "DDOG",
            "CRWD",
            "BILL",
            "APP",
            "GTLB",
            "PCOR",
            "MNDY",
            "HUBS",
            "TEAM",
            "ZS",
            "PANW",
            # Fintech & payments
            "V",
            "MA",
            "PYPL",
            "SQ",
            "SHOP",
            "COIN",
            "AFRM",
            "SOFI",
            "HOOD",
            # Financials
            "JPM",
            "BAC",
            "GS",
            "MS",
            "C",
            "SCHW",
            # Energy — oil & clean
            "XOM",
            "CVX",
            "OXY",
            "ENPH",
            "SEDG",
            "FSLR",
            "RUN",
            "PLUG",
            "BE",
            # Healthcare & biotech
            "PFE",
            "MRNA",
            "ABBV",
            "LLY",
            "JNJ",
            "BMY",
            "EXAS",
            "NBIX",
            "BMRN",
            "INCY",
            "SRPT",
            "ISRG",
            "DXCM",
            "PODD",
            # Defense — large & mid
            "LMT",
            "RTX",
            "NOC",
            "BA",
            "HII",
            "KTOS",
            "RKLB",
            "LDOS",
            # Industrials & infrastructure
            "CAT",
            "GE",
            "HON",
            "EMR",
            "GNRC",
            "AXON",
            "SAIA",
            "XYL",
            # Consumer discretionary
            "NKE",
            "MCD",
            "SBUX",
            "TGT",
            "COST",
            "WMT",
            "HD",
            "DKNG",
            "LYFT",
            "DASH",
            "CHWY",
            "W",
            # Entertainment & media
            "DIS",
            "SNAP",
            "PINS",
            "RBLX",
            # Commodities & materials
            "FCX",
            "AA",
            "MP",
            "CLF",
            # Crypto-adjacent (high volatility movers)
            "MSTR",
            "MARA",
            "RIOT",
            # Broad market ETFs (baseline comparison)
            "SPY",
            "QQQ",
            "IWM",
        ]
    )
)


def get_market_regime(threshold_pct: float = -1.5, vix: float | None = None) -> dict:
    """
    4-state regime: BULL_TRENDING, CHOPPY, HIGH_VOL, BEAR_DAY.
    Each state guides which signal types Claude should favour.
    """
    try:
        hist = yf.Ticker("SPY").history(period="20d")
        if len(hist) < 6:
            return {
                "is_bearish": False,
                "spy_change_pct": 0.0,
                "spy_5d_pct": 0.0,
                "regime": "UNKNOWN",
            }

        spy_1d = float((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100)
        spy_5d = float((hist["Close"].iloc[-1] / hist["Close"].iloc[-6] - 1) * 100)
        is_bearish = bool(spy_1d <= threshold_pct)

        if is_bearish:
            regime = "BEAR_DAY"
        elif vix is not None and vix > 25 and spy_5d < -3:
            regime = "HIGH_VOL"
        elif spy_5d > 2 and spy_1d > 0:
            regime = "BULL_TRENDING"
        else:
            regime = "CHOPPY"

        logger.info(f"SPY 1d: {spy_1d:+.2f}%  5d: {spy_5d:+.2f}%  Regime: {regime}")
        return {
            "is_bearish": is_bearish,
            "spy_change_pct": round(spy_1d, 2),
            "spy_5d_pct": round(spy_5d, 2),
            "regime": regime,
        }
    except Exception as e:
        logger.error(f"Market regime check failed: {e}")
        return {"is_bearish": False, "spy_change_pct": 0.0, "spy_5d_pct": 0.0, "regime": "UNKNOWN"}


def prefilter_candidates(snapshots: list[dict]) -> list[dict]:
    """
    Rule-based screen applied before Claude analysis.
    A stock must match at least one credible technical pattern.
    Stocks trading against their weekly trend are blocked unless deeply oversold.
    This reduces Claude's input size and focuses it on genuine setups.
    """
    qualified = []
    for s in snapshots:
        if s.get("avg_volume", 0) < MIN_VOLUME:
            continue

        rsi = s.get("rsi_14", 50)
        bb = s.get("bb_pct", 0.5)
        vol = s.get("vol_ratio", 1.0)
        ema_up = s.get("ema9_above_ema21", False)
        macd_diff = s.get("macd_diff", 0)
        macd_cross = s.get("macd_crossed_up", False)
        weekly_up = s.get("weekly_trend_up", True)
        ret_5d = s.get("ret_5d_pct", 0)

        mean_reversion = rsi < 38 and bb < 0.30 and vol > 1.0
        momentum = ema_up and macd_diff > 0 and ret_5d > 0 and vol > 1.2
        fresh_breakout = macd_cross and vol > 1.2

        # ── New signals ───────────────────────────────────────────────────────
        # Volatility squeeze: bands contracting → expansion imminent
        bb_squeeze_breakout = s.get("bb_squeeze", False) and (ema_up or macd_diff > 0) and vol > 1.2
        # Near 52-week high with volume: growth / breakout momentum
        breakout_52w = s.get("price_vs_52w_high_pct", -999) >= -3.0 and vol > 1.2 and weekly_up
        # Consistent SPY outperformance: market leader in sustained uptrend
        rs_leader = (
            s.get("rel_strength_5d", 0) > 2.0 and s.get("rel_strength_10d", 0) > 3.0 and ema_up
        )
        # Inside day followed by directional confirmation: coiled spring
        inside_day_breakout = (
            s.get("is_inside_day", False) and (ema_up or macd_diff > 0) and vol > 1.1
        )
        # Pullback to EMA21 within an established uptrend
        pct_ema21 = s.get("price_vs_ema21_pct", 0)
        trend_pullback = ema_up and -3.0 <= pct_ema21 <= -0.5 and 40 <= rsi <= 58 and vol > 1.0

        if not (
            mean_reversion
            or momentum
            or fresh_breakout
            or bb_squeeze_breakout
            or breakout_52w
            or rs_leader
            or inside_day_breakout
            or trend_pullback
        ):
            continue

        # Block buys against the weekly trend unless the stock is deeply oversold
        if not weekly_up and not (rsi < 30 and bb < 0.15):
            continue

        qualified.append(s)

    return qualified


def get_top_movers(n: int = 10) -> list[str]:
    """Return top n symbols from the extended universe by momentum × volume score today."""
    try:
        data = yf.download(
            EXTENDED_UNIVERSE,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty or len(data) < 2:
            return []

        close = data["Close"]
        volume = data["Volume"]

        ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        vol_avg = volume.iloc[:-1].mean()
        vol_ratio = volume.iloc[-1] / vol_avg.replace(0, 1)

        # Score rewards large moves with above-average volume
        score = ret_1d.abs() * vol_ratio.clip(lower=0.5)
        top = score.dropna().nlargest(n).index.tolist()
        logger.info(f"Top movers: {top}")
        return top
    except Exception as e:
        logger.error(f"Top movers scan failed: {e}")
        return []

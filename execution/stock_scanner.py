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
        macd_cross_signal = macd_cross and vol > 1.2

        # ── New signals ───────────────────────────────────────────────────────
        # Volatility squeeze: bands contracting → expansion imminent
        bb_squeeze_signal = s.get("bb_squeeze", False) and (ema_up or macd_diff > 0) and vol > 1.2
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

        # ── Intraday signals (only present when Alpaca intraday fetch succeeded) ──
        # VWAP reclaim: price moved above VWAP with positive intraday momentum —
        # institutional support level confirmed, high-probability continuation.
        intraday_chg = s.get("intraday_change_pct")
        above_vwap = s.get("price_above_vwap")
        pct_vwap = s.get("pct_vs_vwap", 0)
        vwap_reclaim = (
            above_vwap is True
            and intraday_chg is not None
            and intraday_chg > 1.0
            and pct_vwap <= 3.0  # not so extended above VWAP that it's a chase
        )

        # Opening range breakout: price broke above the first-30-min range with
        # above-average volume — classic intraday momentum signal.
        orb_breakout = s.get("orb_breakout_up", False) is True

        # Intraday momentum: stock is up meaningfully from open, above VWAP,
        # and daily technicals confirm the trend — catch moves that develop during
        # the session rather than only at the open.
        id_rsi = s.get("intraday_rsi")
        intraday_momentum = (
            intraday_chg is not None
            and intraday_chg > 2.0
            and above_vwap is True
            and (id_rsi is None or id_rsi < 75)  # not overbought on 5-min bars
            and (ema_up or ret_5d > 3.0)  # daily trend supports the move
        )

        matched = []
        if mean_reversion:
            matched.append("mean_reversion")
        if momentum:
            matched.append("momentum")
        if macd_cross_signal:
            matched.append("macd_crossover")
        if bb_squeeze_signal:
            matched.append("bb_squeeze")
        if breakout_52w:
            matched.append("breakout_52w")
        if rs_leader:
            matched.append("rs_leader")
        if inside_day_breakout:
            matched.append("inside_day_breakout")
        if trend_pullback:
            matched.append("trend_pullback")
        if vwap_reclaim:
            matched.append("vwap_reclaim")
        if orb_breakout:
            matched.append("orb_breakout")
        if intraday_momentum:
            matched.append("intraday_momentum")

        if not matched:
            continue

        # Block buys against the weekly trend unless the stock is deeply oversold
        if not weekly_up and not (rsi < 30 and bb < 0.15):
            continue

        qualified.append({**s, "matched_signals": matched})

    return qualified


def score_candidate(snapshot: dict) -> float:
    """Deterministic score for baseline comparison against Claude's selections.

    Higher score = stronger setup by technical rules alone.
    Used to compare which candidates Claude selected vs. the rule-ranked order.
    """
    rsi = snapshot.get("rsi_14", 50)
    bb = snapshot.get("bb_pct", 0.5)
    vol = snapshot.get("vol_ratio", 1.0)
    rel = snapshot.get("rel_strength_5d", 0.0)
    n_signals = len(snapshot.get("matched_signals", []))

    rsi_dist = abs(rsi - 50) / 50
    bb_dist = abs(bb - 0.5) * 2
    vol_score = min((vol - 1.0) / 1.0, 1.0) if vol > 1.0 else 0.0
    rel_score = min(max(rel, 0.0), 5.0) / 5.0
    sig_score = n_signals / 8

    return round(
        rsi_dist * 0.25 + bb_dist * 0.20 + vol_score * 0.25 + rel_score * 0.20 + sig_score * 0.10,
        4,
    )


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

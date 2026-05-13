import logging

import yfinance as yf

from config import MIN_VOLUME
from signals.evaluator import REGIME_BLOCKED, evaluate_signals

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


def _passes_quality_screen(snapshot: dict) -> bool:
    """Fundamental quality gate. Permissive when fields are absent (ETFs, backtest mode).

    Rejects stocks with: negative ROE, negative profit margins, or extreme leverage (D/E > 300).
    D/E threshold of 300 is intentionally loose to accommodate financials and REITs.
    """
    roe = snapshot.get("roe")
    profit_margin = snapshot.get("profit_margin")
    debt_to_equity = snapshot.get("debt_to_equity")

    if roe is None and profit_margin is None:
        return True

    if roe is not None and roe < 0:
        return False

    if profit_margin is not None and profit_margin < 0:
        return False

    return not (debt_to_equity is not None and debt_to_equity > 300)


def prefilter_candidates(snapshots: list[dict], regime: str | None = None) -> list[dict]:
    """
    Rule-based screen applied before Claude analysis.
    Signal logic delegated to signals.evaluator — single canonical implementation
    shared with the backtest engine to prevent live/backtest divergence.
    Stocks trading against their weekly trend are blocked unless deeply oversold.
    """
    _blocked: frozenset[str] = (
        REGIME_BLOCKED.get(regime or "", frozenset()) if regime else frozenset()
    )
    qualified = []
    for s in snapshots:
        if s.get("avg_volume", 0) < MIN_VOLUME:
            continue

        if not _passes_quality_screen(s):
            continue

        matched = evaluate_signals(s, blocked=_blocked)

        if not matched:
            continue

        # Weekly trend filter: block buys counter-trend unless deeply oversold or
        # fundamental conviction present (insider cluster / post-earnings drift).
        rsi = s.get("rsi_14", 50)
        bb = s.get("bb_pct", 0.5)
        weekly_up = s.get("weekly_trend_up", True)
        if (
            not weekly_up
            and not (rsi < 30 and bb < 0.15)
            and "insider_buying" not in matched
            and "pead" not in matched
        ):
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

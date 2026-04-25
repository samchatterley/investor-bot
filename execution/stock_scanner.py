from __future__ import annotations
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# Broader universe used only for the daily top-movers scan
EXTENDED_UNIVERSE = list(dict.fromkeys([
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX",
    "CRM", "ADBE", "UBER", "JPM", "BAC", "GS", "XOM", "CVX",
    "COST", "WMT", "HD", "INTC", "QCOM", "AVGO", "TXN",
    "V", "MA", "PYPL", "SHOP", "COIN",
    "SNAP", "PINS", "RBLX", "DIS",
    "PFE", "MRNA", "ABBV", "LLY",
    "BA", "CAT", "GE", "LMT",
    "NKE", "MCD", "SBUX", "TGT",
    "PLTR", "SNOW", "NET", "DDOG", "CRWD",
    "SPY", "QQQ", "IWM",
]))


def get_market_regime(threshold_pct: float = -1.5, vix: float | None = None) -> dict:
    """
    4-state regime: BULL_TRENDING, CHOPPY, HIGH_VOL, BEAR_DAY.
    Each state guides which signal types Claude should favour.
    """
    try:
        hist = yf.Ticker("SPY").history(period="20d")
        if len(hist) < 6:
            return {"is_bearish": False, "spy_change_pct": 0.0, "spy_5d_pct": 0.0, "regime": "UNKNOWN"}

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

        if not (mean_reversion or momentum or fresh_breakout):
            continue

        # Block buys against the weekly trend unless the stock is deeply oversold
        if not weekly_up and not (rsi < 30 and bb < 0.15):
            continue

        qualified.append(s)

    return qualified


def get_top_movers(n: int = 10) -> list[str]:
    """Return top n symbols from the extended universe by momentum × volume score today."""
    try:
        import yfinance as yf
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

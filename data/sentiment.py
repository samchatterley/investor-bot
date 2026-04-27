from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

logger = logging.getLogger(__name__)


def _fetch_analyst(symbol: str) -> tuple[str, dict]:
    """
    Fetch analyst sentiment from yfinance.
    recommendationMean: 1.0 = strong buy, 3.0 = hold, 5.0 = strong sell
    """
    try:
        info = yf.Ticker(symbol).info
        mean = info.get("recommendationMean")
        key = info.get("recommendationKey", "")
        analysts = info.get("numberOfAnalystOpinions")
        target = info.get("targetMeanPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")

        if mean is None or not analysts:
            return symbol, {}

        # Convert 1–5 scale to bullish/bearish pct for compatibility with prompt
        # 1.0–2.0 = bullish, 2.0–3.0 = neutral-bullish, 3.0+ = neutral/bearish
        bullish_pct = round(max(0, min(100, (5 - mean) / 4 * 100)))
        bearish_pct = 100 - bullish_pct

        result = {
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "analyst_count": analysts,
            "recommendation": key,
        }
        if target and current:
            result["upside_pct"] = round((target / current - 1) * 100, 1)

        return symbol, result
    except Exception as e:
        logger.debug(f"Analyst sentiment fetch failed for {symbol}: {e}")
        return symbol, {}


def get_sentiment(symbols: list[str]) -> dict[str, dict]:
    """
    Return analyst sentiment for each symbol via yfinance.
    Format: {symbol: {bullish_pct, bearish_pct, analyst_count, recommendation, upside_pct}}
    Symbols with no data are omitted.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_analyst, sym): sym for sym in symbols}
        for future in as_completed(futures, timeout=30):
            try:
                symbol, data = future.result()
                if data:
                    results[symbol] = data
            except Exception:
                pass

    logger.info(f"Sentiment data fetched for {len(results)}/{len(symbols)} symbols")
    return results

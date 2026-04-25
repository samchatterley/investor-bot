import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


def _fetch_stocktwits(symbol: str) -> tuple[str, dict]:
    """Fetch sentiment for a symbol from Stocktwits public API."""
    try:
        url = STOCKTWITS_URL.format(symbol=symbol)
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return symbol, {}

        data = resp.json()
        messages = data.get("messages", [])
        if not messages:
            return symbol, {}

        bullish = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish")
        bearish = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish")
        total = bullish + bearish

        if total == 0:
            return symbol, {}

        return symbol, {
            "bullish_pct": round(bullish / total * 100),
            "bearish_pct": round(bearish / total * 100),
            "message_count": len(messages),
        }
    except Exception as e:
        logger.debug(f"Stocktwits fetch failed for {symbol}: {e}")
        return symbol, {}


def get_sentiment(symbols: list[str]) -> dict[str, dict]:
    """
    Return {symbol: {bullish_pct, bearish_pct, message_count}} for each symbol.
    Symbols with no data are omitted. Fetched in parallel with a timeout.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_stocktwits, sym): sym for sym in symbols}
        for future in as_completed(futures, timeout=15):
            try:
                symbol, data = future.result()
                if data:
                    results[symbol] = data
            except Exception:
                pass

    logger.info(f"Sentiment data fetched for {len(results)}/{len(symbols)} symbols")
    return results

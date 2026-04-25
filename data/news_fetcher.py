import yfinance as yf
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _fetch_single(symbol: str, max_headlines: int) -> tuple[str, list[str]]:
    try:
        ticker = yf.Ticker(symbol)
        raw = getattr(ticker, "news", None) or []
        headlines = []
        for item in raw[:max_headlines]:
            title = item.get("title") or item.get("headline", "")
            if title:
                headlines.append(title)
        return symbol, headlines
    except Exception as e:
        logger.debug(f"News fetch failed for {symbol}: {e}")
        return symbol, []


def fetch_news(symbols: list[str], max_headlines: int = 3) -> dict[str, list[str]]:
    """Fetch recent headlines for each symbol. Returns {symbol: [headline, ...]}"""
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_single, sym, max_headlines): sym for sym in symbols}
        for future in as_completed(futures):
            symbol, headlines = future.result()
            if headlines:
                results[symbol] = headlines
    logger.info(f"Fetched news for {len(results)}/{len(symbols)} symbols")
    return results

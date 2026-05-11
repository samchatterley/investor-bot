import logging

from data.fundamentals import get_analyst_consensus

logger = logging.getLogger(__name__)


def get_sentiment(symbols: list[str]) -> dict[str, dict]:
    """Return analyst consensus per symbol via FMP (24-hour cache).

    Format: {symbol: {bullish_pct, bearish_pct, analyst_count, target_price}}.
    Returns {} when FMP_API_KEY is unset or no data is available.
    """
    return get_analyst_consensus(symbols)

"""Alpha Vantage NEWS_SENTIMENT enrichment.

Augments per-symbol snapshots with a structured numeric sentiment score sourced
from Alpha Vantage's news feed rather than the yfinance headline titles already
fetched by news_fetcher.py.  The two sources are complementary: yfinance feeds
raw headlines to Claude; AV feeds a numeric score to the pre-filter and Claude.

Free tier: 500 calls/day, 5 calls/minute → batch by _BATCH_SIZE symbols per call
and sleep _REQ_DELAY seconds between batches.

Set ALPHA_VANTAGE_API_KEY in .env to enable.  When absent the module returns an
empty dict silently so the rest of the pipeline is unaffected.
"""

import logging
import time
from datetime import UTC, datetime, timedelta

import requests

from config import ALPHA_VANTAGE_API_KEY

logger = logging.getLogger(__name__)

_AV_URL = "https://www.alphavantage.co/query"
_BATCH_SIZE = 10  # symbols per request
_REQ_DELAY = 13.0  # 5 req/min free → 12 s between requests; 13 for safety
_MIN_RELEVANCE = 0.3  # ignore tangential mentions below this score


def get_av_sentiment(
    symbols: list[str],
    lookback_hours: int = 24,
) -> dict[str, dict]:
    """Fetch Alpha Vantage news sentiment for each symbol.

    Result schema per symbol::

        {
            "av_sentiment_score": float,   # avg ticker-level score; -1 to +1
            "av_article_count":   int,     # articles mentioning symbol in window
            "av_sentiment_label": str,     # "Bullish" | "Neutral" | "Bearish"
            "av_top_headline":    str,
        }

    Symbols with no relevant articles in the window are omitted.
    Returns empty dict when ALPHA_VANTAGE_API_KEY is not configured.
    """
    if not ALPHA_VANTAGE_API_KEY:
        logger.debug("ALPHA_VANTAGE_API_KEY not set — AV sentiment disabled")
        return {}

    cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
    result: dict[str, dict] = {}

    for batch_start in range(0, len(symbols), _BATCH_SIZE):
        batch = symbols[batch_start : batch_start + _BATCH_SIZE]
        tickers_param = ",".join(batch)

        try:
            resp = requests.get(
                _AV_URL,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": tickers_param,
                    "apikey": ALPHA_VANTAGE_API_KEY,
                    "limit": 50,
                    "sort": "LATEST",
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"AV sentiment fetch failed for {tickers_param}: {exc}")
            if batch_start + _BATCH_SIZE < len(symbols):
                time.sleep(_REQ_DELAY)
            continue

        if "feed" not in data:
            info = data.get("Information", data.get("Note", ""))
            if info:
                logger.warning(f"AV API message: {info}")
            if batch_start + _BATCH_SIZE < len(symbols):
                time.sleep(_REQ_DELAY)
            continue

        sym_scores: dict[str, list[float]] = {s.upper(): [] for s in batch}
        sym_headlines: dict[str, str] = {}

        for article in data["feed"]:
            try:
                pub_str = article.get("time_published", "")
                pub_time = datetime.strptime(pub_str, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
            except (ValueError, TypeError):
                continue
            if pub_time < cutoff:
                continue

            for ts in article.get("ticker_sentiment", []):
                sym = (ts.get("ticker") or "").upper()
                if sym not in sym_scores:
                    continue
                try:
                    score = float(ts["ticker_sentiment_score"])
                    relevance = float(ts["relevance_score"])
                except (ValueError, KeyError, TypeError):
                    continue
                if relevance < _MIN_RELEVANCE:
                    continue
                sym_scores[sym].append(score)
                if sym not in sym_headlines:
                    sym_headlines[sym] = article.get("title", "")

        for sym in sym_scores:
            scores = sym_scores[sym]
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            label = "Bullish" if avg > 0.15 else "Bearish" if avg < -0.15 else "Neutral"
            result[sym] = {
                "av_sentiment_score": round(avg, 3),
                "av_article_count": len(scores),
                "av_sentiment_label": label,
                "av_top_headline": sym_headlines.get(sym, ""),
            }

        if batch_start + _BATCH_SIZE < len(symbols):
            time.sleep(_REQ_DELAY)

    logger.info(f"AV sentiment: {len(result)}/{len(symbols)} symbols enriched")
    return result

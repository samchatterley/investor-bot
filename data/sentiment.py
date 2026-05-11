import logging

logger = logging.getLogger(__name__)

# Yahoo Finance's quoteSummary endpoint (used for analyst ratings) now requires
# authentication for all recommendationTrend / financialData modules. Attempting
# the call floods the log with 401 errors and returns nothing useful (0/N symbols
# succeed). This module returns empty until a replacement data source is wired up.


def get_sentiment(symbols: list[str]) -> dict[str, dict]:
    """Return analyst sentiment per symbol. Currently unavailable (Yahoo Finance API restriction)."""
    if symbols:
        logger.debug(
            f"Analyst sentiment skipped for {len(symbols)} symbols (Yahoo Finance API restricted)"
        )
    return {}

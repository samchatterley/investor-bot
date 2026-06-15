"""Runner for the data-feed health gate.

Probes every live feed the bot consumes and prints a health report (OK / EMPTY / DEGRADED / STALE /
ERROR), exiting non-zero if any feed needs attention. Use before collecting experiment data, and as
periodic monitoring (feeds rot over time). This is operational glue around live services, so it is
excluded from unit-test coverage; the classification logic lives in experiment/feed_health.py.

Run:  python scripts/feed_health_check.py
"""

from __future__ import annotations  # pragma: no cover

import math  # pragma: no cover
import os  # pragma: no cover
import sys  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from experiment.feed_health import (  # noqa: E402  # pragma: no cover
    DEGRADED,
    OK,
    format_report,
    run_health_checks,
    summarise,
)

_SAMPLE = "AAPL"  # liquid, data-rich probe symbol  # pragma: no cover
_FILER = "NVDA"  # recent 8-K filer  # pragma: no cover


def _vix(v):  # pragma: no cover
    return (OK, f"VIX={v:.1f}") if 5 <= v <= 100 else (DEGRADED, f"implausible VIX={v}")


def _price(v):  # pragma: no cover
    return (OK, f"${v:.2f}") if v and v > 0 else (DEGRADED, f"non-positive {v}")


def _aaii(s):  # pragma: no cover
    b = getattr(s, "bullish_pct", float("nan"))
    if b is None or math.isnan(b):
        return (DEGRADED, "percentages are NaN (parse failure)")
    if not 0.0 < b < 1.0:
        return (DEGRADED, f"implausible bullish_pct={b}")
    return (OK, f"bull={b:.0%} bear={getattr(s, 'bearish_pct', float('nan')):.0%}")


def _finbert(avail):  # pragma: no cover
    return (OK, "model loaded") if avail else (DEGRADED, "transformers/torch not installed")


def _probes():  # pragma: no cover
    from data.analyst_revisions import get_analyst_revisions
    from data.breadth import get_breadth_snapshot
    from data.edgar_client import get_guidance_sentiment
    from data.finbert import is_available
    from data.fred_client import get_10y_yield, get_yield_curve_inverted_days
    from data.fundamental_cache import get_altman_z, get_piotroski_f
    from data.fundamentals import get_fundamentals
    from data.google_trends import get_google_trends_signals
    from data.insider_feed import get_insider_activity
    from data.market_data import get_index_price, get_spy_5d_return, get_vix
    from data.news_fetcher import fetch_news
    from data.options_data import get_options_batch
    from data.sector_data import get_sector_performance
    from data.sentiment_client import get_aaii_sentiment

    return [
        ("market_data SPY price", lambda: get_index_price("SPY"), _price),
        ("market_data VIX", get_vix, _vix),
        ("market_data SPY 5d ret", get_spy_5d_return, None),
        ("sentiment AAII", get_aaii_sentiment, _aaii),
        ("finbert news sentiment", is_available, _finbert),
        ("fred 10y yield", get_10y_yield, None),
        ("fred yield-curve inv days", get_yield_curve_inverted_days, None),
        ("fundamentals (Finnhub)", lambda: get_fundamentals([_SAMPLE]), None),
        ("fundamental_cache altman_z", lambda: get_altman_z(_SAMPLE), None),
        ("fundamental_cache piotroski", lambda: get_piotroski_f(_SAMPLE), None),
        ("options IV surface", lambda: get_options_batch([_SAMPLE]), None),
        ("insider feed (EDGAR)", lambda: get_insider_activity([_SAMPLE]), None),
        ("analyst revisions", lambda: get_analyst_revisions([_SAMPLE]), None),
        ("news headlines", lambda: fetch_news([_SAMPLE]), None),
        ("google trends", lambda: get_google_trends_signals([_SAMPLE]), None),
        ("market breadth", get_breadth_snapshot, None),
        ("sector performance", get_sector_performance, None),
        ("edgar 8-K guidance", lambda: get_guidance_sentiment(_FILER), None),
    ]


def main():  # pragma: no cover
    results = run_health_checks(_probes())
    print(format_report(results))
    _, _, all_green = summarise(results)
    sys.exit(0 if all_green else 1)


if __name__ == "__main__":  # pragma: no cover
    main()

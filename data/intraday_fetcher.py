"""Fetch and cache Alpaca 1-min OHLCV bars for intraday backtesting.

Results are pickled per (symbol, start_date, end_date) to avoid re-downloading
across repeated backtest runs.  Pass cache_dir=None to use the default location
(data/.intraday_cache/) or "" to disable caching entirely.
"""

import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", ".intraday_cache"
)


def _cache_path(cache_dir: str, symbol: str, start: str, end: str) -> str:
    return os.path.join(cache_dir, f"{symbol}_{start}_{end}.pkl")


def fetch_intraday_bars(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | None = None,
) -> dict[str, dict[str, list]]:
    """Fetch Alpaca 1-min bars for symbols over [start_date, end_date].

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to fetch.
    start_date, end_date : str
        ISO-format date strings (inclusive).
    cache_dir : str | None
        Directory for pickle cache.  None → default location.
        Pass "" to disable caching (e.g. in tests).

    Returns
    -------
    dict[str, dict[str, list]]
        {symbol: {date_str: [(datetime_et, bar), ...]}}
        Bars are sorted ascending by time within each date.
        Bar objects expose .open .high .low .close .volume attributes.
    """
    use_cache = cache_dir != ""
    if use_cache and cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    if use_cache:
        assert cache_dir is not None
        os.makedirs(cache_dir, exist_ok=True)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        import config as _cfg
    except ImportError as exc:
        logger.error(f"Alpaca SDK unavailable: {exc}")
        return {}

    if not (_cfg.ALPACA_API_KEY and _cfg.ALPACA_SECRET_KEY):
        logger.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not set — cannot fetch intraday bars")
        return {}

    client = StockHistoricalDataClient(
        api_key=_cfg.ALPACA_API_KEY,
        secret_key=_cfg.ALPACA_SECRET_KEY,
    )
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=_ET)
    end_dt = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=_ET)

    result: dict[str, dict[str, list]] = {}

    for idx, sym in enumerate(symbols):
        if use_cache:
            assert cache_dir is not None
            cache_file = _cache_path(cache_dir, sym, start_date, end_date)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        result[sym] = pickle.load(f)
                    logger.debug(f"Intraday cache hit: {sym}")
                    continue
                except Exception:
                    pass

        logger.info(f"Fetching intraday bars: {sym} ({idx + 1}/{len(symbols)})")
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            req = StockBarsRequest(
                symbol_or_symbols=sym,
                start=start_dt,
                end=end_dt,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                feed="iex",  # type: ignore[arg-type]
            )
            bars_resp = client.get_stock_bars(req)
            bars_data = bars_resp.data.get(sym, [])  # type: ignore[union-attr]
            if not bars_data:
                continue

            bars_by_date: dict[str, list] = defaultdict(list)
            for bar in bars_data:
                bar_et = bar.timestamp.astimezone(_ET)
                bars_by_date[bar_et.strftime("%Y-%m-%d")].append((bar_et, bar))

            for ds in bars_by_date:
                bars_by_date[ds].sort(key=lambda x: x[0])

            sym_data = dict(bars_by_date)
            result[sym] = sym_data

            if use_cache:
                assert cache_dir is not None
                with open(_cache_path(cache_dir, sym, start_date, end_date), "wb") as f:
                    pickle.dump(sym_data, f)

        except Exception as exc:
            logger.warning(f"Intraday fetch failed for {sym}: {exc}")

    logger.info(f"Intraday bars ready for {len(result)}/{len(symbols)} symbols")
    return result

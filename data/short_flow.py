"""FINRA Reg SHO daily short-sale volume — free point-in-time short-FLOW history.

FINRA publishes a consolidated daily file (no key, ~2009→present) with per-name short volume:

    https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

The derived series short_volume_ratio = ShortVolume/TotalVolume is a *positioning/flow* feed the
bot has never had with history. It unlocks:
  * informed short-flow signals (Boehmer-Jones-Zhang: heavy shorting predicts negative returns in
    LARGE caps at daily/weekly horizons — fits the liquid universe + t+1 entry), and
  * a historical crowding/borrow proxy (gates lottery_pop_short; was the blocker for N2/knife-short).

Each day's file is cached to disk after first fetch, so a 12-year history build is a one-time cost
and re-runs are free. Parsing is pure (unit-testable); the single network call is isolated.
"""

from __future__ import annotations

import logging
import os
from datetime import date

import requests

from config import LOG_DIR

logger = logging.getLogger(__name__)

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
_CACHE_DIR = os.path.join(LOG_DIR, "caching", "finra_short_flow")
_HEADERS = {"User-Agent": "investorbot-research"}


def _parse_day(text: str, symbols: set[str] | None = None) -> dict[str, float]:
    """Pure: parse one CNMSshvol file → {symbol: short_volume_ratio}. Bad rows skipped."""
    out: dict[str, float] = {}
    for line in text.strip().split("\n")[1:]:  # skip header
        parts = line.split("|")
        if len(parts) < 5:
            continue
        sym = parts[1].strip().upper()
        if symbols is not None and sym not in symbols:
            continue
        try:
            short_vol = float(parts[2])
            total_vol = float(parts[4])
        except ValueError:
            continue
        if total_vol > 0:
            out[sym] = short_vol / total_vol
    return out


def _fetch_day_text(d: date) -> str | None:
    """Fetch one day's file (network call — mocked in tests). None on any failure/holiday."""
    url = _URL.format(ymd=d.strftime("%Y%m%d"))
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        return str(resp.text)
    except Exception as exc:
        logger.debug("short_flow: fetch failed %s: %s", d, exc)
        return None


def get_day(d: date, symbols: set[str] | None = None, use_cache: bool = True) -> dict[str, float]:
    """{symbol: short_volume_ratio} for one session. Disk-cached per day; {} for holidays/failures."""
    cache_file = os.path.join(_CACHE_DIR, f"{d.isoformat()}.txt")
    text: str | None = None
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                text = f.read()
        except OSError:
            text = None
    if text is None:
        text = _fetch_day_text(d)
        if text is None:
            return {}
        if use_cache:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            try:
                with open(cache_file, "w") as f:
                    f.write(text)
            except OSError:  # pragma: no cover — disk-full edge
                pass
    return _parse_day(text, symbols)

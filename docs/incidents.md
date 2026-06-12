# InvestorBot — Day One Incidents

The bot went live on paper trading on 27 April 2026. Six distinct failures surfaced in the first two hours, none of which appeared in local testing. Each is documented below with root cause and fix — these failures reveal design assumptions worth being explicit about.

---

### Incident 1 — Python 3.9 crash at scheduler startup

**Symptom:** Bot failed to start. `cron.log` showed a `TypeError` at import time on `emailer.py`:

```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

**Root cause:** The type annotation `dict | None` (PEP 604 union syntax) requires Python 3.10+. Development had used Python 3.11; the production Mac ran 3.9. The crash happened at module *import*, not at runtime — the annotation was in a function signature that never ran, but Python 3.9 evaluates all annotations eagerly at import time.

**Fix (1.2):** Added `from __future__ import annotations` to the affected files. This makes all annotations lazy strings (PEP 563), restoring compatibility back to Python 3.7 with no other changes needed.

**Fix (1.4):** Standardised the entire stack on Python 3.12 — venv, cron, and Docker image all pinned to 3.12. The `from __future__ import annotations` shims were subsequently removed as they were no longer needed.

**Learning:** The venv Python and the system Python used by cron can differ silently. Cron entries must point to the venv binary explicitly, not `/usr/bin/python3`.

---

### Incident 2 — News fetcher returning zero results

**Symptom:** `Fetched news for 0/30 symbols` on every run despite the stocks being active and well-covered.

**Root cause:** yfinance had changed its news API response format between versions. Headline text that previously lived at `item["title"]` had moved to `item["content"]["title"]`. The fetcher looked only in the old location and silently returned empty lists.

**Fix:** Added a fallback chain that checks both locations before giving up:

```python
title = (
    item.get("title")
    or item.get("headline")
    or (item.get("content") or {}).get("title")
    or ""
)
```

**Learning:** External API clients silently change response shapes, especially undocumented ones. Adapters should fail loudly (log a warning on unexpected shape) rather than returning empty results that look like "no data".

---

### Incident 3 — Sentiment fetcher returning zero results

**Symptom:** `Sentiment data fetched for 0/10 symbols`. All requests to Stocktwits were returning 403.

**Root cause:** Stocktwits had deployed Cloudflare protection that blocked the requests. Investigation showed the new `api-gw-prd` endpoint returns 401 (requires HTTP Basic auth) and the developer programme — through which API keys would be obtained — had closed registration.

**Fix:** Complete rewrite of `data/sentiment.py`. Replaced Stocktwits with yfinance analyst consensus data (`recommendationMean`, scale 1–5 where 1 = strong buy). This is arguably more useful than social sentiment for a 1–5 day holding strategy — analyst price targets and conviction counts are directly relevant to the signals the bot trades.

```python
bullish_pct = round(max(0, min(100, (5 - mean) / 4 * 100)))
```

The existing prompt format (`bullish_pct`, `bearish_pct`) was preserved so nothing upstream needed changing.

**Learning:** Free public APIs have no SLA. A dependency on an undocumented endpoint is a single point of failure. Where possible, prefer an API that surfaces the same signal through a more durable path (here: broker/data provider data over social media scraping).

---

### Incident 4 — Trailing stops rejected for fractional share positions

**Symptom:** Two stop attachment errors on every run:

```
Failed to place trailing stop for NVDA: fractional orders must be DAY orders
```

Then after changing `time_in_force` to `DAY`:

```
Failed to place trailing stop for NVDA: fractional orders must be market, limit, stop, or stop_limit orders
```

**Root cause:** Alpaca does not support `TrailingStopOrderRequest` for fractional share positions under any `time_in_force`. The Kelly Criterion sizing produced fractional quantities (e.g. 132.65 shares of NVDA), but the assumption was that Alpaca's trailing stop type worked universally.

**Fix:** `place_trailing_stop()` now detects fractional quantities and falls back to a fixed `StopOrderRequest` at `TRAILING_STOP_PCT` below the current price, passing `current_price` through from the position object already in memory. Whole-share positions continue to use the trailing stop.

```python
is_fractional = abs(safe_qty - round(safe_qty)) > 0.000001
```

**Learning:** Broker API constraints don't map cleanly to order type abstractions. Fractional support and order type support are orthogonal features that need to be tested in combination, not assumed.

---

### Incident 5 — Stop qty rounding causing insufficient-qty rejection

**Symptom:** LMT stop order rejected immediately after incident 4's fix:

```
insufficient qty available for order (requested: 64.075232, available: 64.075231525)
```

**Root cause:** `round(64.075231525, 6)` produces `64.075232` — a value fractionally *above* the available quantity that Alpaca considers settleable. Python's rounding is correct (`...525` rounds the 6th decimal up), but Alpaca's available-qty figure and submitted-qty figure need to agree to sub-cent precision.

**Fix:** Replaced `round(qty, 6)` with floor truncation: `math.floor(qty * 1_000_000) / 1_000_000`. This guarantees the submitted quantity never exceeds what the broker considers available.

**Learning:** When submitting quantities back to a broker that supplied them, truncate rather than round. The broker's figure is the authoritative ceiling; rounding can push above it.

---

### Incident 6 — Midday and close runs never scheduled

**Symptom:** No midday or pre-close run despite the system being described as running multiple cycles per day.

**Root cause:** The crontab had only been configured for the open run during initial setup. The scheduler script (`scripts/run_scheduler.py`) was correct, but the cron entries for midday and close had never been added.

**Fix (1.2):** Added the two missing cron entries.

**Fix (1.5):** Removed cron entirely and consolidated on `scripts/run_scheduler.py` as the single production runner. The scheduler was already the intended entrypoint (it is what Docker runs) and includes the Sunday weekly review, proper exception handling, and diagnostics. Cron was a partial implementation missing the Sunday job and running on the wrong Python binary.

**Learning:** "The system supports three modes" and "the system is configured to run three modes" are different claims. The scheduler script is the single authoritative runner — cron entries are a footgun that can diverge silently.

---

### Summary

| # | Failure | Category | Time to fix |
|---|---------|----------|-------------|
| 1 | Python 3.9 `\|` syntax crash | Environment assumption | ~10 min |
| 2 | News fetcher silent zero | External API drift | ~15 min |
| 3 | Sentiment fetcher blocked | Third-party dependency | ~30 min |
| 4 | Trailing stop rejected for fractional | Broker constraint untested | ~20 min |
| 5 | Stop qty rounding above available | Numeric precision | ~10 min |
| 6 | Midday/close never scheduled | Configuration gap | ~5 min |

All six were diagnosed from logs alone without needing to reproduce locally. The system's structured logging — a timestamped record for every run with explicit counts like `Fetched news for 0/30 symbols` — made it possible to identify all failures within the first run's output rather than inferring them from missing behaviour.

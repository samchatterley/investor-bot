# Fail-open audit — risk/execution gates (2026-06-26)

Read-only audit of `except` blocks in the risk/execution path, asking one question per gate: **when
the data it needs is missing or the fetch errors, does the gate fail *closed* (block the trade) or
*open* (permit it)?** Fail-open on a risk gate means a transient data hiccup silently disables a
protection with real money on the line.

## Findings

| Gate | Behaviour on missing/failed data | Verdict |
|---|---|---|
| **Order-ledger / daily-notional / broker-state** (buy + short paths) | raise `OrderLedgerUnavailable` / `BrokerStateUnavailable` → **suspend trading** | ✅ fail-closed (correct) |
| **Quote gate** (`execution/quote_gate.py`) | raises → caught in `_execute_buy_phase` → **suspend buys** | ✅ fail-closed |
| **Correlation gate** (`risk/correlation.py`) | fetch fails → `{}` → `correlated_with_held` returns `False` → **permit** | ⚠️ fail-open, **documented & intentional** (docstring says so) |
| **Borrow-cost / HTB gate** (`data/borrow_cost.py` via `execution/short_risk.fetch_squeeze_info`) | fetch fails → `{short_pct_float: None, days_to_cover: None}` → `estimate_borrow_rate(None,None)` = `GC_RATE` (cheapest) → `is_hard_to_borrow` `False` → **permit** | ⚠️ fail-open, **emergent (undocumented)** |
| **Squeeze gate** (`execution/short_risk.is_squeeze_risk`) | `None` SI inputs → both SI checks skipped, only `ret_5d` remains → **permit** | ⚠️ fail-open, **emergent** |
| **Earnings guard** (`risk/earnings_calendar.py`) | fetch fails → `None` → guard can't see earnings → **permit holding through earnings** | ⚠️ fail-open |

## The one worth fixing: borrow/squeeze conflation

The borrow and squeeze gates fail open on *missing short-interest data* — which is **mostly correct**:
many names legitimately have no SI data, and failing closed there would block ~every single-name
short (kill the short book). The real defect is that the gate **cannot distinguish "no data" from
"fetch errored"** — `fetch_squeeze_info` returns the same `{None, None}` for both. So a transient
API error silently strips borrow + squeeze protection from a short, and the only remaining guard is
the AI veto.

**Surgical fix — APPLIED (1.122):** `fetch_squeeze_info` now returns a distinct `fetch_error: True`
on an API exception (vs `fetch_error: False` for a successful fetch, including genuine no-data), and
`_execute_shorts` **skips the short when `fetch_error` is set** (fail-closed) while still permitting
on legitimate no-data (fail-open, as before). This closes the transient-error hole without disabling
the short book.

## Note
The correlation fail-open is a deliberate, documented choice; the earnings-guard fail-open is lower
stakes (the AI sees earnings context separately). None of these are *bugs* per se — they're
risk-posture choices, some emergent rather than chosen. The actionable one is the borrow/squeeze
error-vs-no-data conflation.

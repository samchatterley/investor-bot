# InvestorBot — Changelog

Full version history. Most recent first.

---

### 1.146 — July 2026 — fix S&P 500 constituent fetch (Wikipedia 403) + add timeout

The live scanner logged `S&P 500 fetch failed (non-fatal): HTTP Error 403: Forbidden`. Wikipedia now
rejects the default urllib/pandas User-Agent, so `_fetch_sp500_symbols` was silently failing and the
universe's S&P 500 refresh tier fell back to the static `STOCK_UNIVERSE` core every scan (non-fatal —
the bot kept trading the ~906-name core — but the constituent list stopped refreshing).

- New `_fetch_sp500_html()` fetches the page via `urllib` with a **descriptive, policy-compliant
  User-Agent** (verified: default → 403, descriptive → 200) and a **30s timeout**. The timeout also
  closes a latent hang: the old `pd.read_html(url)` had no bound, and this runs in the sequential
  prefetch job where an unbounded fetch would freeze every job behind it.
- `pd.read_html` remains the parse seam (still mockable in tests). Tests updated to mock the new
  fetch helper; added a test asserting the UA + timeout are set. Tests 5,024.

---

### 1.145 — July 2026 — capitulation-bounce unblock for `residual_reversal` (deepen-the-edge)

The reversal enhancement study (`scripts/reversal_enhance.py` → `reversal_vix_gate.py` →
`reversal_regime_reconcile.py`, walk-forward train<2021 / test≥2021) found that the live regime
blocks were suppressing reversal's **single best, most robust bucket**: idiosyncratic losers in
**STRESS_RISK_OFF** (acute capitulation). Partitioned against the live regimes, `residual_reversal`
in STRESS nets **+1.86%/3d @7bps** (train +2.11 / test +1.28, 6/7 +yrs, and it *grows* un-winsorised
so real crash tails don't sink it) — the forced-liquidation overshoot snaps back. Meanwhile the
allowed calm bucket (VIX<20) is dead and the elevated-not-stress bucket is train-negative, so the
current live reversal (+0.09% net, train-negative) was mediocre precisely because the block removed
the good bucket.

- **Unblocked `residual_reversal` in STRESS_RISK_OFF only**, for **liquid names only** — the firing
  gate now requires `spread_proxy_20d ≤ spread_proxy_max` when `regime == "STRESS_RISK_OFF"`. This is
  the survivorship guardrail: the distressed names that actually delist in a crash are the illiquid
  ones (and survivorship is the one bias free data can't correct, so we bound it structurally); the
  liquid ones bounce. HIGH_VOL_DOWNTREND reversal is train-negative and stays blocked; UNKNOWN stays
  blocked (no regime info). The market-excess construction (stock − SPY ≤ −7%) already keeps this
  idiosyncratic, not "buy the whole falling market".
- **Plumbing:** `evaluate_signals()` gains a `regime: str = ""` argument (empty disables the
  capitulation leg — safe for plain callers); the live scanner and backtest engine pass the canonical
  regime through. No new signal name, so SYSTEM_PROMPT / wiring parity is unaffected.
- Size is deliberately **not** cut in stress: the guardrails (STRESS-only, liquid-only, idiosyncratic
  market-excess, ≤5 positions, 3-day hold) bound the risk, and cutting size would directly shrink the
  +1.86% edge we are unblocking.

---

### 1.144 — July 2026 — N1 sector conjunct + crowded-popper short shadow (workshop v2 #1/#2)

Two outputs of the v2 signal workshop's top ideas, both measure-first validated:

- **`residual_reversal` sector conjunct (SHIPPED).** The −7% market-excess drop must now ALSO clear
  −7% vs the name's **own sector** (equal-weight universe peers, `sector_ret_5d_pct`, computed in the
  live cross-sectional block; fail-open when sector unknown / in the backtest engine). Head-to-head
  (identical weeks, 2015–2026): the intersection nets **+0.374%/3d @7bps (t=7.1, 10/12 +yrs)** vs
  +0.301% (8/12) for the live spy-only construction — the removed trades are sector-rout beta that
  continues rather than reverts.
- **Crowded-popper short shadow (`analysis/shadow_popper_shorts`).** The lottery_pop_short study
  found the short edge after ≥+10% pops lives ONLY in FINRA-crowded names (top-quartile short-volume
  ratio: +0.86%/3d gross, t=3.7, survives 30%/yr borrow on the mean) — but year-consistency degrades
  under borrow (4/9 at 15%/yr) and the live squeeze tail is unbounded, so it ships as a read-only
  shadow logger (fail-safe in the run loop, like the catalyst shadow) to accrue forward evidence.
  Notable: combined with the N1×SVR result, short-flow is now confirmed as a **mean-reversion
  amplifier on both tails** (crowded names overshoot and snap back harder in either direction), not a
  directional signal. Research: `scripts/lottery_pop_short_backtest.py`, `scripts/n1_sector_residual.py`.

### 1.143 — July 2026 — options kill/keep: retire unusual_options_activity + put_call_contrarian

First-ever evidence for the v1.98 live-only options signals, via Alpaca's historical option bars
(~2.4y, 907 names, 32,555 name-week ATM snapshots; IV by Black-Scholes inversion of synthetic-OCC
contract closes; volume proxies where OI history doesn't exist):

- **`iv_vs_rv_spread` — KEEP.** Its premise (ATM IV / RV20 < 0.70) is the one options idea with real
  signal: +0.262%/4d gross (t=2.6), positive all 3 years, while the rich-vol control arm is dead flat.
- **`unusual_options_activity` — RETIRED.** Premise **inverted**: call-volume spikes precede
  −0.178%/3d (t=−2.4, 0/3 years). "Informed upside buying" is actually retail chase (MAX-effect
  family) — the signal had been feeding systematic underperformers to the AI shortlist since v1.98.
- **`put_call_contrarian` — RETIRED.** Flat on the volume proxy (+0.037%/3d, t=0.99); no supporting
  evidence after two years live.
- `options_skew_signal` — not testable without OTM put chains; stays live-only.

Standard disable checklist applied (GLOBALLY_DISABLED + pragmas + SYSTEM_PROMPT + tests converted +
docs). 2.4-year window is corroborative rather than 9-year-definitive — flagged in the research
script (`scripts/options_iv_backtest.py`).

### 1.142 — July 2026 — FINRA daily short-flow feed (free point-in-time positioning history)

New `data/short_flow`: FINRA's Reg SHO consolidated daily short-sale volume file
(`cdn.finra.org/equity/regsho/daily/`, no key, per-name daily history to ~2009). The derived
`short_volume_ratio` gives the bot its first **flow/positioning** feed with point-in-time history —
the one mechanism family the v2 signal workshop flagged as both unexhausted *and* suited to the
liquid universe. Unlocks: (a) informed short-flow signals (Boehmer-Jones-Zhang — heavy shorting
predicts negative returns in large caps, where borrow is general-collateral so the short side is not
friction-dead); (b) a historical crowding/borrow proxy — the missing gate for `lottery_pop_short`
(workshop v2 #1) and the blocker that killed N2/knife-short backtests. Per-day disk cache; pure
parser + isolated fetch; 100% covered. Feed survey also verified free: Alpaca historical options
(Feb 2024→, unlocks kill/keep tests of the 4 live-only options signals), SEC fails-to-deliver, and
GDELT news tone. +9 tests (4,997 → 5,006).

### 1.141 — July 2026 — raise MAX_POSITIONS 2 → 5 in small-account mode

Two positions is dangerously concentrated for the book (one name dominates the drawdown). The
reversal-basket sweep (the one validated edge) shows the information ratio **peaks at ~5 positions and
is flat-to-lower beyond**, with N=2 the *worst* (most concentrated) — so raising the small-account
default from 2 to 5 lowers single-name blow-up risk essentially for free, and it matches the
non-small-account default. The **dollar** caps (`MAX_DEPLOYED_USD`, `MAX_SINGLE_ORDER_USD`, daily
notional) are unchanged, so total exposure is identical — 5 positions are just smaller, more
diversified slices (and more decision samples for the experiment). Safety invariants and their tests
updated accordingly.

### 1.140 — July 2026 — dynamic rules-based universe builder

New `data/universe_builder`: a self-maintaining tradeable universe built from **Alpaca's free assets
API** + a screen, replacing the hand-curated hardcoded `STOCK_UNIVERSE` (which goes stale — misses
new listings, keeps delisted names). The screen keeps active, tradable, **fractionable** (Alpaca's
own liquidity/establishment proxy) US common stocks on NYSE/NASDAQ/AMEX, excluding ETFs/funds/trusts
by name and warrants/units/rights/preferreds by symbol shape.

Live it yields **~3,979 liquid names vs the 907 hardcoded** (879 overlap + ~3,100 new small/mid-caps)
— a 4.4× expansion. Screen logic is a pure function (unit-testable); the single Alpaca call is
isolated for mocking; cached per calendar day; falls back to [] on failure so the static list stays a
safety net. Not yet wired into the live bot — pending a measure-first check that the wider
cross-section actually improves the (validated) reversal edge, and that the 4.4× prefetch/API scale is
operationally viable. +18 tests (4,979 → 4,997).

### 1.139 — July 2026 — historical EDGAR filing-event feed (unlock the catalyst class)

New **free, point-in-time** data feed: `data/edgar_event_history`. The SEC submissions API returns
~1000 recent filings per company (10+ years for many names) with filing dates — free, no key, already
reachable via `edgar_client`. This exposes them as a historical event archive so the bot's
**catalyst** signals — previously live-only with no history to backtest against — can finally be
validated:

- `secondary_offering_short` → 424B* / S-3 / S-1 (dilution)
- `activist_13d_signal` → SC 13D
- `insider_buying` / `insider_selling_short` → Form 4
- `accounting_concern_short` → 8-K items 4.01 / 4.02

One request per symbol (reuses edgar_client's CIK map + rate-limit), cached per calendar day; pure
parse logic with the single network call isolated for testing. This is the **root capability** the
signal workshop identified: the bot's growth is bottlenecked on *signal supply* (it has one robust
cross-sectional edge, and a diversified sleeve needs several uncorrelated ones), and the whole
catalyst class was untestable for lack of history. Not yet wired into the live bot — a research feed
that unblocks backtesting. +15 tests (4,964 → 4,979).

### 1.138 — July 2026 — lottery / MAX gate (signal workshop)

A ≥+10% single-day pop marks lottery-demand overpricing (Bali-Cakici-Whitelaw MAX effect). The
workshop's standalone isolation study confirmed it at scale: a name that popped ≥10% in a single
session underperforms by **−0.44%/3d (t=−5.1), negative in 7/12 years**.

New gate: when a name has had a ≥+10% single-day return within the last 3 sessions
(`recent_lottery_pop`), the momentum family (`momentum`, `gap_and_go`) is blocked — we don't chase
the pop into its reversal. Computed identically in the live scanner (`data/market_data.py`) and the
backtest (`_compute_indicators` → `_row_to_snapshot`), and seam-verified end-to-end through
`_entry_signal`. It is a pure subtraction (only blocks entries), so the downside is bounded. +2 tests
(4,962 → 4,964).

### 1.137 — July 2026 — new signal: residual_reversal (N1, signal workshop)

The strongest confirmed idea from the 2026-07 signal workshop, and the first net-new signal from it.
Idiosyncratic 5-day losers revert over 1–3 days (liquidity-provision premium): a stock that has
underperformed SPY by ≥7% over the last 5 sessions is bought for a 3-day reversion.

Wired as **market-excess 5d return** (`ret_5d − spy_ret_5d`, both already flowing into
`evaluate_signals`) `≤ −7%`, so it needs **zero new data plumbing** and works identically in the live
scanner and the backtest engine (verified through both `evaluate_signals` and the engine's
`_entry_signal` seam). The market-excess construction fires on *stock-specific* drops — a broad
selloff hits SPY too, so the excess rarely triggers — which mutes the crash-tail risk of raw
mean-reversion; it is additionally regime-gated out of STRESS_RISK_OFF / HIGH_VOL_DOWNTREND /
CREDIT_STRESS as belt-and-suspenders.

Validation (standalone event study, 907 names, 2015–2026, winsorised excess vs SPY, cost-swept):
the −7% threshold nets **+0.31%/3d at 7bps round-trip (t=7.6), positive in 9/12 years, break-even
37.6bps** — a large cushion over the engine's calibrated costs. It outranks `mean_reversion` in
`SIGNAL_PRIORITY` as the better-validated reversion signal. +5 tests (4,957 → 4,962).

### 1.136 — July 2026 — retire golden_cross + macd_crossover (signal workshop)

First implementation output of the 2026-07 signal workshop (a full re-adjudication of the signal
book via standalone isolation event studies — 907 names, 2015–2026, winsorised excess vs SPY,
cost-swept). Both retired for having **no standalone edge**:

- **macd_crossover** — gross −0.008%/4d (t=−0.37), negative even before cost; the crossover adds
  only timing over `momentum` (which already requires MACD positive) and is subsumed by it.
- **golden_cross** — gross +0.038%/5d, break-even 3.8bps, *below* the universe baseline (5.2bps);
  an SMA50/200 cross is a monthly-horizon state variable with no 1–5d cross-sectional edge (the same
  wrong-horizon flaw that retired `death_cross` on the short side).

Both added to `GLOBALLY_DISABLED`, removed from `SYSTEM_PROMPT`, detection branches `pragma: no
cover`, and their firing tests converted to the disabled-pattern. Also corrected `docs/signals.md`,
which still listed `obv_divergence`/`obv_acceleration` as active though both were already disabled.
(Confirmed survivors from the workshop — N1 residual_reversal, N7 gap_down_reclaim, N5
overnight_accumulation, the lottery-MAX gate, and the pead muted-reaction filter — are wired in
follow-up commits.)

### 1.135 — July 2026 — startup prefetch log lines land in scheduler.log

`_startup_prefetch()` — the self-heal that warms caches on a restart when the 07:00 ET prefetch was
missed (e.g. a launchd relaunch after a power outage) — was called **before** the file-log handler was
attached, so its own log lines (`Startup: launching…`, `Pre-market prefetch starting…`) went only to
the console and never reached `scheduler.log`. After the 2026-07-03 power-outage restart this made the
prefetch look silent in the log; its progress could only be confirmed indirectly via cache-file mtimes.

Fix: move the `_startup_prefetch()` call to **after** the file handler is attached, so the startup cache
warm is visible in `scheduler.log`. No behavioural change to the prefetch itself (the `__main__` block
is `pragma: no cover`).

---

### 1.134 — July 2026 — daily email reports change since the last email (close-to-close)

The daily email headlined `daily_pnl`, which is measured **from this morning's open** and so drops
overnight gaps — successive emails didn't reconcile with the actual account balance (2026-07-01: email
+$186 while the account was +$675 since yesterday's close; the ~$558 overnight move vanished from the
email's tally).

Fix: the daily email now headlines the **change since the last email** — close-to-close, from the
prior day's closing portfolio value (`_pnl_since_last_email`), which includes the overnight move and
reconciles with the balance. Added a transparency line splitting it into **bot-intraday vs overnight**
so the reader sees how much the bot's decisions drove vs an overnight gap. The stored `daily_pnl`
(from-open) is unchanged — the weekly review and experiment still use it to evaluate the bot's
decisions. Falls back to from-open for the first-ever email. 100% covered.

### 1.133 — July 2026 — stop placement self-heals (fixed → broker-native trailing fallback)

`place_trailing_stop`'s fractional-quantity path placed a **fixed** stop at
`current_price * (1 − trail%)`. When `current_price` was stale (from the decision snapshot) and the
price had since fallen, the computed stop landed **above** the live market → Alpaca rejected it with
err 42210000 ("stop price must be less than current price") → `STOP_FAILED`. Observed live on MU
(2026-07-01): stop $1,119.66 vs market $1,078 — briefly "unprotected" until the
`ensure_stops_attached` backstop re-attached a trailing stop.

Fix: the fractional path now **falls back to a broker-native trailing stop** on any fixed-stop
rejection (and uses it directly when no anchor price is available). Alpaca anchors a trailing stop to
the LIVE market, so it can't be wrong-sided — the PRIMARY path is now self-healing instead of relying
on the backstop. `STOP_FAILED` is returned only if both the fixed and trailing attempts fail; genuine
sub-share positions still return `UNPROTECTED`. All positions were protected throughout (backstop);
this removes the dependency on it. 100% covered.

### 1.132 — June 2026 — unblock experiment outcome scoring (ATR from history) + schedule the backfill

The experiment had been collecting un-scorable observations for ~2 weeks: `experiment/backfill`'s
`forward_r` is ATR-normalised, but the live snapshot never carries an `atr` field, so the scorer read
`atr=None` and **0 of 8,107** observations scored — N_eff pinned at 0 regardless of accumulation. Two
fixes:

- **ATR computed point-in-time from price history.** `score_observation` now reconstructs the
  decision-bar ATR (price units) from the OHLC history via `_atr_at` (no bar after the decision is
  touched), instead of depending on a logged `atr` that's never present. `backfill` / the runner now
  carry OHLC. Re-running scored the backlog: **6,109 / 4,310 / 2,768** horizons closed at 1d/3d/5d.
- **Backfill wired into the scheduler** (`_backfill_outcomes`, Mon–Fri 16:15 ET, after close) — it was
  an offline step that nothing ran. Fail-safe + halt-aware.

NOTE (still open): N_eff in the monitoring summary still reads 0 — the dataset-assembly / arm-matching
pipeline (`build_dataset`) that turns scored outcomes into the effective 3-arm sample is the next,
separate, unbuilt blocker. Also flagged: a few test rows (`SYM1`/`SYM4`) leaked into the live
observations log (conftest isolation gap).

### 1.131 — June 2026 — weekly review records experiment monitoring even when the AI review fails

The experiment monitoring entry (`docs/EXPERIMENT_LOG.md`) was appended only on the weekly review's
*success* path, so when the 2026-06-28 review failed (the 1.127 truncation) the telemetry entry was
silently dropped — leaving the log stale at 2026-06-21. `build_monitoring_lines()` + `append_log_entry`
now run **before** the AI call (monitoring is descriptive telemetry independent of the AI narrative),
and the lines attach to the degraded fallback review too. So a failed/truncated/timed-out review no
longer creates a hole in the experiment telemetry. 100% covered.

### 1.130 — June 2026 — dedupe repeated AI candidates instead of fail-closing the whole run

Claude occasionally lists the same symbol twice in `buy_candidates` (e.g. `JKHY` twice in the
2026-06-30 midday run). The `DecisionSet` schema flags that as a structural error, which main.py
treats as fatal → **blocks the entire decision set** (the run's valid sells, shorts, and other buys
all discarded). This had happened **12× in 5 weeks** — ~2–3 nuked runs/week from a benign LLM repeat.

Fix: `_dedupe_candidates()` collapses repeated buy/short symbols (keeping the first occurrence) at the
AI-response choke point in `get_trading_decisions`, before validation — so a duplicate is silently
corrected and the run proceeds with intent intact, rather than fail-closed. Genuine contradictions
(a symbol in both BUY and SHORT) remain fatal. 100% covered.

### 1.129 — June 2026 — BUGFIX: correlation filter blocked buys on degenerate (r=1.00) price data

Found via a log scan of the concentration filter: **16 of 20 correlation-skips were `AAPL ↔ MSFT` /
`AAPL ↔ GOOG` at r=1.00**, clustered on four consecutive Fridays (Jun 5/12/19/26), 4× each. A perfect
1.00 between *distinct* stocks is impossible on 20 daily returns (live AAPL↔MSFT ≈ 0.34) — it's
duplicated/degenerate price data from the bulk yfinance fetch. `correlated_with_held` trusted it and
**wrongly blocked AAPL buys ~weekly** (real corr is well under the 0.70 gate → those were missed
trades). The other 4 skips (0.72–0.86) were legitimate.

Fix: fail open on an implausible correlation (`r ≥ 0.999 = _IMPLAUSIBLE_CORR`) — log a warning and
ignore the pair rather than block on bad data (consistent with the module's existing
fail-open-when-data-unavailable design). Updated the filter's blocking tests to use a realistic
high-but-imperfect correlation (r≈0.975) instead of identical series, and added a fail-open
regression test. (Residual: the upstream Friday fetch-degeneracy couldn't be reproduced live; the
guard neutralizes its impact regardless.)

### 1.128 — June 2026 — shadow measurement: do catalyst shorts have edge outside bear regimes?

Motivated by the PAYX case (we detected `eps_estimate_cut` and used it to *exit a long*, but the
regime gate forbade *shorting* it). The short universe is catalyst-enriched on every cycle, but
`scan_short_candidates` is regime-gated (ADR-006 B1), so catalyst shorts never fire outside bear
regimes — even though a catalyst (fraud, EPS collapse, guidance cut) is idiosyncratic and can play
out in any market. Before relaxing that gate we want evidence, and we're still pre-PNR, so a positive
result can be folded in before go-live.

- New `analysis/shadow_catalyst_shorts.capture()` — records every catalyst-flagged name each run,
  **regardless of regime**, with entry price + the catalyst signal(s) → `logs/shadow_catalyst_shorts.jsonl`.
  Wired fail-safe into `_run_inner` (read-only: no trades, no experiment-observation writes). A flag→
  signal map is drift-guarded against `CATALYST_SHORT_SIGNALS`.
- New `scripts/eval_shadow_catalyst_shorts.py` — scores matured rows: forward short return, market-
  excess (beta≈1), net of an assumed flat borrow, split by bear vs non-bear regime and by signal.
  (Borrow is assumed flat — no point-in-time fee feed — so treat `net` as an upper bound.)

No behaviour change to live trading; this only accrues forward evidence. If non-bear catalyst shorts
show edge net of beta+borrow, the follow-up is to relax the regime gate for the catalyst class
specifically (beta-hedged / small-sized), freeze-gated.

### 1.127 — June 2026 — the weekly email's "no trades" was a truncated review, not a quiet week

Correction to 1.126: trades **were** executed last week (broker fills + `load_history()` both confirm
~31 trades). The real cause of the email's "no trades / no lessons" was a bug in the weekly review,
not the regime. `run_weekly_review()` requested `max_tokens=2000`; on an active week the structured
JSON response (summary + worked/didn't + lessons + config_changes) overran that limit and was cut off
mid-string ("Unterminated string"), so JSON parsing failed and the function returned `None`. The
scheduler then fell back to a stub that hardcodes *"No trade history available for this week"* — which
is false whenever the failure happens on a week that actually traded.

Fixes:
- `max_tokens` 2000 → 8192 (the review is once-weekly; cost is negligible and truncation is the bug).
- On AI failure, `run_weekly_review()` now returns a **data-backed degraded review** (real trade count
  + net return from the metrics it already computed, `review_degraded: True`) instead of `None`, so the
  email reports actual activity. The genuine "no records this week" case still returns `None` → the
  stub's message is then accurate.

(The false "failing tests" in the same email was the separate `unittest.discover` → pytest issue fixed
in 1.126.)

### 1.126 — June 2026 — weekly diagnostics run pytest (not unittest) + finish the Anthropic-timeout audit

The weekly email reported "failing tests" that pass under CI. Root cause: `run_diagnostics()` ran the
suite via `unittest.discover` **in-process**, which (a) lacks pytest's conftest/monkeypatch fixture
isolation, so module-global patches leaked between tests and produced **false failures** (verified: all
6 reported failures pass under pytest), and (b) ran the tests *inside the scheduler process*, leaking
test logging into `scheduler.log`. Rewrote it to run **pytest in a subprocess** with JUnit-XML parsing
and a 30-min bounded timeout — now matches CI exactly and keeps test side effects out of the scheduler.

Also finished the 1.125 timeout audit: `analysis/weekly_review.py` and `scripts/phase0_noise_audit.py`
each constructed their own **unbounded** `anthropic.Anthropic()` client (the weekly-review one is
scheduler-reachable — same freeze risk as 1.124). Both now use `timeout=240.0, max_retries=1`.

Not bugs (documented for the record): the weekly email's "no trades this week" was the bearish
DEFENSIVE_DOWNTREND regime (blocks new BUYs) combined with the squeeze crash (1.122) aborting the short
phase until the 1.123–1.125 fixes; "no lessons learned" is downstream of that (the review only forms
lessons from ≥5-trade patterns). Both are resolved by the prior fixes.

### 1.125 — June 2026 — timeout audit: bound every Alpaca call (latent scheduler-freeze fix)

Follow-up to 1.124's AI-call hang. Audited every external call reachable from a scheduled job for
missing timeouts:

- **raw `requests`** (EDGAR, Finnhub, AAII, Alpha Vantage, insider, proxy_comp): all already carry
  explicit `timeout=10–30s` ✓
- **yfinance**: bounded by library defaults — `download`/`history` 10s, `.info`/`.earnings_dates`
  30s (verified in the installed `yfinance/data.py`) ✓
- **alpaca-py 0.43.4**: 🔴 builds a plain `requests.Session` with **no timeout**, never sets one on
  its requests, and its client `__init__` exposes no timeout option → **every broker/data call could
  hang the scheduler indefinitely** (the same failure mode as 1.124, across 6 client-construction
  sites).

Fix: new `utils/alpaca_session.with_request_timeout()` patches an alpaca-py client's session so every
request defaults to a `(10, 30)s` (connect, read) timeout; on a hang the session raises
`requests.Timeout` → propagates to the run-level handler → clean abort, scheduler frees up. Applied
at all 6 construction sites (trader, universe, quote_gate, intraday_fetcher, market_data, backtest
engine). Guard test asserts the patch actually lands on the pinned alpaca-py (so it can't silently
degrade to a no-op).

### 1.124 — June 2026 — HOTFIX: bounded AI-call timeout (a hung call froze the whole scheduler)

The close run (2026-06-26) hung on "Running AI analysis..." for ~1h: a network blip mid-call
(an Alpaca "Connection aborted" was logged seconds before) left `client.messages.create()` waiting
on a socket with no timeout. Because the scheduler runs jobs sequentially, that one hung call
**froze the entire scheduler** (process at 0% CPU, sleeping) — so every later job, including the
next morning's prefetch/open, would never fire.

Fix: construct the Anthropic client with `timeout=240.0, max_retries=1`. Normal calls finish in
~90s even on the large prompt, so 240s is generous headroom; on a hang the SDK now raises
`APITimeoutError` (an `APIError`), which `get_trading_decisions` already catches → returns `None` →
the run aborts cleanly and the scheduler frees up for the next job. Added a regression guard
(`TestClientTimeout`) so the timeout can't be silently dropped.

Separately flagged (not fixed here): the AI prompt is ~379k input tokens/call — bloated and worth
trimming.

---

### 1.123 — June 2026 — HOTFIX: squeeze gate crashed on the new fetch_error key

1.122 added `fetch_error` to `fetch_squeeze_info`'s return dict, but `_execute_shorts` spreads that
dict into `is_squeeze_risk(symbol, candidate, **_squeeze_info)` — which doesn't accept `fetch_error`,
so on the normal path (`fetch_error=False`) every short crashed with `TypeError`, aborting the whole
trading run after the buy phase. Fix: pass the SI fields explicitly
(`short_pct_float=..., days_to_cover=...`) instead of `**`-spreading the whole dict.

Why it slipped 1.122's green suite: the test mocks for `fetch_squeeze_info` returned the *old* dict
shape (no `fetch_error`), so the `**` spread never carried the bad kwarg in tests — classic
mock-drift (the exact over-mocking failure the audit flagged). Fixed the mocks to the real shape
(incl. `fetch_error`) so the squeeze-gate path is now exercised as in production, plus an explicit
regression test (`test_squeeze_gate_handles_full_squeeze_info_dict`).

Full suite green, 100% coverage; ruff + mypy clean.

---

### 1.122 — June 2026 — borrow/squeeze fail-closed-on-error + backtest/live classification test

Acts on the two open items from the 1.121 fail-open audit.

- **Borrow/squeeze gate fails closed on a fetch ERROR (behavior change).** `fetch_squeeze_info` now
  returns a distinct `fetch_error: True` on an API exception, vs `False` for a successful fetch
  (including a genuine no-data result). `_execute_shorts` **skips the short on `fetch_error`** —
  closing the hole where a transient short-interest API failure silently stripped both the borrow
  (HTB) and squeeze protections (both treat missing SI as "safe"). Legitimate no-data still permits
  (common, intentional). The AI veto is no longer the sole remaining guard on a data hiccup.
- **Backtest↔live signal classification is now a hard test.** `signals/registry.py` declares
  `LIVE_ONLY_SHORT_SIGNALS` (10 catalyst/event/live-feed signals with no historical point-in-time
  data) and `BACKTESTABLE_SHORT_SIGNALS` (3: earnings_gap_down, piotroski/accruals via fundamentals).
  `test_wiring` enforces they partition `ACTIVE_SHORT_SIGNALS` (complete + disjoint) and that every
  catalyst short is live-only — so a new active short *must* be classified, and a live signal can
  never silently be absent from the pre-registered backtest baseline (experiment-integrity guard).
  Follow-up (experiment-baseline decision, not done): tighten the engine's `_ACTIVE_SHORT_SIGNALS`
  ablation set to exclude `LIVE_ONLY_SHORT_SIGNALS` (those trade 0 in backtest, so results are
  unaffected — only the ablation table gets cleaner).

Tests across the touched modules; 100% line+branch coverage; ruff + mypy clean.

---

### 1.121 — June 2026 — wiring invariants + wire analyst_upgrade_signal (dead-wiring hardening)

After three "dead-wired" signals shipped this week (active, unit-tested, 100%-covered, but never
firing live because their data was never enriched onto the snapshot), this turns the lesson into
guards instead of vigilance — and fixes the last known straggler.

- **Wire `analyst_upgrade_signal` (long).** `_build_data_bundle` now enriches *long* snapshots with
  `analyst_revisions` too (1.120 only did the short side), so the long `analyst_upgrade_signal` — which
  had never fired in production — now does. (Behavior change, AI-vetoed.)
- **Invariant: no-orphan producers** (`test_wiring`). Every `prefetch_*` in `data/` must be referenced
  in the scheduler's prefetch job. Would have caught the `analyst_revisions` bug instantly (its
  prefetch existed but was only ever called from tests).
- **Invariant: catalyst-enrichment seam** (`test_main::TestCatalystEnrichmentSeam`). Runs the *real*
  `_build_data_bundle` with all catalyst feeds returning positives and asserts every catalyst flag
  reaches the right snapshot type *and* the signal fires end-to-end through the scanner. Catches the
  "enriched for the wrong snapshot type" flavor (the EDGAR-long-only bug) that coverage is blind to.
- **Single-source `CATALYST_SHORT_SIGNALS`** (`signals/registry.py`): the catalyst set was duplicated
  in the scanner and the seam test; now both read one constant, and the seam test iterates it — so a
  new catalyst short forces both its enrichment wiring and its scan wiring or the build fails.
- **Fail-open audit** (`docs/fail_open_audit.md`): read-only review of risk/execution gate `except`
  behaviour. Order-ledger/broker/quote gates fail-closed (correct); correlation fail-open is
  documented/intentional; the borrow + squeeze gates fail open on missing short-interest data and
  **conflate "no data" with "fetch error"** — the one actionable finding (fix proposed, not yet
  applied — a risk-posture call).

Tests across the touched modules; 100% line+branch coverage held; ruff + mypy clean.

---

### 1.120 — June 2026 — EPS estimate-revision short + wire the analyst-revision feed

Adds `eps_revision_down_short` — a cluster of ≥3 downward current-quarter EPS estimate revisions (last
30 days) that outnumber raises. The estimate-revision anomaly is one of the most replicated in the
literature (analyst cuts precede negative price drift), and unlike the index-deletion idea it is cheap:
the data is cacheable/prefetched, not scraped per-run.

- **`data/analyst_revisions.py`** now also reads yfinance `eps_revisions` (`_parse_eps_revisions`) →
  `eps_estimate_cut` flag, alongside the existing rating-shift detection.
- **Critical plumbing fix:** `analyst_revisions` was **never wired into the live pipeline** — so
  `analyst_downgrade_signal` (short) and `analyst_upgrade_signal` (long) never fired. Now
  `prefetch_analyst_revisions()` runs in the 07:00 scheduler prefetch (warmed daily, cheap reads), and
  `_build_data_bundle` enriches short snapshots with it — lighting up `eps_revision_down_short` **and**
  the previously dead-wired `analyst_downgrade_signal`. (`analyst_upgrade_signal` on the long side is
  still unwired — separate follow-up.)
- **`signals/evaluator.py`** — `eps_revision_down_short` (priority 27) firing on `eps_estimate_cut`.
- **`execution/stock_scanner.py`** — added to the RS-agnostic catalyst path (with `analyst_downgrade_signal`).
- **`core/deps.py`** — `analyst_revisions` added to `TradingDeps`.
- **`analysis/ai_analyst.py`** — `eps_revision_down_short` in `SYSTEM_PROMPT`.

Active + AI-citeable; gated by the B2 AI-veto; not backtestable (forward-evidence). 13 active short
signals now. Tests across all touched modules; 100% line+branch coverage; ruff + mypy clean.

We deliberately did **not** build a dedicated short-universe news fetch for `index_deletion_short`:
rare event, decayed/front-run edge, expensive uncached fetches. See `docs/short_disabled_backtest_findings.md`.

---

### 1.119 — June 2026 — three catalyst short signals (ADR-006 Tier-1)

The short-signal research (see `docs/short_disabled_backtest_findings.md`) found that price/technical
shorts are dead but shorts work as **catalysts**. This adds the three uncovered catalyst-short groups
whose data feeds already exist, and fixes a latent gap where catalyst flags never reached the short
snapshots at all:

- **`insider_selling_short`** — `data/insider_feed.py` now parses open-market *sales* (Form 4 code
  'S'), not just purchases; fires on a cluster of ≥3 distinct insider sellers (higher bar than the
  buy side — sells are noisier).
- **`accounting_concern_short`** — fires on an 8-K restatement / non-reliance / auditor change
  (the EDGAR `accounting_concern` flag, previously only used to *block longs*).
- **`index_deletion_short`** — `data/index_membership.classify_index_deletion` detects a name being
  *removed* from a major index (forced index-fund selling). News-derived; coverage limited to the
  long-side news set for now.

**Plumbing:** `_build_data_bundle` now enriches the short snapshots with the same EDGAR + Form-4 feeds
the long side uses — which also lights up the previously **dead-wired** `guidance_downgrade` and
`secondary_offering_short` (they read flags the short path never set). `scan_short_candidates` gains an
RS-rank-agnostic catalyst path (a corporate catalyst doesn't need the name to already be a laggard).
All three are active and AI-citeable; the B2 AI-veto gates every one before a live order. They are
**not backtestable** (no historical point-in-time event feed) and ship on forward-evidence.

Also hardened `run()`'s startup guards to `sys.exit(1)` **and** `return`, so a mocked `sys.exit` in
tests can't fall through into the live trading flow (this was masking a network call in the guard
tests). New unit + wiring + scanner tests; 100% line/branch coverage held; ruff + mypy clean.

---

### 1.118 — June 2026 — route shorts through the AI (ADR-006 part B / B2)

Shorts were taken **mechanically** after B1: the rule scanner picked them and `_execute_shorts` placed them with no AI judgement, while every long passed through Claude for ranking, veto, and context-weighting. The two sides were asymmetric — the AI could not down-weight a crowded or thesis-stale short, and the experiment had no short-side decision record to measure.

**Fix — full long/short parity.** Short candidates are now rule-gated *before* the AI call (`_build_data_bundle` → `db.short_candidates`, regime-gated so it is empty outside the bear regimes) and routed through the same structured tool call as buys:

- **`signals/registry.py`** — `ACTIVE_SHORT_SIGNALS` / `AI_CITEABLE_SHORT_SIGNALS`, derived from `SHORT_SIGNAL_PRIORITY − SHORT_GLOBALLY_DISABLED` (one source of truth, mirrors the long sets).
- **`models.py`** — `ShortCandidate` (clone of `BuyCandidate`, `key_signal` validated against the short universe); `DecisionSet.short_candidates` + no-duplicate-short and no-buy/short-conflict validators; `DataBundle.short_candidates`.
- **`analysis/ai_analyst.py`** — `short_candidates` in the tool schema (enum = active short signals), a SHORT-SIDE briefing in `SYSTEM_PROMPT`, and a rendered SHORT CANDIDATES block in `build_prompt` (present only when shorts are on offer).
- **`utils/validators.py`** — Phase-2 short context checks (known short universe; not held long).
- **`main.py`** — the AI now runs whenever there are longs **or** shorts to decide; domain-validation errors block only the offending side (buys, shorts, or both) while preserving independent sells; `_execute_shorts` consumes the AI-approved `decisions["short_candidates"]`, merges each back onto its scanned dict, and applies the **unchanged** sector/correlation/borrow/squeeze gates and standalone-vs-hedge caps. The AI's confidence now drives short sizing.

This is the load-bearing build of the short-side redesign: the bot now takes **AI-judged** shorts, and short decisions are recorded for the experiment's veto analysis. Still ahead: **B4** (backtest the disabled trend-short book before any re-enable). New unit + wiring tests across all six files; 100% line/branch coverage held, ruff + mypy clean.

---

### 1.117 — June 2026 — regime-first short gate (ADR-006 part B / B1)

`_execute_shorts` returned early unless the VIX term structure was inverted (`VIX9D/VIX > 1.05`) — a vol-panic precondition that almost never coincides with an ordinary grind-down, so the bot took **zero** shorts through the 06-18..06-24 `DEFENSIVE_DOWNTREND` even though the regime detector correctly flagged it. The VIX gate also overrode the system's own design, which already grants a *standalone* short book in bear regimes.

**Fix.** The gate is now regime-first: in the standalone-short regimes (`DEFENSIVE_DOWNTREND` / `HIGH_VOL_DOWNTREND` / `STRESS_RISK_OFF` / `CREDIT_STRESS`) a confirmed bear regime is itself the short signal, so shorts are admitted directly — VIX inversion is no longer a hard precondition there. Outside those regimes shorts remain stress hedges and still require inversion. The bot can now short in ordinary downtrends instead of waiting for a vol panic; the downstream sector/correlation/borrow/squeeze gates and the standalone-vs-hedge caps are unchanged.

**B1 of the short-side redesign (ADR-006).** Still ahead: B2 (route shorts through the AI for context parity — the load-bearing build), B3 (the index hedge is a config toggle `INDEX_HEDGE_ENABLED`, not a code bug), B4 (backtest the disabled trend-short book before any re-enable). **Note:** B1 means the bot now takes *mechanical* (not yet AI-judged) shorts in bear regimes until B2 lands. Two regression tests pin the gate. Also bundled: a conftest autouse fixture pins `INDEX_HEDGE_ENABLED` off in the test environment (config calls `load_dotenv()` at import) so enabling the index hedge in a deployment `.env` can't leak the live order path into unmocked decision-loop tests — hedge tests opt in explicitly. Full suite green, 100% coverage, ruff clean.

---

### 1.116 — June 2026 — correct sell-side exit attribution (the ai_sell mislabel)

The sell phase recorded every exit's cause as `exit_reason="ai_sell" if a position decision existed else "time_exit"` — so a position the AI said HOLD on, but which a *mechanical* rule (hard stop / regime-change / stale / adverse-volume / dust) actually closed, was logged as `SELL — {the AI's HOLD reasoning}`, stored as `exit_reason="ai_sell"`, and tagged `decision_type="sell"`. The live record thus **attributed mechanical exits to the AI** — which reads as a contradiction in the logs (the MRVL case: "SELL … hold") and would corrupt the experiment's sell-side veto analysis (you couldn't separate AI exit-skill from stop-skill).

**Fix.** `_execute_sell_phase` now threads a `{symbol: trigger}` map (`sell_reasons`) instead of a bare set; each exit path stamps its true trigger (`ai_sell` / `hard_stop` / `time_decay` / `rs_decay` / `stale_exit` / `adverse_volume` / `regime_exit` / `dust_sweep`), first-claim-wins with the AI's SELL ranked first. `_check_rule_based_stops` now returns `{symbol: trigger}` (single caller). The log (`SELL {sym} [{trigger}] — …`), the DB trade's `exit_reason`, and `all_trades`' `decision_type`/`reasoning` are all derived from the trigger — so a mechanical exit is never narrated with the AI's (often HOLD) reasoning. The regime-change exit and the AI's discretionary authority are unchanged; only the *attribution* is corrected.

Freeze-relevant: the corrected `exit_reason` is the field the experiment joins (observation ↔ trade) to separate AI-driven from mechanical exits — a sell-side data-integrity prerequisite for the PNR. New regression test pins the MRVL scenario (HOLD + stale → `rule_based` / trigger detail, not the HOLD reasoning); 100% coverage held, ruff clean.

---

### 1.115 — June 2026 — stop the DEFENSIVE_DOWNTREND long-churn (ADR-006 part A)

A multi-day live audit traced a steady paper bleed (~100k → ~97k over 06-18..06-24, **27% win rate, −0.89%/trade**) to **regime-exit churn**: the bot kept opening `pead` catalyst longs while in the `DEFENSIVE_DOWNTREND` regime, and the regime-change exit (`_DEFENSIVE_REGIMES` — force-closes any long held <2 days) liquidated them the next run, sometimes the same day, *against the AI's explicit HOLD*. Entry and exit held opposite views of the same regime: `DEFENSIVE_DOWNTREND` was `block_new_buys=False, max_orders_per_run=2` (entry permitted) yet sat in `_DEFENSIVE_REGIMES` (exit dumped). The `ai_sell` exit-reason mislabel hid that these were regime exits, not AI sells.

**Fix (entry-side only).** `DEFENSIVE_DOWNTREND`'s regime policy is now `block_new_buys=True, max_orders_per_run=0` — identical to the existing `STRESS_RISK_OFF`/`UNKNOWN` no-buy regimes, which block entries via the `max_orders_per_run=0` → `effective_max_orders` cap (an already-tested path, so **no new branch in `main.py`**). Entry now agrees with the exit: a regime bearish enough to dump fresh longs no longer opens them. The regime-change exit is unchanged — it still protects positions caught by a regime *flip*. Effect: in `DEFENSIVE_DOWNTREND` the bot now holds cash instead of churning longs.

This changes the **live trading harness**, not the experiment's candidate observations (the overlay still records every surfaced candidate; only live execution is gated). The broader short-side redesign — regime-first short gate, AI-driven shorts, index-hedge repair — is scoped in **`docs/adr/ADR-006`** as forward work; this is part A. A regression test in `test_wiring.py` locks the policy; 100% coverage held, ruff clean.

---

### 1.114 — June 2026 — logs/ cleanup: regenerable API caches → logs/caching/ (option a)

The ~20 regenerable API caches now live under `logs/caching/` (`config.CACHE_DIR`) instead of the logs/ root — completing the cache half of the deferred "option (a)" foldering, after the operator manually moved the files into the subfolder. Each cache module's path constant was repointed from `os.path.join(LOG_DIR, "x_cache.json")` to `os.path.join(LOG_DIR, "caching", "x_cache.json")` (23 constants across `data/*` + `execution/universe.py` + `market_regime`'s `spy_vix_cache.pkl`); `config.CACHE_DIR` is created at import so the directory always exists.

Critical live state (DB, baselines, regime state, records, run logs) stays at the logs/ root — the live-state fold is a separate, deliberate pass (a path bug there is F1-class, so it's not bundled with this low-risk regenerable move).

**Freeze-neutral** (cache paths touch no decision or logged experimental variable). 100% coverage held — the path constants are import-time and the cache tests patch `_CACHE_PATH`, so they're unaffected; ruff/mypy clean. Effect: on the next scheduler restart, the prefetch reads the already-moved caches in place instead of cold-rebuilding them at root.

---

### 1.113 — June 2026 — cleanup: silence sector-momentum divide warning + bump tornado (Dependabot #9)

Two freeze-neutral, behaviour-identical cleanups:

- **`data/sector_data.py` — spurious numpy warning.** `rank_sectors_by_momentum` computes `series.iloc[-1] / series.iloc[-lookback] - 1` and relies on the downstream `pd.isna(ret)` check to skip an all-zero (0/0 → NaN) ETF series; the division leaked `RuntimeWarning: invalid value encountered in scalar divide` on that intentional edge case. Wrapped in `np.errstate(invalid="ignore", divide="ignore")` — the NaN is still produced and the sector still skipped, just without the spurious warning. Coverage unchanged.
- **`tornado` 6.5.6 → 6.5.7 (Dependabot #9, medium).** CurlAsyncHTTPClient leaks per-request credentials on handle reuse. tornado is a directly-pinned dependency with no dependents, so the patch bump is conflict-free (the bot doesn't use the curl client, but patched is patched). Upgraded in the live env too.

100% coverage held; mypy gate clean.

---

### 1.112 — June 2026 — experiment freeze protocol (Point of No Return) + freeze the prompt (PNR P1)

Establishes the **Point of No Return (PNR)** — the operationalisation of the freeze `docs/EXPERIMENT.md` already requires before any primary data is collected — and takes its first prerequisite.

- **`docs/POINT_OF_NO_RETURN.md` (new).** Defines `t0` (the freeze-commit boundary), the **freeze manifest** (every frozen artifact pinned — *value hash* for data artifacts, *golden-fixture hash* for code-behaviour artifacts, so the CI guard neither false-positives on a cosmetic edit nor misses dependency-driven drift), the **frozen / safe-after / "the trap"** classification (which future requests are safe — feed repairs, fail-open→fail-safe changes, and prompt tweaks all *feel* like fixes but are frozen), the **P1–P12 checklist** (full-gauntlet: Gate A/B, A3.1 parity, fitted `evidence_score_v2`, controls+ledger, version pinning, A10.1 cost-model realism, A8.1 ET-anchoring), the **crossing ritual** (with a dry-run), and **governance** — discretionary changes bump `EXPERIMENT_VERSION` + reset the eval period, while *exogenous* changes (universe attrition, model deprecation, feed death) are handled by kind, with a bridging characterisation on a forced model swap.
- **P1 — prompt frozen.** `ADAPTIVE_PROMPT_ENABLED` flipped `True → False` (EXPERIMENT.md §15.9): the outcome-derived blocks (weekly-review lessons + performance feedback) no longer mutate the AI prompt, so the contextual arm (Arm 3) is **stationary** and per-decision IC/veto can be pooled across weeks. The self-learning loop's own value becomes a separate, pre-registered ablation. Both `build_prompt` branches were already independently covered (`test_include_adaptive_false_omits_lessons_and_feedback`), so coverage is unaffected.
- **EXPERIMENT.md §4 — delisting/halt rule.** Forward-R now specifies `exit_price` handling for a name that stops trading inside the H-window (last clean close / announced cash terms; `delisted_no_exit` excluded), frozen at t0 (manifest item 13).

Pre-data: the system remains in shakedown and pre-PNR observations stay quarantined as pilot.

Also closes a **pre-existing coverage gap** surfaced while validating this change: the live sector-correlation injection's `except` path (`get_market_snapshots`, `data/market_data.py`) was never exercised — its test used the fragile hardcoded-index `with patches[0..N]` form and silently skipped the side-effect patch (the "hardcoded-index fragility" already flagged in that test module), so the block wasn't even entered. Rewritten to the loop-enter pattern with `compute_stock_sector_corr` raising. 100% coverage restored; mypy gate clean.

---

### 1.111 — June 2026 — logs/ foldering: daily market-data caches → logs/market_data/

Daily `market_data_*.pkl` bulk caches (the bulk of the logs/ clutter) now live under `logs/market_data/` (`config.MARKET_DATA_DIR`) instead of the logs/ root. `migrate_bulk_caches_to_subdir()` runs at scheduler startup to move any legacy root-level pkls into the subfolder (no re-download); the auto-prune (1.110) now operates within that subfolder.

Critical live state (DB, position metadata, baselines, regime state), append logs, and run records remain at the logs/ root. Foldering the ~20 regenerable API caches into `logs/cache/` is a planned follow-up — each owns its own path constant and a few modules mix a cache file with a state file (e.g. `market_regime` has both `spy_vix_cache.pkl` and `regime_state.json`), so it needs a careful per-module pass rather than a blanket move.

100% coverage held; mypy gate clean.

---

### 1.110 — June 2026 — logs/ cleanup: auto-prune daily market-data caches

`logs/` had grown to ~106 MB, dominated by daily `market_data_*.pkl` bulk caches (~4-5 MB/day) that accumulated with no pruning. Manually removed 14 stale caches (06-02..06-15), reclaiming ~36 MB. Added `_prune_old_bulk_caches`: after each bulk-cache save, `market_data_*.pkl` older than `_BULK_CACHE_KEEP_DAYS` (3) are auto-deleted, so the caches no longer grow unbounded.

Foldering the remaining loose cache/state files into subfolders is a separate, larger refactor (path constants across ~15 modules + a coordinated migration of live state + a restart) — proposed, not done here.

100% coverage held; mypy gate clean.

---

### 1.109 — June 2026 — fix decisions.jsonl `executed` flag (always false)

`log_decisions` records `executed = (symbol in executed_symbols)`, but it was called inside `_run_ai_phase` — the AI-analysis phase, which runs **before** the sell/buy execution phases that populate `executed_symbols`. So the set was always empty at log time and **every decision recorded `executed=false`**. Moved the call to `_run_inner`, after the sell/buy/short phases and `_reconcile_late_fills`, so the flag now reflects what actually filled. Removed the now-unused `_decision_log` local from the AI phase.

Historical `decisions.jsonl` rows stay false (not retroactively fixable); `trades_executed` in the daily run JSONs remains the authoritative fill record.

100% coverage held; mypy gate clean.

---

### 1.108 — June 2026 — sector-momentum gate → advisory + logged (audit A3.1, step 1)

Resolves the live side of the A3.1 parity gap. The sector-momentum long gate was a **live-only** filter absent from the validated backtest; F7 had activated it, so live diverged from the backtest baseline. It is now **advisory** (`SECTOR_MOMENTUM_GATE_ENFORCE = False`): the gate's verdict (pass/block, sector, 20d-momentum rank) is recorded on each candidate and logged, but it no longer changes what trades — so the live deterministic baseline again matches the (gate-less) validated backtest, keeping the AI experiment clean.

The recorded verdict + each candidate's backfilled forward outcome make the gate a **measurable variable observationally** — no sample split, no contamination of the AI endpoint (per the agreed design). Flip `SECTOR_MOMENTUM_GATE_ENFORCE` to True only once the gate is validated as a real edge.

`_sector_gate_skip` extracted as a pure, unit-tested helper. **Next (separate):** add the gate/cap/churn as backtest toggles for a powered A/B (A3.1 step 2).

100% coverage held; mypy gate clean.

---

### 1.107 — June 2026 — pre-go-live audit fixes (circuit-breaker lookback, dust sweep, deprecation)

From the pre-go-live audit (`docs/PRE_GOLIVE_AUDIT_2026-06-17.md`):

- **A1.2 — circuit-breaker lookback.** `check_circuit_breaker` used the last 5 *records* (`[-5:]`); with 4 runs/day that was ~1.25 days, not the documented "5-day peak," so it missed slow multi-day bleeds. It now collapses to one value per calendar day (the day's last record) and takes the last 5 days. Undated records (unit tests) stay distinct.
- **A4.1 — dust sweep.** Negligible fractional residuals (e.g. a 7.89e-07-share leftover worth <$1) are auto-closed in the sell phase (`DUST_THRESHOLD_USD`) rather than lingering until the AI notices.
- **A8.1 (partial) — deprecation.** `datetime.utcnow()` → `datetime.now(UTC)` in the regime-state save. The latent `date.today()`→`today_et()` cache-key standardization is **deferred** (never triggers given the 12:00–20:30 BST run schedule; touches cache invalidation + ~13 test sites — better as a focused change).
- **A2.1 — verified clean (no fix).** Lookahead controls are strong and explicit: backtest earnings/insider/PEAD signals are point-in-time-safe ("strictly before sim_date"), the experiment backfill uses `known_at < decision_date`, live is real-time. The one known leak (distress-short fundamentals, pre-2020) is documented and scoped to the short book the experiment doesn't rely on.

Open items remain **decisions/investigations, not mechanical fixes**: A3.1 (live-only filters vs backtest parity), A7.1 (experiment power), A5.1 (resilience sweep), A9.1 (reconciliation depth), A10.1 (cost-model realism — needs accumulated trade data).

100% coverage held; mypy gate clean.

---

### 1.106 — June 2026 — harden the sector-cache build (incremental save)

Follow-up to 1.105: `build_sector_map()` fetched all ~907 symbols and saved the cache only at the very end, so a restart/crash mid-build (the first full build takes minutes over yfinance) lost all progress and started over next time. It now persists the partial map every 50 symbols (`_SECTOR_CACHE_SAVE_EVERY`), so an interrupted build resumes from the remaining symbols rather than restarting.

Also fixed mis-targeted `get_sector` patches in `test_pairs.py`: they patched `data.sector_data.get_sector`, which never applied because `data.pairs` binds it via `from data.sector_data import get_sector`. The tests silently relied on the (previously empty) sector cache returning uniform sectors; once F7 populated the cache with real sectors (e.g. GOOGL is Communication Services, not Technology), symbols split across sectors and one test broke. They now patch `data.pairs.get_sector`.

100% coverage held; mypy gate clean.

---

### 1.105 — June 2026 — wire the symbol→sector cache (audit F7)

Follow-up finding from the live logs (`pead [BULL_TREND | Unknown | 1d | conf=8]`): the symbol→sector cache was **never populated** in the live pipeline — `build_sector_map()` had no caller — so `get_sector()` fell back to a 53-symbol legacy map and returned "Unknown" for ~the entire 907-symbol universe. Same silent-graceful-degradation class as the 1.104 audit, exposed by the 500→907 universe expansion (1.102).

Effects: the **sector-momentum long gate** (`sector_allowed_long`) fails open on "Unknown" (`rank is None → allow`), so it was passing virtually every candidate instead of restricting to the top-4 momentum sectors — a documented selection filter that wasn't filtering; per-signal `by_sector` stats were almost all "Unknown"; short-sector logic was degraded the same way.

Fix: the 07:00 prefetch now calls `build_sector_map()` (incremental — loads the cache, fetches only the missing symbols from yfinance at 0.05 s each, saves), so it's a one-time full build then cheap daily top-ups. Restores both the per-sector signal-stats attribution and the sector-momentum gate. (Open follow-up: confirm the backtest applies the same gate with real sectors — a potential live/backtest divergence.)

100% coverage held; mypy gate clean.

---

### 1.104 — June 2026 — data-integrity audit fixes (P&L baseline, market-data freshness, churn guard, run-file naming)

A full integrity audit (`docs/INTEGRITY_AUDIT_2026-06-16.md`), prompted by a daily P&L logged as −$310 that was actually −$819 and a "SPY +1.7% today" narrative on a day SPY fell −0.6%, traced four defects. **None were signal-book failures** — all were instrument (data/accounting) defects, which is why the bot appeared to "underperform SPY" while largely sitting in cash or trading on stale data.

- **F1 — daily P&L baseline (HIGH; reporting + risk).** `save_daily_baseline` fired only on `mode=open` (the 10:00 run, not the 09:31 true open) and overwrote unconditionally, so any later `mode=open` invocation — e.g. a `python main.py` restart — clobbered it with an intraday value. Both `daily_pnl` *and* the daily-loss circuit breaker (which shares the baseline) then measured from a drifting baseline; the breaker could fail to trip. The baseline is now set **idempotently at the first trading run** (the true market open) and never overwritten.
- **F2 — market-data freshness (HIGH; decision quality).** The regime's "1d move" comes from the latest *complete* daily bar, which intraday is the prior session — but it was labelled "today", driving a NEUTRAL_CHOP→BULL_TREND flip on a down day and aggressive deployment into it. The regime now carries `data_as_of` + `data_is_stale`; the AI prompt labels a stale bar as "prior session (date)" and flags that today's move isn't yet reflected; every run logs a freshness warning.
- **F4 — churn guard (MED).** A *discretionary* exit of a position opened the same day (e.g. HPE bought 10:03, dumped 12:03 at −$180) is now allowed only on very-high conviction (`SAME_DAY_SELL_MIN_CONFIDENCE = 9`) or a hard negative catalyst (guidance_negative / accounting_concern / regulatory_event); otherwise it is held. Stop-losses, trailing stops, stale-age, adverse-volume and regime exits are unaffected (separate paths, always fire).
- **F3 — run-file naming (MED; observability).** Every run now writes `{date}-{mode}.json`; the open run previously wrote the bare `{date}.json`, which was easy to miss when auditing a day's logs (trades were never lost — just under a confusing name). `get_day_summary` still surfaces the open run's analysis.
- **F5 — snapshot chaining (verified benign).** Per-run cash/buying-power can re-allocate between independent broker fetches as settlements clear; equity (`portfolio_value`) chains correctly run-to-run (each gap is market drift). No fix needed.

100% coverage held; mypy gate clean.

---

### 1.103 — June 2026 — retire dead FRED series (AAII-via-FRED, ISM PMI/NAPM)

Live warnings surfaced two FRED series that no longer exist; both were silent data degradations.

- **AAII (AAIIBULL / AAIIBEAR).** AAII is not a FRED series — the bot already fetches the AAII survey correctly from aaii.com via `sentiment_client`. `fred_client.get_aaii_sentiment` queried two non-existent FRED series, so `market_data` had been injecting empty AAII into every snapshot. Removed it and pointed the injection at `sentiment_client`, mapping `extreme_bearish/extreme_bullish` and `bearish_pct*100`. Restores real AAII context to snapshots.
- **ISM Manufacturing PMI (NAPM).** ISM withdrew its data from FRED over licensing, and there is no free national replacement (regional Fed surveys use different, 0-centered semantics). The `macro_pmi_*` flags had therefore been always-False, and a monthly manufacturing survey is the wrong timescale for the bot's multi-day holds and is redundant with the faster macro signals already in place (yield curve, jobless claims, VIX, breadth). Removed the feature outright: `get_pmi_snapshot`, the `macro_pmi_*` flags (live and backtest paths), the evaluator defensive-gate term, and the position-sizer cyclical-boost branch. Behaviour-neutral (the flags were already always-False).

100% coverage held; mypy gate clean.

---

### 1.102 — June 2026 — universe expansion to S&P 500 + 400 (large + mid cap)

Widened the tradeable universe from the S&P 500 (507 symbols) to **S&P 500 + S&P 400 (907 symbols)**, large and mid cap, after validating the engine generalises down to mid cap.

**Edge check first (`scripts/universe_edge_check.py`).** An event study + full-pipeline backtest across cap tiers on the pre-holdout window. The raw signal book is large-cap-calibrated, but the live filters (cross-sectional RS-rank, regime blocks, fundamental quality screens, cost model) generalise it to mid caps (S&P 400: +0.23%/trade, +6.6% total, Sharpe 0.39). Small caps (S&P 600) stay negative even fully filtered (−0.11%/trade, Sharpe −0.15), so they are **excluded from live trading** (candidate for experiment-only use later). The universe is frozen and versioned in config for reproducibility (it is a pre-registered experimental variable).

**EDGAR prefetch dedupe.** With ~1.8× the symbols, the daily prefetch had to stay inside its 07:00→09:31 ET window. EDGAR was 58% of it, and each symbol re-downloaded its `CIK.json` up to four times (guidance / activist / secondary / narrative). `_fetch_recent_filings` now fetches the submissions once per (CIK, day), lru-cached, with `_get_recent_filings` a thin form filter over it — roughly quartering EDGAR's submissions cost, which absorbs the universe growth.

100% coverage held; mypy gate clean. README/config provenance updated.

---

### 1.101 — June 2026 — data-feed integrity sweep + experiment material-context coverage

Pre-data-collection hardening for the AI-alpha experiment. The bot degrades gracefully on any data failure, so a broken feed returns a neutral default and stays invisible — exactly how several feeds had silently rotted. This release makes feed health explicit and broadens the experiment's material-context coverage.

**New: data-feed health gate.** `experiment/feed_health.py` (pure classifier, 100% covered) plus `scripts/feed_health_check.py` probe every live feed and report OK/EMPTY/DEGRADED/STALE/ERROR, exiting non-zero if any need attention. Run before each collection window and as ongoing monitoring. Feeds that are legitimately empty most days (insider buys, earnings-in-window, high short interest) are probed for *machinery* health, not the rare qualifying result, so the gate does not cry wolf. Current state: 21/21 green.

**Four silently-degraded feeds repaired** (all found via the gate / its build):
- **AAII sentiment** — missing `xlrd` dependency plus a NaN-row parse bug; survey now parses (added `xlrd==2.0.2`).
- **8-K guidance** — the classifier read the cover page (always neutral); now reads the EX-99 exhibits, and the keyword lists were enriched (~70 terms each, word-boundary matched).
- **FinBERT news sentiment** — built with `return_all_scores=True`, which transformers 5.x silently collapses to top-1, so every classification returned None; switched to `top_k=None`. `torch`/`transformers` installed.
- **Insider Form 4** — EDGAR drifted `primaryDocument` to the XSL-styled HTML view, which is not parseable XML; the error was swallowed, so insider activity was blank for every symbol. Strip the `xsl.../` prefix to fetch the raw ownership XML.

**Class-share ticker normalisation (BRK.B, BF.B).** yfinance uses a hyphen for class shares (`BRK-B`, `BF-B`) and returns zero rows for the dot form the universe stores (`BRK.B`, `BF.B`) — so Berkshire Hathaway and Brown-Forman, both S&P 500 names, were silently dropped from *every* yfinance-backed feed (prices, earnings, short interest, news) on every run, surfacing only as a recurring "Insufficient data" warning. Added `utils/symbols.to_yf_symbol` and applied it at the yfinance query boundary in `market_data` (per-symbol and bulk, mapping results back to the original symbol), `earnings_surprise`, `short_interest`, and `news_fetcher`. Verified live: both names now return prices, short interest, earnings, and headlines.

**Experiment: all ten material-context categories now wired.** Mapping the ten pre-registered categories to feeds showed four were unwired. Three are now detected from EDGAR 8-K item codes (`data/edgar_client.py`): M&A (item 2.01, or 1.01 + keyword confirmation), accounting concern (4.02/4.01), regulatory event (3.01, or 8.01 + keyword confirmation). The tenth, index inclusion/deletion, is detected from news headlines (`data/index_membership.py`) since the index providers expose no clean point-in-time API but membership changes are reliably newsworthy (high-precision phrasing — a change verb plus a preposition next to a named index — avoids generic index mentions). All are direction-agnostic enrichment flags merged onto the snapshot (`main.py`) and read by `experiment/material_context.py` — the engine still selects the candidate; the AI judges the implication. See `docs/EXPERIMENT.md` §15.1.

**Dependencies:** added `xlrd==2.0.2`; installed `torch`/`transformers` (already in requirements); bumped idna, tornado, anthropic, alpaca-py, pydantic for Dependabot advisories.

**Tests:** new coverage for the feed-health classifier, the FinBERT shape fix, the insider XSL-path fix, and the three EDGAR narrative detectors (every item-code and keyword-confirmation branch). 100% coverage maintained.

---

### 1.100 — June 2026 — 100th release: full line-by-line audit (all ~89k lines) + mypy gate cleanup

The 100th commit. Audited every source file line-by-line (report in `docs/audit_v1.100.md`) and cleared the mypy backlog. No Critical findings — the fail-closed broker core, fail-safe data layer, and dormant-by-design AI self-modification all held up. The fixes below are the High/Medium items that audit surfaced.

**Risk / sizing:**
- **R1 — drawdown_scalar plausibility floor is now account-relative.** The hardcoded `_MIN_PLAUSIBLE_VALUE = 1_000.0` would have discarded the entire portfolio-value history of a sub-$1k account (e.g. the ~$150 `SMALL_ACCOUNT_MODE` account), silently disabling the drawdown circuit-breaker. Replaced with `max(10.0, peak * 0.5)` — scales to the account and still rejects implausible zero/garbage reads.
- **R2 — `SIGNAL_SHARPE_MULTIPLIER` brought back in sync with the book.** `range_reversion` moved to the disabled (0.0) section and the v1.99-disabled longs (`obv_divergence`, `obv_acceleration`, `volume_climax_reversal`, `tax_loss_reversal`) added at 0.0. Documentation/telemetry only — these signals can never fire — but it stops the AI prompt from quoting a non-zero multiplier for a dead signal.

**Macro / calendar:**
- **M1 — `get_macro_risk` docstring corrected.** NFP is *deliberately* excluded from the high-risk set (it releases pre-market and the reaction is absorbed before our 10:00 ET buy window — enforced by `test_nfp_date_is_not_high_risk`); the docstring wrongly implied it was included.
- **M2 — macro-calendar expiry warning.** `get_macro_risk` now logs a warning when queried past the last hardcoded macro-event date so a stale calendar can't silently degrade to "no events ever" without an operator signal.

**Short side:**
- **S1 / A3 — live Path D rebuilt around the failed-bounce short.** `scan_short_candidates` Path D now fires `post_earnings_gapdown_failed_bounce` (the v1.99 catalyst short) and blocks the superseded naive `earnings_gap_down` in every live path (`_live_blocked`); `earnings_gap_down` remains active in the backtest engine only.
- **E1 — late-fill recovery for shorts.** `place_short_order` now mirrors `place_buy_order`: if the order fills just after `wait_for_fill` gives up, a final `get_order_by_id` check recovers it as FILLED instead of losing it to a spurious TIMEOUT.
- **D1 — `get_short_interest` schema docstring** now documents the `short_pct_float` field it actually returns.
- **S2 — `score_candidate` co-firing term clamped.** The stale `n_signals / 8` denominator is now `min(n_signals, 5) / 5`, bounding the telemetry score to [0, 1] (display/ranking only — does not gate trades).

**Index hedge (opt-in, from v1.99):**
- **A1 — index hedge respects the fat-finger / daily-notional guards.** `_execute_index_hedge` now runs `check_pre_trade` (against `MAX_SINGLE_ORDER_USD` / `MAX_DAILY_NOTIONAL_USD`) before placing, and books the notional into the daily tally on fill — previously the hedge could bypass both limits.
- **A2 — documented the no-stop design** of the index hedge (covered by regime exit, not a trailing stop) so it reads as intentional.

**Config / data hardening:**
- **F1 — `config.validate()` bounds new params.** `INDEX_HEDGE_WEIGHT` ∈ (0, 0.5]; `SHORT_SIZE_SCALE`, `MAX_SHORT_STANDALONE_RATIO`, `MAX_SHORT_HEDGE_RATIO` ∈ (0, 1] — fail-fast on misconfiguration.
- **F2 — pruned stale `SIGNAL_MAX_HOLD_DAYS` entries** for the now-disabled signals so the map reflects only the active book.
- **D2 — per-symbol fault isolation in `_live_fetch_earnings`.** A malformed payload for one symbol no longer aborts the whole batch — fetch+parse run in a single per-symbol `try`, logging and skipping the bad symbol.

**Types:** cleared the mypy backlog and expanded the `[tool.mypy]` gate to 11 modules (`disable_error_code=["import-untyped"]`); explicit None-checks and coercions in `data/fundamental_cache.py` / `data/fundamentals.py`, casing fix in `risk/exit_optimiser.py`, removed stale `type: ignore`s across the data feeds.

**Methodology note (B-obs):** the v1.98/v1.99 signal disables were validated on a 2020–2022 / 2015–2026 window. They should be re-validated on the pre-2024 holdout window before being treated as permanent; tracked for a future release.

**Tests:** new coverage for every fix above (drawdown floor, macro expiry, Path D failed-bounce, short late-fill recovery, index-hedge pre-trade gating, earnings parse isolation, config bounds). Full 100% coverage maintained.

---

### 1.99 — June 2026 — Signal book rationalisation + short-book rebuild (borrow model, catalyst short, index hedge)

Continued the v1.98 rationalisation after a targeted ΔSharpe test and a short-side design review, then rebuilt the short book around the structural problems that review identified.

**Short-book rebuild (new):**
- **`data/borrow_cost.py`** — stock-borrow cost estimator. No paid cost-to-borrow feed exists, so the annualized borrow rate is estimated from short-interest tiers (the strongest free proxy): <5% float → 0.5% GC, 5–15% → 3%, 15–30% → 10%, 30–50% → 30% (hard-to-borrow), >50% → 80%. `estimate_borrow_rate`, `is_hard_to_borrow`, `borrow_cost_usd`. This closes the blind spot where every prior short backtest modelled borrow as free, overstating short P&L.
- **Backtest borrow cost** — `_run_short_simulation` and `_run_combined_simulation` net borrow cost from short P&L at every cover (opt-in via `borrow_rate_by_symbol`, default off so legacy results are unchanged). `run_combined_analysis(use_quality_fundamentals=True)` now derives per-symbol borrow rates from the short-interest data it already fetches.
- **Live borrow gate** — `_execute_shorts` estimates each candidate's borrow rate, skips hard-to-borrow names (before the squeeze gate, which uses a lower SI threshold), and records `borrow_rate_annual` on every short trade.
- **`post_earnings_gapdown_failed_bounce`** (new active short) — negative-PEAD continuation entered *after* the reflexive bounce fails, not on the gap bar. Computed live in `scan_short_universe` from daily OHLCV (`detect_failed_gapdown`): a ≥7% earnings/news gap-down whose low is subsequently broken. The failed-bounce filter removes the dead-cat-bounce losses that make the naive gap-day short unreliable — the one short with a documented short-horizon edge. Accumulating live evidence.
- **`index_regime_hedge`** (new, opt-in) — `_execute_index_hedge` shorts an index ETF (`INDEX_HEDGE_SYMBOL`, default SPY) at `INDEX_HEDGE_WEIGHT` of the portfolio in confirmed bear regimes (`INDEX_HEDGE_REGIMES`) and covers when the regime exits. Index ETFs borrow cheap, are deeply liquid, and carry no single-name squeeze risk. Disabled by default (`INDEX_HEDGE_ENABLED`) — it is a live order path; honours `dry_run`/`_live_shadow`. Backtest overlay via `compute_index_hedge_pnl`, reported as `result["index_hedge"]`.

**Long signals disabled (`GLOBALLY_DISABLED`):**
- `obv_divergence` + `obv_acceleration` — joint removal **ΔSharpe +0.12, ΔReturn +7.0%** on the 2020–2022 combined long/short window (targeted elimination test, `scripts/obv_elimination_test.py`). The two together were 44% of all long trades but the mechanism is slot competition: removing them frees slots for `pead` (604→692 trades, WR 54%→56%). `obv_acceleration` was WR 44% / negative avg in every window; `obv_divergence` was regime-inconsistent (+0.50% in 2020–2022, −0.07% on the full 2015–2026 run).

**Short signals disabled (`SHORT_GLOBALLY_DISABLED`):**
- `death_cross`, `altman_distress_short`, `gross_margin_deterioration_short` — the three active fundamental shorts that were dragging in the combined production backtest. All three are *confirming* signals (fire after the market has already shorted the name) and encode multi-month theses that cannot resolve inside our 1–5 day hold. `earnings_gap_down` remains the one catalyst-anchored active short.

**Short-book diagnosis:** The structural reasons the short book underperforms — confirming-not-leading signals, holding-period/thesis mismatch, no catalyst anchor, no borrow-cost model, and fighting the equity risk premium — are documented in `docs/signals.md`. The rebuild above addresses the first four directly.

**`SIGNAL_PRIORITY`** now 41 entries (31 active, 10 in `GLOBALLY_DISABLED`). **`SHORT_SIGNAL_PRIORITY`** gains `post_earnings_gapdown_failed_bounce`.

**Tests:** Converted 13 signal-fires tests to globally-disabled assertions across `test_backtest.py` and `test_new_signals.py` (7 were latent failures from the v1.98 disable that shipped without test updates — fixed here). New suites: `test_borrow_cost.py`, plus borrow/gap/hedge coverage in `test_backtest.py`, `test_short_side.py`, `test_new_signals.py`, `test_main.py`, `test_market_data.py`. Full 100% coverage maintained.

---

### 1.98 — June 2026 — Institutional-grade audit: 12 critical/high findings hardened + signal book rationalisation

Full-codebase institutional audit covering AI governance, broker safety, signal wiring, scheduler reliability, secrets hygiene, and observability. All findings addressed in a single release alongside signal book rationalisation.

**Critical safety fixes:**
- **C2 — `main._execute_buy_phase()`** — `key_signal` attribution bug: `record_buy` was reading `candidate.get("key_signal")` (the raw, potentially hallucinated AI value) instead of the corrected and validated local `key_signal` variable used for the 12 sizing scalars. One-line fix; regression test added.
- **C3 — `main._run_inner()`** — Options signals dead-wiring: post-filter re-evaluation stored results in `s["signals"]` but never updated `s["matched_signals"]`, so options signals could never propagate to sizing logic. Merge loop now syncs both structures after re-evaluation; wiring test added.
- **C4 — `main._execute_shorts()`** — Short execution path lacked fat-finger cap, daily-notional accounting, and `MAX_DEPLOYED_USD` check. Added `check_pre_trade()` + `add_daily_notional()` calls mirroring the long path; `today` param threaded through; test added.
- **C6 — `utils/audit_log.has_open_buys_run_today()`** — DB exception returned `False` (assume not yet run) — opposite of every other safety guard. Changed to return `True` (fail-closed); test updated.

**Governance fixes:**
- **C1 — `analysis/ai_analyst.SYSTEM_PROMPT`** — Complete rewrite: removed 4 globally-disabled signals (including `vix_fear_reversion` which was described as "highest-priority"), added all 33 active signals in organised family blocks. Parity tests: `test_system_prompt_contains_no_globally_disabled_signals` + `test_system_prompt_mentions_all_active_long_signals`.
- **C8** — `NEUTRAL_CHOP` regime description now advises mean-reversion block and catalyst-confirmed entry preference.
- **C11** — Removed misleading "The scheduler sets this in its environment;" comment from `LIVE_CONFIRM`.
- **C12** — "SOCIAL SENTIMENT" section header renamed to "ANALYST CONSENSUS" to match actual data source.

**Observability:**
- **Full LLM response** now persisted to audit store (not just 500-character snippet).

**Infrastructure:**
- **C7 — `.gitignore`** — Added `.env.*` / `!*.env.example`; ran `git rm --cached .env.canary` to untrack the accidentally committed canary env file.
- **Lock liveness** — Stale-lock age heuristic (30-min) replaced with PID-based liveness check (`os.kill(pid, 0)`); PID written to lock file payload. Test updated to write valid JSON lock payload.
- **Circuit-breaker `_MIN_PLAUSIBLE`** — Hardcoded `1_000.0` (which silently disabled the 5-day drawdown circuit breaker for accounts under $1,000) replaced with `max(10.0, peak_raw * 0.5)` — account-size-relative floor.

**Signal book rationalisation:**
- **Disabled** (`GLOBALLY_DISABLED`, v1.98): `range_reversion` (2 production-backtest trades, WR 0%, avg −16.2%; backward elimination Step 3), `volume_climax_reversal` (1 trade, WR 0%, avg −2.8%), `tax_loss_reversal` (38 trades, WR 37%, avg −1.02%).
- **Elevated**: `fcf_yield_signal` priority 29 → 12 (563 backtest trades, WR 51%, avg +0.16%).
- **Wired live**: `options_skew_signal`, `unusual_options_activity`, `put_call_contrarian`, `squeeze_setup_long`, `squeeze_momentum_long`, `short_interest_trend_long` — now fully active post-C3 fix; accumulating live evidence from v1.98.
- **`SIGNAL_PRIORITY`** now 41 entries (33 active, 8 in `GLOBALLY_DISABLED`).
- `SYSTEM_PROMPT` signal families and `docs/signals.md` updated for full parity.

---

### 1.97 — June 2026 — Five new data pipelines + 15 long / 6 short new signals: analyst revisions, fear/greed, Google Trends, lockup calendar, ERP gate

The deepest signal expansion to date: 15 new long signals and 6 new short signals spanning options microstructure, fundamental quality, short-squeeze mechanics, alternative data, and cross-asset pairs, underpinned by 5 new data pipelines. Also fixes a latent options dead-code bug where options signals were evaluated before options data was injected.

**New data modules (4):**
- **`data/analyst_revisions.py`** — daily-cached analyst recommendations via yfinance `recommendations_summary`. Detects upgrades (buy% rose >10pp month-over-month) and downgrades (sell% rose >10pp or buy% fell >10pp). Minimum 3 analysts required; ETF symbols skipped. `prefetch_analyst_revisions()` warms cache during the pre-market run.
- **`data/fear_greed.py`** — composite 0–100 fear/greed index: VIX component 30%, AAII 25%, NH/NL breadth 20%, SPY momentum 15%, breadth % above SMA50 10%. `is_extreme_fear(score < 20)` and `is_excessive_greed(score > 80)` predicates.
- **`data/google_trends.py`** — per-symbol search-interest spike detection via `pytrends`. Spike fires when current week ≥ 150% of 12-week baseline average (minimum baseline 10). Graceful `ImportError` fallback when `pytrends` is not installed. Daily-cached per date key.
- **`data/lockup_calendar.py`** — IPO lock-up expiry tracker. Detects IPO date from `yf.Ticker.info` (`ipoExpectedDate` or `firstTradeDateEpochUtc`) or price-history first-trade date. Alerts 5–10 calendar days before the 180-day lock-up expires. `refresh_ipo_dates()` prunes stale entries (> 550 days old).

**Updated data modules (4):**
- **`data/fred_client.py`** — `get_10y_yield()` (latest DGS10) and `get_aaii_sentiment()` (bulls%, bears%, extreme_fear, excessive_bulls) added.
- **`data/fundamental_cache.py`** — `compute_accruals_ratio()` (net income minus operating cash flow, normalised by average total assets) and `get_accruals_ratio()` public getter added.
- **`data/short_interest.py`** — `short_pct_float` coercion hardened: non-numeric values (e.g. `"n/a"`) caught by `except (TypeError, ValueError)` → `None`.
- **`data/market_data.get_market_snapshots()`** — injects `macro_10y_yield`, `aaii_*` sentiment fields, `analyst_*` revision flags, `lockup_expiry_soon`, `lockup_days_to_expiry`, and `google_trends_bullish` into every snapshot. Each pipeline is exception-guarded independently so a single feed failure does not abort the run.

**New gates in `signals/evaluator.py` (9):**
- *Fundamental quality:* `altman_z < 1.1` → `_DISTRESS_BLOCKED` (momentum, breakout, gap_and_go, bb_squeeze); `piotroski_score < 3` → `_PIOTROSKI_GATED` (momentum, breakout); `forward_pe > 60` → `_EXPENSIVE_BLOCKED` (momentum, breakout, macd_crossover); `gross_margin_trend < −0.03` → `_GM_GATE_BLOCKED` (momentum, trend_pullback, guidance_raise, pead); `accruals_ratio > 0.10` → `_EXPENSIVE_BLOCKED`.
- *Market microstructure:* `nhl_ratio < 0.5` → `_WEAK_BREADTH_BLOCKED` (gap_and_go, momentum, bb_squeeze, inside_day_breakout, orb_breakout, intraday_momentum); `sector_correlation_20d > 0.75` → blocks momentum, breakout_52w, bb_squeeze; ERP gate (`1/forward_pe − 10y_yield/100 < 0.01`) → `_EXPENSIVE_BLOCKED`; `aaii_excessive_bulls` → blocks gap_and_go, momentum, breakout_52w.

**New long signals in `signals/evaluator.py` (15):**
- *Options/IV:* `iv_vs_rv_spread` (ATM IV/RV < 0.70 — vol genuinely cheap vs realised), `options_skew_signal` (panic put-skew or call-skew spike), `unusual_options_activity` (OTM call OI surge — informed upside), `put_call_contrarian` (P/C OI > 2.5 + trend → contrarian long).
- *Squeeze:* `squeeze_setup_long` (crowded dormant short at 20d low — pre-squeeze), `squeeze_momentum_long` (high SI + strong return + above 20d high — squeeze in motion).
- *Alternative data:* `short_interest_trend_long` (SI% falling >30% from peak + price rising), `analyst_upgrade_signal` (buy% rose >10pp), `aaii_extreme_fear_long` (AAII bears > 50%), `fear_greed_extreme_fear` (composite index < 20), `google_trends_bullish` (search-interest spike).
- *Fundamental/catalyst:* `activist_13d_signal` (13D filing within 30 days + EMA aligned), `guidance_raise_signal` (positive 8-K guidance; fires without price confirmation), `fcf_yield_signal` (FCF yield > 5% + Piotroski F ≥ 5).
- *Cross-asset:* `sector_pair_mean_reversion` (intra-sector RS spread z-score extended → long the laggard).

**New short signals in `signals/evaluator.py` (6):**
`altman_distress_short` (Z < 1.1 in bear/stress regime), `piotroski_distress_short` (F ≤ 2 + bearish EMA), `gross_margin_deterioration_short` (GM trend < −5pp), `accruals_quality_short` (accruals ratio > 0.15 + extended price), `lockup_expiry_short` (lockup expires in 5–10 days + bearish EMA), `analyst_downgrade_signal` (sell% rose >10pp or buy% fell >10pp).

**Options dead-code fix (`main.py`):**
Options signals require `iv_rank`, `put_call_ratio`, `unusual_call_oi`, and related fields, which are injected by the options pipeline **after** `prefilter_candidates()`. Previously these signals could never fire because the pre-filter ran before injection. Fixed: `_run_inner()` now performs a post-filter signal re-evaluation pass that re-scores filtered candidates against the fully options-enriched snapshot. Google Trends injection also added (exception-guarded, WARNING log on failure).

**`config.SIGNAL_MAX_HOLD_DAYS`** — added entries for all 15 new long signals (2–5 days).

**`SIGNAL_PRIORITY`** now 41 entries (36 active, 5 in `GLOBALLY_DISABLED`). **`SHORT_SIGNAL_PRIORITY`** now 23 entries (9 active, 14 in `SHORT_GLOBALLY_DISABLED`).

**Tests:** ~275 new tests across 11 new/modified test files. 4,494 tests total. 100% line and branch coverage on all new and changed files.

---

### 1.96 — June 2026 — Institutional-grade system review: critical safety fixes + registry unification

Full-codebase review identifying and fixing 17 findings across crash safety, sizing, signal governance, data integrity, and documentation accuracy.

**Critical fixes (run-blocking bugs):**
- **C1 — `risk/regime_policy.py`** — Added `CREDIT_STRESS`, `LATE_CYCLE_BULL`, and `RECOVERY` policies to `REGIME_POLICY`. These three `MarketRegime` enum values had no policy entry, causing a `KeyError` crash mid-run. Added module-level totality assertion (`assert set(MarketRegime) == set(REGIME_POLICY)`) so any future gap is caught at import time. `get_regime_policy()` now uses `dict.get` with a safe UNKNOWN fallback + logged alert instead of bare `[]` access.
- **H7 — `execution/trader.py`** — `get_daily_notional()` and `get_open_shorts()` previously returned `0.0` / `set()` on DB failure — silently resetting the daily notional cap and bypassing the short-slot count. Both now raise `OrderLedgerUnavailable`, which callers treat as buy/short-blocking.
- **M7 — `main._evaluate_risk_limits()`** — Daily-loss liquidation loop now wraps each `close_position()` call in a try/except. Failed closes are collected; if any fail, a halt file is written and an alert is sent. Previously a failed close was silently ignored with no verification.
- **H6 — `execution/trader.cancel_open_orders()`** — Rewrote from cancel-all to a two-phase approach: scoped cancel first (symbol-only orders via `GetOrdersRequest(symbols=[symbol])` + `cancel_order_by_id`), with cancel-all fallback only if shares are still held after the scoped cancel (GTC trailing-stop workaround). This prevents stripping stops from all other positions on every sell.

**Sizing fixes:**
- **C2+H5 — `main._execute_buy_phase()`** — `regime_policy.position_size_multiplier` (previously a dead field) is now multiplied into the notional chain. After all 12 scalars are applied, a hard cap of `account_now["portfolio_value"] × config.MAX_POSITION_WEIGHT` (15%) is enforced, preventing the joint product of multipliers from exceeding the documented position-weight limit.

**Signal registry unification (H1):**
- **`signals/registry.py`** (new) — Single source of truth deriving `ACTIVE_LONG_SIGNALS` and `AI_CITEABLE_SIGNALS` from `SIGNAL_PRIORITY − GLOBALLY_DISABLED`. Eliminates five independent copies of the signal list.
- **`models.VALID_BUY_SIGNALS`** — Now derived from `AI_CITEABLE_SIGNALS`; phantom signals (`news_catalyst`, `rsi_oversold`, `trend_continuation`) and disabled signals removed automatically.
- **`analysis/ai_analyst._DECISION_TOOL`** — `key_signal` enum now built from `sorted(AI_CITEABLE_SIGNALS)` instead of a hardcoded list. Disabled signals (`rs_leader`, `momentum_12_1`, `vix_fear_reversion`, `breakout_52w`) are no longer offered to the AI.
- **`risk/position_sizer.SIGNAL_SHARPE_MULTIPLIER`** — Disabled signals zeroed (`rs_leader: 0.0`, `momentum_12_1: 0.0`, `vix_fear_reversion: 0.0`); previous 1.2× and 1.1× boosts for globally-disabled signals removed.
- **`main._execute_buy_phase()`** — Added `key_signal ∈ matched_signals` cross-check: if the AI cites a signal that didn't actually fire on the candidate, the highest-priority fired signal is used for sizing instead.
- **`config.SIGNAL_MAX_HOLD_DAYS`** — Added entries for 8 active signals that were missing: `range_reversion` (2d), `golden_cross` (5d), `candle_exhaustion` (3d), `obv_divergence` (3d), `obv_acceleration` (3d), `volume_climax_reversal` (3d), `breadth_thrust` (4d), `tax_loss_reversal` (5d).

**Stop-exit outcome recording (H3):**
- **`execution/trader.py`** — New `_record_stop_exit_outcome()` function queries recent closed SELL orders for stale symbols during `reconcile_positions()`. Broker-side stop exits are now recorded via `record_trade_outcome()` with actual fill price, fixing the survivorship bias that was systematically inflating signal win-rates.

**Signal correctness:**
- **M1 — `signals/evaluator.py`** — `tax_loss_reversal` now checks `snapshot.get("price_vs_52w_high_pct") is not None` before evaluating the drawdown threshold. The −999 sentinel default previously satisfied `< −30%` on any snapshot missing that field, causing false signals in January.
- **M2 — `main._execute_buy_phase()`** — `place_trailing_stop` now receives raw `buy_result.filled_qty` (float) instead of `int(math.floor(...))`. The function's fractional branch (whole-share stop + remainder liquidation) now correctly executes; previously the floor discarded the fractional tail, leaving it unprotected.

**Data integrity:**
- **M5 — `config.py` / `execution/universe.py`** — Removed `Q` (Quintiles; IQV already present) and `MRSH` (no current S&P 500 constituent) from `STOCK_UNIVERSE`. Both added to `_EXCLUDED_SYMBOLS` to prevent re-entry via dynamic expansion.
- **M6 — `main._execute_buy_phase()`** — `record_buy` now uses `buy_result.filled_avg_price` as `entry_price` when available, instead of the pre-trade snapshot price.

**Model correctness:**
- **L1 — `models.PositionDecision`** — Added `confidence: int = Field(ge=1, le=10, default=5)`. The AI tool schema required `confidence` for SELL decisions but the Pydantic model silently dropped it, making `decision.get("confidence")` always return `None` in sell-phase logging.

**Wiring/consistency tests (11 new):**
`test_regime_policy_covers_all_regimes`, `test_get_regime_policy_returns_for_every_regime`, `test_valid_buy_signals_derived_from_registry`, `test_no_globally_disabled_signal_in_ai_citeable`, `test_all_active_signals_have_hold_days`, `test_ai_tool_enum_matches_registry`, `test_max_position_weight_respected_with_all_scalars_at_max`, `test_readme_risk_numbers_match_config`, `test_get_daily_notional_raises_on_db_failure`, `test_get_open_shorts_raises_on_db_failure`, `test_tax_loss_reversal_does_not_fire_on_missing_data`.

**README corrections (H4):**
- `RISK_PER_TRADE_PCT`: "0.25%" → "0.6%"
- `MAX_POSITION_WEIGHT`: "5% per position" → "15% per position" (two instances)
- Daily-loss halt behaviour: auto-resumes next day; halt file only written on close failure
- Short regime gate: "BULL_TREND and NEUTRAL_CHOP only" → bear regimes only (STRESS_RISK_OFF, HIGH_VOL_DOWNTREND, DEFENSIVE_DOWNTREND, CREDIT_STRESS)
- Per-signal hold-days list updated to reflect current active signals

**Tests:** 11 new (all in `tests/test_wiring.py`). 4,220 tests total. 100% coverage on all changed lines.

---

### 1.95d — June 2026 — Batch 5 microstructure signals + NEUTRAL_CHOP confidence fix

Adds three market microstructure signals and fixes the `min_confidence_bump=1` bug in `NEUTRAL_CHOP` (the same regression that was fixed in `DEFENSIVE_DOWNTREND` in v1.95c).

- **`risk/regime_policy.py`** — `NEUTRAL_CHOP.min_confidence_bump` corrected from `1` to `0`. The erroneous value was raising the AI confidence threshold by +1 in a regime where no such bump is warranted.
- **`data/sector_correlation.py`** (new) — `compute_stock_sector_corr(symbol, etf, price_data)` computes a rolling 20-day Pearson correlation between a stock and its sector ETF (mapped via `get_sector_etf()`). Returns `float | None`; falls back to yfinance if price data is not pre-loaded. `_get_df()` helper handles cache vs. live fetch.
- **`risk/position_sizer.correlation_scalar(corr)`** — new multiplier: `0.85×` when `corr > 0.75` (dampens size when the stock moves in lockstep with the sector); `1.10×` when `corr < 0.35` (boosts when the stock is decorrelated); `1.0×` otherwise. `None` → `1.0`.
- **`risk/position_sizer.nhl_scalar(nhl_ratio)`** — new multiplier: `1.10×` when NH/NL ratio `> 2.0` (broad expansion supports longs); `0.80×` when `< 0.5` (contraction pressure); `1.0×` otherwise. `None` → `1.0`.
- **`data/market_data.get_intraday_data()`** — computes `premarket_gap_retrace`: `True` when a gap ≥ 2% has retraced more than 50% of its opening distance by the 09:35 bar (first 5 one-minute bars). Added to every intraday snapshot dict.
- **`data/market_data.get_market_snapshots()`** — injects `nhl_ratio` (from `BreadthSnapshot.nh_nl_ratio`) and `sector_correlation_20d` (per-symbol 20d rolling correlation vs. sector ETF) into all live snapshots. ETF price data is bulk-downloaded once via `_bulk_download` and reused across all symbols.
- **`signals/evaluator.py`** — `premarket_gap_quality` gate: when `premarket_gap_retrace=True`, `gap_and_go` is added to `blocked`. Suppresses the signal when opening gap momentum has already evaporated.
- **`main._execute_buy_phase()`** — `_corr_scalar` and `_nhl_scalar` multiplied into the notional chain (after `_macro_scalar`); both logged when ≠ 1.0. Notional chain is now 12 multipliers deep.
- **Tests:** 48 new tests. 4,209 tests total. 100% coverage on all changed lines.

---

### 1.95c — June 2026 — Batch 4 macro/rates signals: credit_spread_gate, duration_flight, copper_gold_ratio, dollar_strength, yield_curve regime, PMI regime, initial_claims

Adds eight macro and rates-driven signals/gates that inject real-time credit, FX, yield-curve, and PMI data into the scanner and position sizer.

- **`data/fred_client.get_pmi_snapshot()`** — fetches FRED NAPM series; returns `{latest, ma_3m, expanding (ma_3m > 55), contracting (latest < 45)}`.
- **`data/macro_data.get_combined_macro_flags()`** — merges ETF-derived `MacroSnapshot` with FRED yield-curve and PMI flags into a flat `macro_*` dict injected into every stock snapshot.
- **`signals/evaluator.py`** — three new macro gates consuming `macro_*` snapshot fields:
  - `macro_credit_stress` → adds `_HIGH_VOL_BLOCKED` (`breakout_52w`, `momentum`, `gap_and_go`, `orb_breakout`, `candle_exhaustion`, `breadth_thrust`).
  - `macro_duration_flight | macro_claims_deteriorating | macro_pmi_contracting` → adds `_DEFENSIVE_BLOCKED` (`breakout_52w`, `momentum`, `gap_and_go`, `macd_crossover`, `inside_day_breakout`, `range_reversion`).
  - `macro_yield_curve_inverted_days >= 20` → adds `_LATE_CYCLE_BULL_BLOCKED` (`_DEFENSIVE_BLOCKED` + `mean_reversion` + `iv_compression`).
- **`risk/position_sizer.macro_scalar(snapshot, signal)`** — new multiplier: 0.80× recession (yield curve < 0 for 60+ days); 1.10× expansion (curve ≥ 1.5 and signal in `_CYCLICAL_SIGNALS`); 1.10× copper-gold positive and cyclical; 0.90× USD strong; 1.05× PMI expanding and cyclical. Clamped [0.70, 1.25].
- **`core/deps.py`** — `get_combined_macro_flags` wired into `TradingDeps` and `production()`.
- **`main._build_data_bundle()`** — calls `get_combined_macro_flags()` and injects result into every snapshot before candidate scoring.
- **`main._execute_buy_phase()`** — `macro_scalar` multiplied into the notional chain; logged when ≠ 1.0.
- **`backtest/engine._fetch_macro_flags_for_backtest()`** — downloads ETF price history and FRED series to reproduce historical macro flags per trading day; passed to `_entry_signal()` in both simulation functions.
- **Tests:** 42 new tests. 4,161 tests total. 100% coverage on all changed lines.

---

### 1.95b — June 2026 — Batch 3 calendar/seasonal signals: turn_of_month, opex, halloween, quarter-end, tax_loss_reversal, pre_holiday

Adds six calendar- and seasonality-driven signals and position-sizing adjustments that exploit well-documented calendar effects without requiring any external data feed.

- **`risk/macro_calendar.get_seasonal_context()`** — new function returning six boolean flags: `turn_of_month` (±2 trading days of month-end), `opex_week` (Mon–Fri of third-Friday week), `post_opex` (Mon–Tue after OPEX), `halloween_bullish` (Nov–Apr), `quarter_end_dressing` (last 7 days of Mar/Jun/Sep/Dec), `pre_holiday` (next weekday is a NYSE holiday). Added supporting helpers: `_third_friday()`, `_next_weekday()` (weekend-skip only), `_next_trading_day()` (weekend + holiday skip), and `NYSE_HOLIDAYS` frozenset (2026–2028).
- **`risk/position_sizer.seasonal_scalar(signal, check_date)`** — new sizing multiplier: halloween bullish +10% / bearish −10%; post-OPEX +10%; turn-of-month +5%; quarter-end dressing +10% for momentum/bb_squeeze/trend_pullback; pre-holiday +5%; OPEX week −30% for gap_and_go/momentum. Scalars stack multiplicatively, clamped to [0.70, 1.25]. `_OPEX_WEEK_DAMPENED` frozenset exported from `signals/evaluator.py`.
- **`signals/evaluator.tax_loss_reversal`** — new long signal (priority 25): fires in January when `price_vs_52w_high_pct < −30%` AND `ema9_above_ema21=True`. Catches beaten-down stocks whose tax-loss selling pressure reverses at year-start. `calendar_month` field injected into snapshots at source.
- **`data/market_data.summarise_for_ai()`** — injects `calendar_month: date.today().month` into live snapshots.
- **`backtest/engine._entry_signal()`** — `calendar_month: int` parameter; passed as `int(prev_date_str[5:7])` at both simulation call sites.
- **`main._execute_buy_phase()`** — `_seasonal_scalar = seasonal_scalar(key_signal)` multiplied into the notional chain; logged when ≠ 1.0.
- **Tests:** 59 new tests. 100% coverage on all changed lines.

---

### 1.95 — June 2026 — Batch 2 signals: spread_proxy_gate, breadth_thrust, vol_of_vol position-sizing

Adds one new long signal (`breadth_thrust`), a per-stock execution-cost gate (`spread_proxy_gate`), and a VIX volatility-of-volatility position-sizing multiplier.

- **`signals/evaluator.py`** — `breadth_thrust` signal at priority 24: fires when Zweig breadth-thrust flag is set, EMA9 > EMA21, and regime is not STRESS. Blocked in `_BEAR_DAY_BLOCKED` and `_HIGH_VOL_BLOCKED`. New `_SPREAD_PROXY_GATED` frozenset (`gap_and_go`, `mean_reversion`, `range_reversion`, `candle_exhaustion`, `orb_breakout`, `vwap_reclaim`, `intraday_momentum`) — dynamically merged into `blocked` when `spread_proxy_20d > 0.5%`. Parameters: `spread_proxy_max=0.005`, `bt_min_symbols=50`.
- **`backtest/engine._compute_indicators()`** — `spread_proxy_20d`: 20-day rolling mean of (High−Low)/midpoint. `_compute_breadth_thrust_by_date()`: converts breadth series into per-date Zweig thrust booleans using `is_breadth_thrust()`. Both wired into `_entry_signal()` via `_run_simulation()` and `_run_combined_simulation()`. `run_backtest()` fetches and computes breadth-thrust map.
- **`data/market_data.fetch_stock_data()`** — `spread_proxy_20d` column. `summarise_for_ai()` exposes it. `get_market_snapshots()` injects `breadth_thrust` and `breadth_symbols_counted` via `get_breadth_snapshot(price_data=live_bulk)` (live pipeline only).
- **`data/market_regime.RegimeFeatures`** — `vol_of_vol: float | None`: 10-day std of daily VIX changes. Computed when VIX has ≥11 bars. Exposed in `to_dict()` as `"vol_of_vol"`.
- **`risk/position_sizer.vol_of_vol_scalar()`** — returns 0.7 when VoV > 3.5, 1.2 when VoV < 1.0, else 1.0. Constants: `_VOV_REDUCE_THRESHOLD=3.5`, `_VOV_BOOST_THRESHOLD=1.0`.
- **`main._execute_buy_phase()`** — `_vov_scalar = vol_of_vol_scalar(mc.regime.get("vol_of_vol"))` multiplied into the notional chain. Log message when scalar ≠ 1.0.
- **Tests:** 45 new tests. 100% coverage on all changed lines.

---

### 1.94 — June 2026 — Batch 1 OHLCV technical signals: golden_cross, candle_exhaustion, obv_divergence, obv_acceleration, volume_climax_reversal

Adds five new long-side signals and one short-side signal (death_cross) derived purely from OHLCV data, with full backtest and live pipeline integration.

- **`signals/evaluator.py`** — 5 new long signals (`golden_cross`, `candle_exhaustion`, `obv_divergence`, `obv_acceleration`, `volume_climax_reversal`) at priorities 19–23; `death_cross` short signal at priority 12. Regime blocking: `candle_exhaustion`, `obv_divergence`, `obv_acceleration` blocked in `_BEAR_DAY_BLOCKED`; `candle_exhaustion` also blocked in `_HIGH_VOL_BLOCKED`. Four short-side variants (`candle_exhaustion_short`, `obv_divergence_short`, `obv_acceleration_short`, `volume_climax_reversal_short`) added to `SHORT_GLOBALLY_DISABLED` pending backtest validation. Batch 1 params added to both `DEFAULT_SIGNAL_PARAMS` and `DEFAULT_SHORT_SIGNAL_PARAMS`.
- **`backtest/engine._compute_indicators()`** — 13 new OHLCV indicator columns: `golden_cross`, `death_cross`, `obv`, `obv_5d_slope`, `obv_20d_slope`, `obv_divergence_bull`, `obv_divergence_bear`, `obv_accelerating_up`, `obv_accelerating_down`, `near_20d_low`, `near_20d_high`, candle patterns (`hammer`, `bullish_engulf`, `shooting_star`, `bearish_engulf`), `high_vol_streak`. `_row_to_snapshot()` maps all 13 to the snapshot dict consumed by the evaluator.
- **`data/market_data.fetch_stock_data()`** — same 13 indicator columns computed before `df.tail(days)` return. `summarise_for_ai()` exposes all 13 as typed fields in the scanner snapshot dict.
- **Tests:** 63 new tests. 100% coverage on all changed lines.

---

### 1.93 — June 2026 — Standalone short book in bear regimes

Removes the hedge-only restriction on short entries so the bot can run a directional short book when no long positions are held during bear market regimes.

- **`main._STANDALONE_SHORT_REGIMES`** — new module-level constant: `{DEFENSIVE_DOWNTREND, HIGH_VOL_DOWNTREND, STRESS_RISK_OFF, CREDIT_STRESS}`. Shorts are allowed without longs only in these regimes.
- **`main._execute_shorts()`** — the `long_notional == 0` early-return is now regime-conditional: non-bear regimes still skip (hedge-only); bear regimes enter standalone mode with the short book capped at `MAX_SHORT_STANDALONE_RATIO × portfolio_value` (default 30%) instead of against long notional. Log message distinguishes `standalone` vs `hedge` mode. Per-order cap check updated accordingly.
- **`config.MAX_SHORT_STANDALONE_RATIO`** — new config knob, default 0.3, env-overridable.
- **Tests:** 4 new / 1 renamed test in `test_main.py`. 100% coverage on changed lines.

---

### 1.92 — June 2026 — Regime model v2: historical breadth series wired into backtest

Closes the live/backtest asymmetry introduced in v1.91: the backtest now computes a true historical % above 50d SMA breadth series for the regime map rather than passing `None`.

- **`backtest/engine._fetch_breadth_series_for_backtest()`** — downloads `STOCK_UNIVERSE` price history from 100 calendar days before the backtest start (providing the 50-bar SMA warmup). Vectorised rolling SMA50 computation over all symbols; excludes dates with fewer than 10 valid readings. Returns `None` gracefully so existing tests are unaffected (mock universe has 3 symbols < threshold).
- **`backtest/engine._build_regime_map()`** — now passes `breadth_series` alongside `hyg_lqd_series` and `t10y2y_series` to `compute_regime_series`. All three macro inputs are consistent between live and backtest.
- **Tests:** 7 new tests in `test_backtest.py` (643 total). 100% coverage on `backtest/engine.py`.

---

### 1.91 — June 2026 — Regime model v2 phase 2: macro inputs wired through live pipeline and backtest

Completes phase 2 of the v2 regime classifier by feeding the three new macro series (HYG/LQD credit spread, breadth % above 50d SMA, T10Y2Y yield curve) through every code path that calls `get_market_regime` or `_build_regime_map`.

- **`data/market_regime.py`** — `fetch_hyg_lqd_history()` and `fetch_t10y2y_series()` are new public functions. `fetch_spy_vix_history()` now also downloads and caches the HYG/LQD ratio. `_load_cache`/`_save_cache` extended to a 5-tuple (`spy`, `vix`, `vix9d`, `hyg_lqd`, `date`) with backward-compatible `.get()` for old pickle files.
- **`execution/stock_scanner.get_market_regime()`** — now fetches HYG/LQD, T10Y2Y, and breadth snapshot on every call and passes them to `_compute_regime`, enabling CREDIT_STRESS and LATE_CYCLE_BULL classification in live trading.
- **`backtest/engine._build_regime_map()`** — `_fetch_hyg_lqd_for_backtest()` downloads HYG/LQD history for the backtest window; `fetch_t10y2y_series()` provides FRED yield-curve data; both are forwarded to `compute_regime_series` so backtest regime maps reflect the same 9-state logic as live trading.
- **Tests:** 14 new tests in `test_market_regime.py`, 1 new in `test_stock_scanner.py`, 5 new in `test_backtest.py`. 100% coverage on all changed files.

---

### 1.90 — June 2026 — Regime model v2: CREDIT_STRESS, LATE_CYCLE_BULL, RECOVERY states

Extends the 6-state regime classifier to 9 states by integrating three new macro inputs: HYG/LQD credit spread 10-day ROC, breadth % of stocks above their 50-day SMA, and T10Y2Y yield curve spread.

- **`MarketRegime` enum** — 3 new members: `CREDIT_STRESS`, `LATE_CYCLE_BULL`, `RECOVERY`.
- **`RegimeFeatures`** — 3 new optional fields (`credit_spread_roc`, `breadth_pct_above_sma50`, `t10y2y`) defaulting to `None` for backward compatibility.
- **`RegimeThresholds`** — 5 new defaults: `credit_stress_roc_min=-2.0`, `t10y2y_inversion_threshold=0.0`, `breadth_divergence_max=0.50`, `recovery_spy_5d_min=0.5`, `recovery_drawdown_max=-5.0`.
- **`resolve_regime()` priority chain** — STRESS_RISK_OFF → HIGH_VOL_DOWNTREND → DEFENSIVE_DOWNTREND → **CREDIT_STRESS** → (LATE_CYCLE_BULL or BULL_TREND) → **RECOVERY** → NEUTRAL_CHOP. CREDIT_STRESS fires when HYG/LQD 10d ROC ≤ −2%; LATE_CYCLE_BULL fires when bull price conditions hold but yield curve is inverted or breadth is narrow (<50%); RECOVERY fires when SPY 5d ≥ 0.5% but drawdown ≤ −5%.
- **Hysteresis** — STRESS_RISK_OFF confirms immediately; all other new states require 2 consecutive bars to confirm.
- **`signals/evaluator.py`** — `REGIME_BLOCKED` extended: CREDIT_STRESS inherits HIGH_VOL blocking; LATE_CYCLE_BULL inherits NEUTRAL_CHOP blocking; RECOVERY blocks `{breakout_52w, momentum, gap_and_go, macd_crossover, inside_day_breakout}` while allowing `mean_reversion`, `trend_pullback`, `iv_compression`. `SHORT_ALLOWED_REGIMES` gains `CREDIT_STRESS`.
- **Tests:** 60+ new tests across `test_market_regime.py` and `test_risk_config.py`. **146 tests in test_market_regime.py, 100% coverage on changed files.**

---

### 1.89 — June 2026 — rs_leader and momentum_12_1 globally disabled

Walk-forward backtest evidence confirms no edge for either signal in any market regime.

- **`rs_leader` → `GLOBALLY_DISABLED`** — standalone Sharpe −0.93 over 9-year walk-forward; exhaustive param sweep (5d excess threshold 2–10%, 10d threshold 3–12%) yields best-case Sharpe 0.15 at tightest thresholds (too few trades). Removed from `_BEAR_DAY_BLOCKED` and `_HIGH_VOL_BLOCKED`; per-regime blocking replaced by global freeze.
- **`momentum_12_1` → `GLOBALLY_DISABLED`** — WR 48%, avg −0.2%, n≥97 in every tested regime (BULL_TREND, HIGH_VOL, NEUTRAL_CHOP); no combination of ADX, pullback, or threshold parameters recovers a positive Sharpe. Removed from `_BULL_TREND_BLOCKED` and `_DEFENSIVE_BLOCKED`.
- **Tests:** 10 tests updated across `test_backtest.py` (7), `test_stock_scanner.py` (2), `test_risk_config.py` (1). **3856 passing, 100% coverage on changed files.**

---

### 1.88 — June 2026 — TradingDeps dependency injection refactor

Replaced module-level globals with a single injectable `TradingDeps` dataclass, eliminating 15+ `unittest.mock.patch` call sites and making `_run_inner` fully testable without import-level side effects.

- **`core/deps.py`** — new `TradingDeps` dataclass with 23 fields (trader, stock_scanner, market_data, ai_analyst, position_sizer, validate_ai_response, + 9 new: short_risk, sector_momentum, options_data, get_macro_snapshot, get_sentiment_snapshot, get_short_universe, scan_short_universe, short_interest, edgar_client). `TradingDeps.production()` constructs the live instance; `TradingDeps.testing()` is replaced by `make_test_deps()` in `conftest.py`.
- **`main.py`** — `_run_inner(deps: TradingDeps | None = None)` calls `TradingDeps.production()` when `deps is None`; all ~15 pipeline helpers accept `deps: TradingDeps | None = None` with the same lazy-init pattern. All module-level globals removed from hot paths.
- **Tests:** `TestMaxOrdersPerRun._run_buys` rewritten to use `make_test_deps` with `MacroSnapshot` / `SentimentSnapshot` dataclass objects (not dicts). `TestRunInnerMQSBoost` added to cover `main.py:1910` MQS boost logger. Dead `_shadow_run(overrides=...)` parameter removed. **3856 passing, 100% coverage.**

---

### 1.87 — June 2026 — 100% coverage enforcement; VSCode extension excluded

- **`pytest.ini`** — `--cov-fail-under=100` added; VSCode extension path excluded from measurement via `omit` to prevent false coverage shortfalls.

---

### 1.86 — June 2026 — Five position-sizing and exit-quality features

Five independent improvements to position sizing accuracy and exit timing, each backed by 100% test coverage.

- **Amihud illiquidity gate** (`data/market_data.py`, `risk/position_sizer.py`) — cross-sectional illiquidity ranking using Amihud (2002): `mean(|daily_return| / dollar_volume)` over 20 bars per symbol. When ≥10 symbols have non-zero ratios, the 90th-percentile threshold is computed; symbols above it are flagged `amihud_illiquid=True` in the snapshot. `position_sizer.amihud_size_scalar` reduces position size 50% for flagged symbols.
- **GARCH(1,1) volatility forecast** (`risk/exit_optimiser.py`) — `compute_garch_vol_scalar(symbol)` downloads 90 days of daily closes, fits a GARCH(1,1) model (`arch` library), and compares the one-step-ahead forecast volatility to the 60-day historical std dev. When `forecast_vol / hist_vol > 1.5`, the size scalar is `hist_vol / forecast_vol` (floored at 0.5). Returns 1.0 gracefully on any data or model failure.
- **Momentum quality score** (`risk/position_sizer.py`) — `momentum_quality_score(candidate)` sums three binary components: RS percentile rank ≥ 60, `pead_candidate` flag, and profitability composite (ROE > 0 AND profit margin > 0). Score 3 triggers `mqr_size_multiplier` → 1.5× size boost.
- **Sector momentum rank gate** (`data/sector_momentum.py`, `main.py`) — ranks all 11 SPDR ETFs by 20-day return each session. Only symbols in top-4-ranked sectors are eligible for long entry; shorts restricted to bottom-3 sectors. Results cached 24 hours to `logs/sector_momentum_cache.json`. Fail-open: empty ranks allow all trades.
- **Signal invalidation exit** (`risk/exit_optimiser.py`, `execution/trader.py`, `utils/db.py`) — DB migration 9 adds `entry_snapshot TEXT` column to `positions`. At buy time, `record_buy(entry_snapshot=candidate)` stores the full snapshot as JSON. At midday, `signal_invalidated(symbol, meta, pos)` re-evaluates technical signals against fresh market data; if the qualifying signal(s) from entry are no longer active (and minimum 2-day hold is met), the position is closed.
- **Tests:** 107 new tests across 5 test files. **3854 passing, 100% coverage on all changed files.**

---

### 1.85 — June 2026 — insider_buying three-tier conviction filter

Raises the bar for the `insider_buying` signal by introducing a three-tier firing hierarchy, eliminating weak cluster signals that lack supporting conviction.

- **Three-tier firing logic** — `activist_filing` always fires. `insider_strong_cluster` (≥3 distinct insiders buying open-market within 5 calendar days) always fires. Standard cluster (≥2 insiders / 10 days) fires only when `insider_comp_ratio ≥ 0.02` OR `insider_large_buy` (single transaction > $100k). Bare cluster alone no longer fires.
- **`data/insider_feed.py`** — `_fetch_one` now computes `insider_strong_cluster` and `insider_comp_ratio` (max purchase notional / annual comp via `get_exec_compensation` + `match_compensation` from `data/proxy_comp.py`).
- **Tests:** 18 new tests. **3747 passing.**

---

### 1.84 — June 2026 — DEF 14A executive compensation fetcher

- **`data/proxy_comp.py`** — parses the SEC EDGAR Summary Compensation Table from annual proxy statements (DEF 14A). Locates the most recent filing via the submissions JSON, downloads the primary HTML document, extracts name→total USD pairs using BeautifulSoup, and caches for 90 days. Public API: `get_exec_compensation(cik)` and `match_compensation(reporter, comp_map)` (token Jaccard fuzzy-match for Form 4 name strings). Required by the `insider_buying` signal improvement for purchase-size-to-compensation scaling.
- **Tests:** 49 new tests in `tests/test_proxy_comp.py`; 100% coverage on `data/proxy_comp.py`. **3761 passing.**

---

### 1.83 — June 2026 — pead tightened: 10% EPS threshold, 7-day entry window

- **EPS beat threshold raised 5% → 10%** — `_PEAD_MIN_SURPRISE` in `backtest/historical_fundamentals.py` and `_MIN_SURPRISE_PCT` in `data/earnings_surprise.py`; weaker beats have less predictive value for the drift effect.
- **Entry window reduced 30 → 7 days** — `pead_active_on_date` default `lookback_days` and `_PEAD_WINDOW_DAYS`; constrains entries to the initial drift period (≈5 trading days) where the anomaly is strongest.
- **Pre-existing coverage gap closed** — `earnings_miss_active_on_date` and `recent_earnings_date` had no tests; 13 new tests cover both functions fully.
- **Tests:** 13 new tests in `test_historical_fundamentals.py`; 100% coverage on both changed modules. **3712 passing.**

---

### 1.82 — June 2026 — iv_compression loosened + momentum_12_1 pullback filter

- **`iv_compression` threshold loosened** — `ivc_hv_rank_max` raised from 0.10 → 0.15; moderate vol compression is still predictive (extreme-only threshold was too restrictive).
- **`momentum_12_1` pullback filter** — new `mom12_1_pullback_ret5d_max: 1.0` param; signal now requires 1-week return ≤ 1% to ensure we buy on a retracement in a strong trend, not chase an already-extended move.
- **`iv_compression_short` added (disabled)** — mirror of the long setup. Added to `SHORT_GLOBALLY_DISABLED` pending initial backtest validation.
- **Tests:** 24 new tests; 100% coverage on `signals/evaluator.py`. **3699 passing.**

---

### 1.81 — June 2026 — pairs trading infrastructure + FinBERT NLP pipeline

- **`data/pairs.py`** — sector-grouped cointegration engine using Engle-Granger (statsmodels `coint`, p<0.05), OLS hedge ratio, spread z-score computation, and a 7-day disk cache. Public API: `get_cointegrated_pairs`, `compute_zscore`, `refresh_pairs`. Required by `sector_pair_mean_reversion`.
- **`data/finbert.py`** — lazy-loaded FinBERT wrapper (ProsusAI/finbert via HuggingFace `transformers.pipeline`). Normalises three-label scores (positive/negative/neutral), truncates to 2000 chars, degrades gracefully when `transformers`/`torch` are absent. Public API: `is_available`, `classify_text`, `classify_texts`.
- **`requirements.txt`** updated: `statsmodels==0.14.6` (required), `transformers>=4.40.0` and `torch>=2.0.0` (optional, for FinBERT).
- **Tests:** 50 new tests. **3675 passing, 100% coverage on new files.**

---

### 1.80 — June 2026 — parabolic_exhaustion and overbought_downtrend disabled

Backward elimination analysis identified two short signals as net destroyers of Sharpe. Both added to `SHORT_GLOBALLY_DISABLED`.

- **`parabolic_exhaustion` disabled** — ΔSharpe +0.570 when removed; contributed -99.5% return over the 9-year period.
- **`overbought_downtrend` disabled** — ΔSharpe +0.060 drag; only `earnings_gap_down` survives short backward elimination (Sharpe 0.720, 152 trades, +227.1% return).
- **Tests:** 7 tests updated. **3625 passing, 100% coverage on changed files.**

---

### 1.79 — June 2026 — rs_leader live-system bug fixed

`rs_leader` was firing 0 times in live runs despite meeting all technical conditions. Root cause: `prefilter_candidates` called `evaluate_signals` without passing `spy_ret_5d` / `spy_ret_10d`, so the signal's first guard (`spy_ret_5d is not None`) was always `False`.

- **Bug fixed** — `prefilter_candidates` in `execution/stock_scanner.py` now accepts and passes `spy_ret_5d: float | None = None` and `spy_ret_10d: float | None = None`.
- **Caller updated** — `_build_data_bundle` in `main.py` now calls `market_data.get_spy_5d_return()` and `market_data.get_spy_10d_return()` and forwards results to `prefilter_candidates`. The backtest engine already passed these correctly; this was live-only.
- **Tests:** 9 new tests. **3625 passing, 100% coverage on changed files.**

---

### 1.78 — June 2026 — exit optimiser and position-sizer wired into live pipeline

Seven signal-based controls built in v1.74–1.77 were unit-tested but not yet called from the live pipeline. This release completes the wiring end-to-end.

- **RS-decay exit** — fires when a position's RS percentile rank drops >25 points from entry. Only applies to RS-momentum signals. Entry RS rank stored via new `rs_rank_pct` column in the `positions` table (DB migration 8).
- **Adverse-volume exit** — fires when two consecutive days show vol_ratio ≥ 2.5 with return ≤ −1.5%.
- **Profit-acceleration exit** — fires for mean-reversion/range-reversion signals only; returns `full_exit`, `partial_exit`, or `hold` based on unrealised gain and days held.
- **Regime-change exit** — when regime is `DEFENSIVE_DOWNTREND` or `BEAR_MARKET`: positions held <2 days are exited immediately; positions held ≥3 days receive an advisory log.
- **ATR-based position sizing** (`position_sizer.atr_position_size`) — in the buy loop, `compute_atr_pct` is called for each candidate; when non-None the ATR-derived notional replaces `risk_budget_size` as the base.
- **Signal Sharpe multiplier** (`position_sizer.get_signal_size_multiplier`) — applied to base notional at buy time; scales down low-Sharpe signals and up high-Sharpe ones.
- **Co-firing boost** (`position_sizer.cofiring_boost`) — returns 1.5× when ≥2 signals fire simultaneously.
- **Tests:** 19 new tests. **3593 passing, 100% coverage on changed files.**

---

### 1.77 — June 2026 — infra wiring: macro, options, sentiment, EDGAR data into live pipeline

Four data modules built in v1.74 were fully fetched at prefetch time but never read by any decision logic. This release wires them end-to-end.

- **`data/edgar_client.py` — `get_edgar_signals_batch()`** — batch cache-first fetch; enriches signals before the prefilter pass.
- **`models.py` — `MarketContext.cross_asset_macro` and `sentiment_snapshot`** — two new optional dict fields.
- **`main.py` — `_fetch_market_context()`** — adds `get_macro_snapshot()` and `get_sentiment_snapshot()` to the thread pool.
- **`main.py` — `_build_data_bundle()`** — EDGAR signals (pre-filter) + Options IV (post-filter, 5–20 symbols only).
- **`signals/evaluator.py`** — signal extensions: `insider_buying` fires on `activist_filing=True`; `pead` fires on `guidance_positive=True`; `iv_compression` fires when `iv_cheap=True`; new short signals `guidance_downgrade` (priority 9) and `secondary_offering_short` (priority 10).
- **`scripts/run_scheduler.py`** — `_prefetch()` now calls `prefetch_edgar_data()`, `get_macro_snapshot()`, and `get_fear_greed_composite()`.
- **Tests:** 43 new tests. **3574 passing, 100% coverage on changed files.**

---

### 1.76 — June 2026 — short universe capped to static list + open-buys guard fixes

Four correctness and performance fixes uncovered during live runs on 2026-06-05.

- **`execution/short_universe.py` — intersect Alpaca ETB with `STATIC_SHORT_UNIVERSE`.** `get_short_universe` was returning all ~4947 Alpaca easy-to-borrow symbols; `yf.download(threads=False)` then spent ~14 minutes downloading them. The scan universe is now capped: Alpaca's ETB list is used only to verify which `STATIC_SHORT_UNIVERSE` symbols (~212) are borrowable today.
- **`execution/short_universe.py` — `threads=False` in `yf.download`** — prevents "can't start new thread" errors when called after the parallel insider fetch has many threads in flight.
- **`risk/macro_calendar.py` — NFP removed from high-risk block** — Non-Farm Payrolls releases at 08:30 ET, before market open; by our 10:00 ET buy window the reaction is absorbed. Treating NFP as high-risk was incorrectly blocking all buys on the first Friday of each month.
- **`main.py` + `utils/audit_log.py` — open-buys guard correctness fixes** — Two compounding bugs: (1) `log_open_buys_locked` was written before `skip_buys` was evaluated; (2) `has_open_buys_run_today` matched rows by `ts LIKE '2026-06-05%'` only, not by payload date. Both fixed.
- **Tests updated**: `test_macro_calendar.py` — NFP assertion flipped. `test_short_side.py` — mock symbols updated to `STATIC_SHORT_UNIVERSE` members. **3531 passing, 100% coverage on changed files.**

---

### 1.75 — June 2026 — parallel insider fetch + Alpaca short-universe retry

- **`data/insider_feed.py` — parallel EDGAR fetch** — extracted `_fetch_one(sym, cik_map, ...)` as a pure per-symbol worker; `_live_fetch` now submits all symbols to `ThreadPoolExecutor(max_workers=10)`. A global `_edgar_sleep()` rate-limiter (`threading.Lock` + `_last_req_time`) serialises sleeps across threads to stay inside EDGAR's 10 req/s ceiling while allowing HTTP calls to overlap. Expected improvement: ~43 min → 2–5 min on cache-miss symbols.
- **`execution/short_universe.py` — retry on Alpaca `get_all_assets` failure** — `get_short_universe` now retries up to 2 times with 3-second backoff before falling back to the static list. `RemoteDisconnected` errors on `client.get_all_assets()` are almost always transient.
- **5 new tests; 3531 passing, 100% coverage on changed files.**

---

### 1.74 — June 2026 — macro, options, sentiment, and EDGAR data infrastructure

Four new data modules that provide cross-asset and corporate-event signals as inputs for the signal evaluator. All modules use daily caching, degrade gracefully on network failure, and are wired to the 07:00 ET pre-market prefetch job.

- **`data/macro_data.py`** — downloads HYG, LQD, IEF, TLT, CPER, GLD, UUP, SPY daily via yfinance. `MacroSnapshot` dataclass exposed via `get_macro_snapshot()` with daily cache at `logs/macro_data_cache.json`.
- **`data/options_data.py`** — fetches yfinance option chains for the expiry closest to 30 DTE. `OptionsSnapshot` per symbol, cached daily. `get_options_batch()` fetches all symbols in parallel via `ThreadPoolExecutor`.
- **`data/sentiment_client.py`** — three independent sentiment feeds: AAII (weekly survey), Fear & Greed composite (5-component score), Google Trends (pytrends spike/decline detection per symbol). `get_sentiment_snapshot()` combines AAII and F&G into `contrarian_long_signal` / `contrarian_short_signal` booleans.
- **`data/edgar_client.py`** — SEC EDGAR REST API (no auth required). Three filing types: 8-K items 2.02/7.01 (guidance sentiment), SC 13D/G (activist investor detection), 424B4/S-3/S-1 (secondary offering supply shock). `prefetch_edgar_data()` warms all universe symbols at 07:00 ET.
- **243 new tests** across `test_edgar_client.py`, `test_macro_data.py`, `test_options_data.py`, `test_sentiment_client.py`. 100% coverage on all four modules. **3526 passing.**

---

### 1.72 — June 2026 — AV sentiment same-day cache + parallel market context fetch

- **`data/av_sentiment.py` — same-day cache** — `get_av_sentiment` reads `logs/av_sentiment_cache.json` first; only live-fetches cache misses. `prefetch_av_sentiment` warms all ~509 symbols at 07:00 ET. Estimated saving: ~65 s per window eliminated.
- **`_fetch_market_context()` — parallelized with `ThreadPoolExecutor(5)`** — The five I/O calls were sequential; now run concurrently. Wall time drops from `sum(latencies)` to `max(latency)`, estimated 10–20 s saving.
- **36 new/updated tests**: 31 in `test_av_sentiment.py`, 5 in `TestFetchMarketContext`. **3276 passing.**

---

### 1.71 — June 2026 — startup cache warm on late scheduler restart

Fixes the cold-cache problem when the scheduler is killed and restarted after the 07:00 ET prefetch window has passed.

- **`_startup_prefetch()`** (new function in `scripts/run_scheduler.py`). Fires `_prefetch()` in a background daemon thread immediately when the scheduler starts. No-op on weekends and instantly exits per-symbol if the same-day cache is already warm.
- Root cause fixed: on 2026-06-03 the scheduler process (PID 34349) was restarted at 13:51 BST, after the 12:00 BST (07:00 ET) prefetch window. All four caches were empty; `open_sells` had to fetch insider data live, taking ~80 minutes. After v1.71 any restart — at any time of day — triggers an immediate background warm.
- **3256 passing** (3 new tests), 100% coverage on changed files.

---

### 1.70 — June 2026 — same-day cache for earnings and short interest data

Extends the pre-market prefetch introduced in v1.66/v1.69 to cover all remaining static signals, eliminating ~64 seconds of sequential yfinance requests from every intraday trading window.

- **`data/earnings_surprise.py` — same-day cache + shared single-fetch** — fetches `yf.Ticker(sym).earnings_dates` once per symbol and computes both the PEAD beat and negative-PEAD miss results in a single pass. Cache stored at `logs/earnings_cache.json` keyed by ET business date. `None` sentinels mark ETFs and no-data symbols so they are not re-queried within the same day.
- **`data/short_interest.py` — same-day cache** — `logs/short_interest_cache.json` with `None` sentinels for below-threshold symbols.
- **`prefetch_earnings_data` / `prefetch_short_interest`** (new public functions). Called from the 07:00 ET pre-market prefetch job.
- **Latency savings**: earnings surprise + earnings miss reduced from ~64 s to ~0 s; short interest from ~32 s to ~0 s.
- **61 new/updated tests**. **3253 passing, 100% coverage.**

---

### 1.69 — June 2026 — same-day cache for SEC EDGAR insider activity

Eliminates the 19-minute `open_sells` block caused by sequential EDGAR HTTP requests for 641 symbols.

- **`data/insider_feed.py` — same-day cache** — New `get_insider_activity` checks `logs/insider_cache.json` first and only calls EDGAR on cache miss. `None` sentinels prevent repeat requests within the same calendar day.
- **`prefetch_insider_activity`** (new public function). Called from the 07:00 ET prefetch job.
- **Latency saved**: `open_sells` EDGAR block reduced from ~19 minutes to <1 second.
- **33 new/updated tests**. **~3220 passing, 100% coverage.**

---

### 1.66 — June 2026 — same-day market data cache + pre-market prefetch

Eliminates redundant data downloads across the four daily trading windows.

- **`_bulk_download` cache** (`data/market_data.py`). First call each ET calendar day downloads all symbols and serialises to `logs/market_data_YYYY-MM-DD.pkl`. Subsequent calls load from disk and only download symbols absent from cache.
- **`prefetch_market_data`** (new function). Warms the cache with no trading logic.
- **07:00 ET prefetch trigger** (`scripts/run_scheduler.py`). Scheduler fires a silent prefetch 2.5 hours before open_sells. Expected run-time drop: ~94 min → ~10 min per intraday window.
- **16 new tests**. **~2971 passing, 100% coverage.**

---

### 1.65 — June 2026 — expand long universe from 52 to 509 symbols (S&P 500 + ETFs)

`STOCK_UNIVERSE` in `config.py` replaced with the full S&P 500 current constituents (503 stocks, sourced from Wikipedia 2026-06-02) plus 6 broad-market and sector ETFs. Total: **509 symbols** (was 52).

- Dual-class share pairs (GOOG/GOOGL, FOXA/FOX, NWS/NWSA) included.
- `BRK.B` and `BF.B` use dot notation; `execution/universe.py` normalises to hyphens for Alpaca at runtime.
- Short universe (`STATIC_SHORT_UNIVERSE`, 212 symbols) unchanged.
- **~2950 passing, 100% coverage.**

---

### 1.64 — June 2026 — comprehensive signal testing suite (8 new modes)

Eight new analysis modes covering every angle of signal validation, all wired to CLI flags:

- **`run_signal_isolation`** (`--signal-isolation`) — runs each long signal in complete isolation.
- **`run_short_ablation`** (`--short-ablation`) — measures ΔSharpe when each short signal is removed.
- **`run_short_backward_elimination`** (`--short-backward-elimination`) — greedy iterative removal of short signals.
- **`run_short_regime_analysis`** (`--short-regime-analysis`) — stratifies COVER trades by `entry_regime` and `days_held`.
- **`run_monte_carlo`** (`--monte-carlo`) — two-tier statistical test: portfolio-level Sharpe permutation test + per-signal bootstrap 95% CI.
- **`run_multi_fold_walk_forward`** (`--multi-fold`) — non-overlapping windows of three fold sizes (63 / 126 / 252 trading days).
- **`run_crisis_slices`** (`--crisis-slices`) — runs the long simulation independently across GFC 2008–09, COVID 2020, and 2022 rate-hike year.
- **`run_co_firing_analysis`** (`--co-firing`) — analyses co-firing rates for all signal pairs above the 20% overlap threshold.
- **59 new tests**; **~2950 passing, 100% coverage.**

---

### 1.63 — June 2026 — redesign overbought_downtrend + parabolic_exhaustion; disable faded_earnings_gap_up

- **`faded_earnings_gap_up` disabled** — Mean Sharpe −0.201, only 2/9 profitable folds; 2020–2021 fold produced −35% return. Structural flaw: gap-ups that close weak still continue higher in strong FOMO markets.
- **`overbought_downtrend` redesigned** — Changed trigger from `price < sma50` → `price < sma200`. Split single RSI threshold into separate entry/exit levels: `ordt_rsi_entry` (65.0) and `ordt_rsi_exit` (60.0). Requires a meaningful 5+ point RSI move.
- **`parabolic_exhaustion` redesigned** — Root cause of 0 trades was regime gating: moved parabolic_exhaustion to a dedicated path evaluated before the regime gate so it fires in BULL_TREND and NEUTRAL_CHOP.
- **0 net new tests; ~2891 passing, 100% coverage.**

---

### 1.62 — June 2026 — three new short signals: overbought_downtrend, parabolic_exhaustion, faded_earnings_gap_up

- **`overbought_downtrend`** — fires when a stock is below its 50-day SMA and RSI crosses back below `ordt_rsi_cross` (default 62.0) after bouncing above it — fading the relief rally.
- **`parabolic_exhaustion`** — fires when a stock is up ≥ 80% in 60 trading days, RSI ≥ 72.0, and volume drying up (vol_ratio ≤ 0.9).
- **`faded_earnings_gap_up`** — fires the session after a stock gaps up ≥ 5% on earnings but closes in the bottom 30% of the day's range on volume ≥ 1.5×. T+1 entry.
- **36 new tests**; **~2891 passing, 100% coverage.**

---

### 1.61 — May 2026 — earnings_gap_down: tighten params to egd_gap_pct_max=−7, egd_vol_min=2.5

- **`DEFAULT_SHORT_SIGNAL_PARAMS` updated** — `egd_gap_pct_max`: −5.0 → −7.0; `egd_vol_min`: 1.5 → 2.5.
- Combined walk-forward at `−7, 2.5`: mean Sharpe **+0.643** across all 9 folds (vs +0.258 at defaults). **2855 passing, 100% coverage.**

---

### 1.60 — May 2026 — earnings_gap_down: same-bar entry, walk-forward fix, STATIC_SHORT_UNIVERSE

- **Same-bar gap-open entry** — previously entered T+1; now detects gap on the reaction bar itself and enters at the market open on the gap day.
- **Walk-forward earnings_history bug fixed** — `run_short_walk_forward()` never called `prefetch_earnings_history()`, causing 0 gap trades in all 11 walk-forward folds.
- **Short CLI broadened to `STATIC_SHORT_UNIVERSE` (~300 symbols)** — PEAD literature documents the effect predominantly in small/mid-cap stocks with thinner analyst coverage.
- **2855 passing, 100% coverage.**

---

### 1.59 — May 2026 — earnings_gap_down short signal (backtest-enabled)

- **`earnings_gap_down` signal (new)** — fires when a stock gaps down ≥ 5% on the first open after earnings with volume ≥ 1.5×. Path D fires before the RS-rank gates so it applies to all stocks regardless of RS tier.
- **`run_short_walk_forward()` (new function)** — walk-forward stability check for a fixed short parameter set.
- **68 new tests**; **2847 passing, 100% coverage.**

---

### 1.58 — May 2026 — Short squeeze avoidance gate (live-only)

- **`execution/short_risk.py` (new module)** — `is_squeeze_risk(symbol, snapshot, *, short_pct_float, days_to_cover, ...)` blocks short entry when: reported short interest > 20% of float, days-to-cover > 5, or 5-day price momentum > 15%. Live-only (no backtest integration).
- **13 new tests**; **2779 passing, 100% coverage.**

---

### 1.57 — May 2026 — Disable rs_deterioration; fix backtest weekend crash

- **`rs_deterioration` disabled** — walk-forward 2015–2026 showed 0/11 profitable folds, mean Sharpe −0.872, 619 trades, WR 36%, avg/trade −1.17%.
- **Weekend crash fix** — `pd.bdate_range(end=yesterday, periods=1)` returned an empty array on Saturdays and Sundays. Fixed by using `pd.Timestamp.today().normalize() - pd.offsets.BDay(1)`.
- **2765 passing, 100% coverage.**

---

### 1.56 — May 2026 — rs_deterioration signal + VIX term structure gate + Alpaca short universe

- **`rs_deterioration` signal (new)** — cross-sectional leader-to-laggard rotation signal. Fires when a stock was in top 35% of universe 10 days ago but has since fallen below median and is down > 2% over five days.
- **VIX term structure gate (new)** — `vix_term_inverted = VIX9D / VIX > 1.05`. Applied as a hard gate in `_execute_shorts()` (live) and `_short_entry_signal()` (backtest).
- **Path C (Deterioration) in `scan_short_candidates()` (new)** — third short entry path.
- **`execution/short_universe.py` (new)** — `get_short_universe(client)` queries Alpaca for all easy-to-borrow assets. Falls back to `STATIC_SHORT_UNIVERSE` (~300 curated sector-diverse names) on failure. `scan_short_universe(symbols)` downloads OHLCV, computes cross-sectional RS ranks, and returns enriched snapshot dicts.
- **28 new tests**; **2765 passing, 100% coverage.**

---

### 1.55 — May 2026 — Disable all short signals; add short walk-forward; research new signal candidates

- **All short signals permanently disabled** via `SHORT_GLOBALLY_DISABLED`: `ema_breakdown`, `winner_reversal`, `failed_breakout`, `high_vol_reversal`, `earnings_miss`. Walk-forward confirmed no edge: 1/11 profitable folds, mean Sharpe −0.201.
- **`high_short_interest` unblocked** on both scan paths — only live short signal; zero trades in practice until real short-interest data is wired.
- **`run_short_walk_forward()` (new function)** — runs `_run_short_simulation` across non-overlapping date folds with fixed short params.
- **2737 passing, 100% coverage.**

---

### 1.54 — May 2026 — Replace short signals: failed_breakout + high_vol_reversal, two-path RS architecture

- **`ema_breakdown` and `winner_reversal` permanently disabled** — net-negative across all parameter combinations.
- **`failed_breakout` signal (new)** — bull-trap pattern: stock closed above its 20-day high yesterday, failed back below it today.
- **`high_vol_reversal` signal (new)** — distribution/exhaustion bar: high volume, close in the bottom 30% of the day's range, RSI already elevated, 5-day return shows prior strength.
- **Two-path RS architecture for short entries** — Reversal path (rs_rank ≥ 65): recently-strong stocks showing exhaustion. Fundamental path (rs_rank < 25): bottom-quartile laggards with catalyst.
- **`SHORT_ALLOWED_REGIMES` exported from `signals/evaluator.py`** — fixes critical inconsistency: live scanner was allowing shorts in `BULL_TREND`/`NEUTRAL_CHOP` (the opposite of correct).
- **Regime pass-through bug fixed in `_run_short_simulation`** — the `regime` value was never passed to `_short_entry_signal`; shorts fired in all regimes during isolated backtest runs.
- **2737 passing, 100% coverage.**

---

### 1.53 — May 2026 — Phase 5: disable drag signals + short parameter sweep framework

- **`GLOBALLY_DISABLED` frozenset added to `signals/evaluator.py`** — Initial members: `rsi_divergence` and `breakout_52w`.
- **`run_short_param_sensitivity()` (new function)** — one-at-a-time parameter sweep for short signal thresholds. Exposed as `--short-param-sensitivity` CLI flag.
- **17 new tests**; **2730 passing, 100% coverage.**

---

### 1.52 — May 2026 — Phase 4: dual-track intraday live pipeline

- **`track` column added to `positions` and `trades` tables (DB migrations v6, v7)** — distinguishes `'intraday'` from `'multiday'` positions.
- **Intraday signal tagging on buy and short** — positions opened by signals in `INTRADAY_SIGNALS` are tagged `track='intraday'`.
- **`_force_cover_intraday_positions()` (new function in `main.py`)** — at the `close` pass, all intraday positions are market-sold before the regular sell phase.
- **28 new tests**; **2513 passing, 100% coverage.**

---

### 1.51h — May 2026 — Option B complete: `_run_inner` fully modularised

- **`_run_inner` refactored into a clean 12-phase pipeline.** All inline business logic extracted into typed module-level helper functions. `_run_inner` is now a sequence of named calls with no embedded logic: `_evaluate_risk_limits` → `_fetch_market_context` → `_get_position_snapshot` → `_manage_existing_positions` → `_build_data_bundle` → `_run_ai_phase` → `_execute_sell_phase` → `_execute_buy_phase` → `_execute_shorts` → `_reconcile_late_fills` → `_finalise`.
- **0 new tests** (pure refactor); **2485 passing, 100% coverage.**

---

### 1.48 — May 2026 — adopt 8 parameter wins from full stretch test

- **Full one-at-a-time parameter sweep run across all 29 `DEFAULT_SIGNAL_PARAMS`** — 145 simulations (2015–2023). 8 clear wins adopted: `bb_threshold` 0.25→0.15, `vfr_vol_min` 1.0→1.5, `bk52_pct_min` -3.0→-2.0, `gap_vol_min` 1.5→2.0, `tp_ema21_lo` -3.0→-2.0, `tp_rsi_lo` 40.0→50.0, `ivc_hv_rank_max` 0.20→0.10, `ivc_vol_min` 1.1→1.2.
- **2397 passing, 100% coverage.**

---

### 1.47 — May 2026 — full signal parameter sweep framework

- **All signal thresholds now in `DEFAULT_SIGNAL_PARAMS`** — 19 new entries across all signals. All default to their previous hardcoded values — no behaviour change.
- **`run_param_sensitivity()` (new function in `backtest/engine.py`)** — one-at-a-time parameter sweep. Exposed as `--param-sensitivity` CLI flag.
- **15 new tests**; **2397 passing, 100% coverage.**

---

### 1.46 — May 2026 — rsi_div_bb_max default tightened to 0.30

- **`rsi_div_bb_max` default changed from 1.0 → 0.30** — portfolio return +36.7% vs +31.2% baseline (+5.5pp), Sharpe 0.290 vs 0.270. Trade count 240 vs 272.
- **2382 passing, 100% coverage.**

---

### 1.45 — May 2026 — rsi_divergence parameter gates + disabled_signals bug fix

- **`rsi_divergence` signal gates now configurable via `DEFAULT_SIGNAL_PARAMS`** — Three new thresholds: `rsi_div_rsi_max`, `rsi_div_vol_min`, `rsi_div_bb_max`.
- **`--disabled-signals` wired into all entry points** — was only connected to `run_walk_forward_optimized`. Bug fixed: `run_ablation` and `run_backward_elimination` previously passed `disabled_signals` twice causing `TypeError`.
- **3 new tests**; **2382 passing, 100% coverage.**

---

### 1.44 — May 2026 — iv_compression blocked in NEUTRAL_CHOP + rsi_divergence signal

- **`iv_compression` blocked in `NEUTRAL_CHOP`** — backtest shows WR 51%, avg +0.0%, n=506 — 506 trades generating zero net alpha, well below the 0.32% round-trip cost threshold.
- **`rsi_divergence` signal (new)** — fires when price is lower than 5 days ago but RSI is recovering (bullish structural divergence). ADX < 25, RSI < 45. Priority 12.
- **19 new tests**; **2380 passing, 100% coverage.**

---

### 1.43 — May 2026 — PEAD fix, signal blocking refinements, cost sensitivity output, research-grade warnings

- **PEAD portfolio trades fixed** — `run_backtest()` lacked `use_earnings_only`, so the `--use-earnings-only` CLI flag was silently dropped, producing 0 PEAD portfolio entries despite 1,543 signal-analysis occurrences.
- **`range_reversion` blocked in `NEUTRAL_CHOP` and `DEFENSIVE_DOWNTREND`** — WR 46%, avg -0.0%, p>0.05, n=52 in NEUTRAL_CHOP; WR 30%, avg -2.1%, n=10 in DEFENSIVE_DOWNTREND.
- **`momentum_12_1` blocked in `BULL_TREND`** — WR 48%, avg -0.2%, n=97.
- **Cost sensitivity table in signal analysis output** — `_print_cost_sensitivity()` flags signals with avg < 2× round-trip cost (0.32%) as fragile.
- **Survivorship/proxy warning** — backtest page shows a persistent warning banner noting survivorship bias and rule-proxy nature.
- **10 new tests**; **2361 passing, 100% coverage.**

---

### 1.42 — May 2026 — regime blocking refinements + range_reversion signal + stop delay extension

- **`mean_reversion` blocked in `NEUTRAL_CHOP` and `STRESS_RISK_OFF`** — negative expected value in these regimes confirmed by walk-forward analysis.
- **`rs_leader` disabled** — 246 trades in BULL_TREND (its only firing regime): WR 51%, avg -0.13%.
- **`stop_activation_delay` default extended from 1 to 2** — Day 2 exits show same gap-through pattern as Day 1; Day 3 recovers to 55–69% WR.
- **`range_reversion` signal (new)** — ADX < 20 + BB < 0.10 + RSI < 30. RS-exempt. Priority 11.
- **`REGIME_BLOCKED` restructured** — split `_CHOPPY_BLOCKED` into `_DEFENSIVE_BLOCKED` and `_NEUTRAL_CHOP_BLOCKED`. Added `_BULL_TREND_BLOCKED`.
- **11 new tests**; **2351 passing, 100% coverage.**

---

### 1.41 — May 2026 — stop-activation delay + regime table fix + coverage hardening

- **`stop_activation_delay` parameter** — skips stop-loss checks for `trading_days_held` in `[1, stop_activation_delay]`. Day 1 gap-through exits average -5% at 0% WR; Day 3 exits recover to 56–68% WR.
- **`_REGIMES_ORDER` bug fix** — `_print_regime_table` used stale regime labels that never matched the 5-state labels introduced in v1.38, so the regime breakdown table was always blank.
- **2344 passing, 100% coverage.**

---

### 1.40 — May 2026 — cross-sectional relative strength rank filter

- **`_compute_rs_ranks`** — vectorised cross-sectional RS rank using `pandas.rank(axis=1, pct=True)`. Non-exempt signals from symbols with `rank_pct < 75` are silently skipped. Exempt signals: `mean_reversion`, `insider_buying`, `pead`.
- **Live scanning extended** — `get_spy_20d_return()` and per-snapshot `rs_rank_pct` (0–100) computed from ≥4 symbols.
- **12 new tests**; **2335 passing, 100% coverage.**

---

### 1.39 — May 2026 — daily P&L anchored to start-of-day baseline

- **`save_daily_run` P&L fix** — each mode-suffixed run file now anchors `daily_pnl` to `load_daily_baseline()` — the open-of-day portfolio value saved at the first run. Previously the close run showed a small negative P&L because it only measured the delta since the close run started.
- **2321 passing, 100% coverage.**

---

### 1.38 — May 2026 — 5-state market regime detection with hysteresis

- **`signals/market_regime.py` (new shared module)** — replaces the duplicated inline regime logic in `backtest/engine.py` and `execution/stock_scanner.py`. Five states: `BULL_TREND`, `NEUTRAL`, `CHOPPY`, `BEAR_DAY`, `STRESS_RISK_OFF`. Hysteresis via `_transition_with_hysteresis()` prevents single-bar whipsaws.
- **`REGIME_BLOCKED` canonical dict** now lives in `signals/evaluator.py` and is shared by both the backtest engine and the live scanner.
- **2320 tests, 100% coverage.**

---

### 1.37 — May 2026 — FMP fundamentals and analyst consensus

- **`data/fundamentals.py` (new)** — fetches financial ratios (ROE, profit margin, D/E, current ratio) via FMP and analyst consensus data. Both use a 24-hour JSON cache. Falls back to empty gracefully when `FMP_API_KEY` is unset.
- **`data/sentiment.py` restored** — now delegates to `get_analyst_consensus()` from `data/fundamentals.py`.
- **34 new tests**; **1842 passing**, 94% coverage.

---

### 1.36 — May 2026 — bulk yfinance download to eliminate 401 errors

- **`_bulk_download()` in `data/market_data.py`** — replaces 75+ parallel `Ticker.history()` calls with a single `yf.download(threads=False)` call. One session → one crumb handshake → Yahoo never sees the burst that triggers "Invalid Crumb" / 401 responses. OHLCV now succeeds 73/73 symbols.
- **Yahoo Finance `quoteSummary` endpoint removed** — restricted to paid subscribers; replaced `data/sentiment.py` with a no-op stub.
- **Tests updated**: 5 new `TestBulkDownload` tests. **1808 passing**, 94% coverage.

---

### 1.36 — May 2026 — signal & risk parity: canonical REGIME_BLOCKED + RiskConfig

- **`signals/evaluator.py` now exports `REGIME_BLOCKED`** — eliminates the longstanding divergence between `backtest/engine.py`'s `_REGIME_BLOCKED` and `execution/stock_scanner.py`'s `_LIVE_REGIME_BLOCKED` (which had a looser CHOPPY set and missed 9 signals blocked in the engine).
- **`risk/risk_config.py` (new)** — `RiskConfig` frozen dataclass bundles 5 exit-risk parameters with a `from_config()` classmethod.
- **`tests/test_risk_config.py` (new)** — 13 tests including AST-level checks that neither engine nor scanner defines a local blocking dict.
- **1826 tests passing, 93% coverage.**

---

### 1.35 — May 2026 — unified decision→execution audit trail + urllib3 CVE fix

- **Enriched `trades_executed` entries** — every BUY/SELL/WOULD_BUY/WOULD_SELL entry now carries `decision_type`, `confidence`, `key_signal`, and `reasoning` pulled from the corresponding AI decision.
- **Unified `decisions` list in portfolio tracker** — `save_daily_run` now builds and stores a `decisions` key — a single flat list of all buy and sell/hold decisions from one run.
- **urllib3 upgraded 2.6.3 → 2.7.0** — fixes two high-severity CVEs: GHSA-qccp-gfcp-xxvc and GHSA-mf9v-mfxr-j63j.
- **26 new tests**; 1813 passing.

---

### 1.34 — May 2026 — intraday session-replay engine

- **`backtest/intraday_engine.py` (new)** — rigorous intraday backtester that replays Alpaca 1-min bars bar-by-bar within each session. Strictly more valid than the daily engine's `--use-intraday` overlay: entry fills at the open of the bar following the signal bar (no lookahead). `run_intraday_backtest()` returns a dict matching the daily `run_backtest()` schema.
- **`data/intraday_fetcher.py` (new)** — standalone Alpaca 1-min bar fetcher with disk-level pickle cache per (symbol, start, end).
- **29 new tests**; 1784 passing.

---

### 1.33 — May 2026 — multiple testing correction (Holm-Bonferroni)

- **`_binomial_p_value(wins, n, p0=0.5)`** — exact one-sided binomial test computed in log-space from `stdlib math` only (no scipy dependency).
- **`_holm_bonferroni(p_values, alpha=0.05)`** — step-down Holm-Bonferroni correction across all regime×signal cells.
- **`compute_regime_blocked(regime_stats, min_trades=20)`** — data-driven `{regime: {signals_to_block}}` derived from regime stats.
- **28 new tests**; 1755 total.

---

### 1.28 — May 2026 — backtest validity warnings + beat-baseline metric + test reliability

- **`_REGIME_BLOCKED` and `_LIVE_REGIME_BLOCKED` relabelled as working hypotheses** — comments explicitly state blocks are economically plausible and empirically suggestive but not independently validated.
- **Low-confidence cell warnings in regime table output** — flags any cell with n < 30 trades with `*`.
- **`beat_baseline_folds` / `beat_baseline_pct` added to walk-forward summary** — reports how many OOS folds beat the equal-weight universe baseline.
- **Survivorship bias warning on all backtest print output.**
- **1713 tests, zero ruff violations, zero mypy errors.**

---

### 1.27 — May 2026 — regime-aware live scanner + backtest signal sync

- **`_LIVE_REGIME_BLOCKED` (new module-level dict in `execution/stock_scanner.py`)** — blocks signals whose per-regime performance was validated negative across the 2021–2026 walk-forward OOS run.
- **`prefilter_candidates` now accepts `regime: str | None = None`** — signals in the corresponding blocked set are silently skipped before the matched-signals list is assembled.
- **`rs_leader` removed from live scanner** — no positive edge in any regime across 57 backtest trades.
- **`main.py` passes detected regime to `prefilter_candidates`.**
- **8 new tests**; 1711 total.

---

### 1.26 — May 2026 — regime-stratified signal analysis + walk-forward OOS validation

- **`run_signal_analysis` (new function)** — groups closed trades by (signal, entry_regime) and (signal, days_held) to produce regime-stratified win-rate/avg-return tables and hold-period decay tables.
- **`--signal-analysis` CLI flag.**
- **`entry_regime` added to SELL trade records.**
- **`_REGIME_BLOCKED` updated** with evidence-backed additions: `iv_compression` added to BEAR_DAY block (avg -1.3%, 24 trades); `mean_reversion`, `macd_crossover`, `inside_day_breakout` added to CHOPPY block.
- **16 new tests**; 1707 total.

---

### 1.25 — May 2026 — regime-stratified signal breakdown + hold-period decay

- **Regime detection added to backtest simulation** — entry regime stored on each position at open time.
- **`run_regime_analysis` / `run_hold_decay_analysis` (new functions)** — group closed trades by regime and hold period.
- **12 new tests**; 1691 total.

---

### 1.24 — May 2026 — greedy backward elimination + --use-earnings-only

- **`run_backward_elimination` (new function)** — greedy backward elimination: iteratively disables the signal whose removal most improves Sharpe, stopping when no further improvement is possible.
- **`--backward-elimination` CLI flag.**
- **`--use-earnings-only` flag** — prefetches yfinance EPS history only; skips the 90-minute EDGAR insider fetch; enabling `pead` in ~2 minutes.
- **`disabled_signals` parameter** added to `_entry_signal` and `_run_simulation`.
- **24 new tests**; 1691 total.

---

### 1.23 — May 2026 — independent ablation study

- **`run_ablation` (new function)** — single-pass independent ablation: disables each signal in isolation against the same baseline, measuring ΔSharpe.
- **`--ablation` CLI flag.**
- **12 new tests**; 1679 total.

---

### 1.22 — May 2026 — historical fundamentals: point-in-time pead + insider_buying backtesting

- **`backtest/historical_fundamentals.py` (new module)** — pre-fetches all available historical EPS surprise events and SEC EDGAR Form 4 open-market purchases once at backtest startup. During simulation, `pead_active_on_date` and `insider_state_on_date` walk these lists with strictly no lookahead.
- **`pead` and `insider_buying` fully backtestable** — both signals now fire in the simulation engine when historical data is loaded.
- **41 new tests**; 1667 total.

---

### 1.21 — May 2026 — iv_compression signal: historical volatility percentile squeeze

- **`iv_compression` signal** — measures where today's 20-day annualized realized volatility sits in its rolling 252-day range. `hv_rank < 0.20` means the stock is in the bottom quintile of its annual vol history.
- **`hv_20d` and `hv_rank` indicators** added to both `data/market_data.py` (live) and `backtest/engine.py`. Signal is fully backtestable without external data.
- **13 new tests**; 1626 total.

---

### 1.20 — May 2026 — PEAD signal: post-earnings announcement drift

- **`pead` signal (`data/earnings_surprise.py`)** — fetches `earnings_dates` from yfinance for each symbol. Fires when `Surprise(%) >= 5.0%` (analyst beat) and the 5-day return is still positive (price confirming drift, not reversing). Hold limit: 3 days.
- **16 new tests**; 1613 total.

---

### 1.19 — May 2026 — momentum_12_1, insider buying (SEC EDGAR Form 4), AV news sentiment, quality pre-filter

- **`momentum_12_1` signal (Jegadeesh-Titman 12-1)** — computes 12-month return minus 1-month return. Fires when the factor exceeds 10.0%, EMA9 > EMA21, and ADX ≥ 20. Blocked on `BEAR_DAY` and `CHOPPY` regimes.
- **Fundamental quality pre-filter (`_passes_quality_screen`)** — rejects stocks with negative ROE, negative profit margins, or debt-to-equity > 300.
- **Insider cluster buying signal** — `data/insider_feed.py` queries the SEC EDGAR submissions API. Only open-market purchases (transaction code `P`) counted. Fires when ≥2 distinct corporate insiders buy within 10 days.
- **Alpha Vantage NEWS_SENTIMENT enrichment** — `data/av_sentiment.py` fetches structured per-ticker sentiment scores (−1 to +1). Silently disabled when `ALPHA_VANTAGE_API_KEY` is absent.
- **91 new tests**; 1597 total.

---

### 1.18 — May 2026 — Intraday signals: VWAP, opening range breakout, intraday momentum

- **Alpaca minute-bar intraday layer** (`data/market_data.get_intraday_data`) — fetches Alpaca minute bars from market open to now and computes gap_pct, intraday_change_pct, VWAP + price_above_vwap, opening range high/low, `orb_breakout_up` (volume-confirmed break above 30-min range), and `intraday_rsi` (RSI-14 on 5-minute bars).
- **Three new prefilter signals** — `vwap_reclaim`, `orb_breakout`, `intraday_momentum`.
- **Midday run now eligible for buys** — `skip_buys` gate changed from `mode in ("midday", "close", "open_sells")` to `mode in ("close", "open_sells")`.
- **1506 tests, 98.25% coverage.**

---

### 1.17 — May 2026 — Pre-market order fill reconciliation

- **`place_buy_order()` final-check "filled" path** — the post-timeout `get_order_by_id` check previously only handled `"partially_filled"`, not `"filled"`. Orders that filled between `wait_for_fill` exhausting its poll window and the final broker query fell through to `TIMEOUT` with `filled_qty=0.0`.
- **Late-fill reconciliation in `main.py`** — queries the order-ledger for today's `timeout` intents, cross-references against live Alpaca positions, and for any confirmed fill: updates the intent to `filled`, calls `record_buy()`, appends to `all_trades`.
- **`reconcile_filled_intents()` in `order_ledger.py`** — mirror of `auto_cancel_timeout_intents()`.
- **6 new tests**; 1321 total.

---

### 1.16 — May 2026 — Alpha instrumentation: candidate funnel visibility, replay context parity, engine labeling

- **`matched_signals` annotation on prefiltered candidates** — `prefilter_candidates()` now returns each qualified snapshot with a `matched_signals: list[str]` field.
- **`score_candidate()` deterministic scoring function** — scores each prefiltered candidate by RSI distance, BB distance, volume confirmation, relative strength, and signal count.
- **`PREFILTER_CANDIDATES` and `CANDIDATE_SELECTION` audit events.**
- **`summarise_for_ai()` bar provenance fields** — `bar_date`, `bar_is_final`, `data_source`.
- **`backtest/engine.py` rule-proxy labeling** — `_run_simulation()` return dict now includes `validation_scope: "rule_proxy_only"`, `signals_tested`, and `signals_not_tested`.
- **1312 tests, 93% coverage.**

---

### 1.15 — May 2026 — Prompt quality: decision-support framing, do_nothing_case, structured lessons

- **`SYSTEM_PROMPT` reframed from "trader" to "decision-support analyst"** — removes action-seeking bias. Adds evidence weighting hierarchy and confidence calibration rubric.
- **`do_nothing_case` and `invalidation_trigger` required on every buy candidate** — added to `_DECISION_TOOL` schema, `BuyCandidate` Pydantic model, and TASK prompt.
- **Weekly review lessons are now structured with expiry and regime filter** — `applies_when` and `expiry` fields. `get_latest_review()` filters out expired lessons and lessons inapplicable to the current regime.
- **29 new tests**; 1313 total.

---

### 1.14 — May 2026 — Execution quality telemetry, live-shadow audit events, LIVE_RUNBOOK

- **Fill avg price captured end-to-end** — `wait_for_fill()` now returns `(filled_qty, filled_avg_price)`.
- **`ORDER_EXEC_QUALITY` event now populated with actual fill data** — `bid`, `ask`, `spread_bps`, `fill_avg_price`, `slippage_vs_mid_bps`.
- **`--live-shadow` emits `WOULD_BUY` / `WOULD_SELL` audit events.**
- **`LIVE_RUNBOOK.md` added** — full operations guide with pre-live checklist, canary procedure, incident response.
- **9 new tests**; 1284 total.

---

### 1.13 — May 2026 — Structural safety caps (reviewer-required fixes)

- **`MAX_POSITIONS` always caps sizer** — broker/sizer can never grant more slots than the hard config cap.
- **Experiment drawdown cap enforced** — `MAX_EXPERIMENT_DRAWDOWN_USD` compared against a write-once baseline; buys blocked once the cumulative experiment loss reaches the cap.
- **PARTIAL/TIMEOUT buy triggers immediate stop check** — after any ambiguous fill, `ensure_stops_attached()` runs immediately in the buy loop.
- **Unexpected broker positions halt in live mode** — `reconcile_positions()` returns `set[str]` of unknown symbols; if non-empty in live mode, bot writes halt file and exits.
- **7 new tests**; 1267 total.

---

### 1.12 — May 2026 — Close last permissive fallbacks (10/10 safety)

- **`--live-shadow` now runs all live gates** — only order submission is suppressed.
- **`has_active_intent()` fails closed** — DB failure now raises `OrderLedgerUnavailable` instead of returning `False`.
- **`OrderLedgerUnavailable` wired into buy loop** — buy loop catches it and breaks (with alert).
- **`create_intent()` failure blocks live broker submission** — in live mode (`IS_PAPER=False`), if intent creation fails the bot raises `OrderLedgerUnavailable`.
- **Quote gate last-trade failure fails closed** — a trade-feed exception now raises `BrokerStateUnavailable`.
- **3 new CI invariant tests**; 1260 total.

---

### 1.11 — May 2026 — Live safety hardening (10/10 safety for £150 experiment)

- **Fail-closed broker state** — `has_pending_buy()` and `get_total_open_exposure()` now raise `BrokerStateUnavailable` instead of returning safe defaults on exception.
- **Durable order-intent ledger** — `order_intents` + `order_events` SQLite tables record every buy attempt before broker submission.
- **Live quote/spread/freshness gate** — `execution/quote_gate.py` validates real-time Alpaca data before every live order.
- **Startup health report** — `utils/health.py` runs 7 checks at startup; RED blocks buys.
- **CI invariant tests** — `tests/test_safety_invariants.py` (25 tests) encodes safety properties that must not regress.
- **Adversarial LLM fixtures** — `tests/test_adversarial_llm.py` (26 tests) deterministically rejects prompt injection, malformed Claude output, hallucinated tickers, duplicate symbols, buy/sell conflicts.
- **1260 tests, 100% coverage.**

---

### 1.10 — May 2026 — Live-safety hardening for £150 experiment

- **Capital containment bounded** — `SMALL_ACCOUNT_MODE=true` activates a complete small-account cap profile.
- **Duplicate-buy prevention** — `has_pending_buy()` queries broker open orders before every buy. `client_order_id` now uses `{SYMBOL}-BUY-{DATE}` (stable across same-day restarts; Alpaca deduplicates).
- **Stop failure is fatal** — when trailing stop placement fails after a live fill, `_handle_stop_failure()` immediately attempts to flatten the position. If that also fails, a halt file is written.
- **VIX-adjusted stop wired** — `place_trailing_stop()` now accepts a `trail_percent` override; `main.py` passes the VIX-adjusted trail on every stop placement.
- **`ensure_stops_attached()` returns fatal bool** — if stop re-attachment fails for a whole-share live position at startup, the bot writes a halt file and exits.
- **Dollar daily loss cap** — `MAX_DAILY_LOSS_USD` (default $20 in small-account mode) triggers close-all independently of the percentage cap.
- **1206 tests, 100% coverage.**

---

### 1.9 — May 2026 — Backtest integrity + risk hardening

- **Lookahead bias eliminated** — signals now use T-1 bar indicators; entries fill at T open price, not T close.
- **Transaction costs modelled** — `SLIPPAGE_BPS=5` and `SPREAD_BPS=3` added to `config.py` and applied to every fill in `_run_simulation`.
- **Walk-forward parameter optimisation** — `run_walk_forward_optimized` performs genuine out-of-sample validation: grid-search over 576 param combinations on a rolling training window, evaluate best params on the immediately following test window.
- **MIN_TRAIN_TRADES raised 5 → 20** — Sharpe estimates based on fewer than 20 trades are too noisy.
- **Historical replay harness** — `backtest/replay.py` adds `run_historical_replay` — downloads full OHLCV history once, then simulates the live pipeline day-by-day using strict point-in-time slicing. Claude is called for real on each simulation date.
- **Ruff format enforced across entire codebase** — first full format pass applied.
- **1151 tests, 100% coverage, zero ruff violations.**

---

### 1.8 — May 2026 — Dynamic universe + 5 new strategies

- **Dynamic scan universe** — `execution/universe.py` fetches all tradable + fractionable US equity symbols, screens by price (≥ $5) and volume (≥ 500K) via the Alpaca snapshot API, and caches up to 500 symbols for 24 hours.
- **Five new signal types** — `bb_squeeze`, `breakout_52w`, `rs_leader`, `inside_day_breakout`, `trend_pullback`. The bot now operates across 12 distinct signal types covering mean-reversion, momentum, trend, volatility expansion, and catalyst families.
- **1042 tests, 100% coverage.**

---

### 1.7 — May 2026 — 100% test coverage & code quality

- **100% line coverage** — test count grew from 460+ to 981 across 3,017 executable statements.
- **New test files** — `tests/test_dashboard.py` and `tests/test_run_diagnostics.py`.
- **Zero ruff violations** — fixed 5 linting violations in production files.

---

### 1.6 — April 2026 — FDE hardening

- **SQLite migration** — replaced JSON file state with a single SQLite database (`logs/investorbot.db`). ACID transactions eliminate the partial-write race condition that caused trade history loss in earlier versions.
- **run_id correlation** — every run generates a UUID attached to every audit event, AI decision, and order.
- **LLM cost tracking** — token usage and estimated cost per Claude call logged to the `llm_usage` table.
- **Demo mode** — `python cli.py demo` runs a complete simulated cycle on static fixture data with no API credentials required.
- **ADRs** — five Architecture Decision Records added to `docs/adr/`.
- **LLM eval fixtures** — six fixture files in `evals/` covering prompt injection, hallucinated tickers, bear market suppression, conflicting signals, earnings risk, and malformed AI responses.

---

### 1.5 — April 2026 — Pre-live hardening

Four bugs discovered through log review after the first live paper-trading session:

- **Trailing stops never attached for fractional positions at buy time** — both call sites were not passing `current_price`; function silently returned `None`.
- **Stop exposure window between runs** — `ensure_stops_attached` only ran at the start of each run, not after the buy loop. Fixed by adding a second call after the buy loop.
- **`wait_for_fill` too short for paper API** — increased default from 10 seconds to 30 seconds.
- **`conf=` not humanised in email** — the detail string was built with `conf=8` but `_humanise_detail` only matched `confidence=`.

Consolidated on `scripts/run_scheduler.py` as the single production runner. Cron entries removed.

---

### 1.4 — April 2026 — Python 3.12 standardisation

- **Standardised entire stack on Python 3.12** — venv, Dockerfile (`python:3.12-slim`), and scheduler all now use the same interpreter.
- **Removed `from __future__ import annotations` shims** from all 12 production files.
- **Upgraded all dependencies** to latest: pandas 3.0.2, numpy 2.4.4, yfinance 1.3.0, curl_cffi 0.15.0, requests 2.33.1.

---

### 1.3 — April 2026 — Full test coverage

Comprehensive test pass covering every public function and every unhappy path across all modules. Test count: 203 → 460.

---

### 1.2 — April 2026 — Day-one incident fixes

Six failures surfaced in the first two hours of live paper trading on 27 April 2026. All diagnosed from logs alone; all fixed within the same session. See [docs/incidents.md](docs/incidents.md) for full details.

---

### 1.1 — April 2026

Added web dashboard (Streamlit, 5 pages), CLI (`cli.py`), Docker deploy, AI decision log, personalised email greetings per recipient, Sharpe ratio in backtester, backtest results persisted for the dashboard, file locking on position metadata, and dynamic backtest end date.

---

### 1.0 — April 2026

Initial release. Full AI-governed paper-trading capability with AI-driven decision making, risk-budget sizing with Kelly telemetry, multi-layer risk management, regime-aware signal tracking, weekly self-review with constrained parameter proposals, and multi-recipient email reporting.

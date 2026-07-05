"""Workshop v2 #7 — GDELT news-tone divergence: FEASIBILITY PROBE (not a full backtest).

Thesis (orthogonal alt-data, the last non-price/flow angle in the v2 queue): when a name's price
falls but its news TONE improves (or vice-versa), the divergence flags mispricing that reverts. If
real and independent of price-reversal (N1), it would be genuinely additive to a book whose only
edge is price mean-reversion. Prior: LOW — GDELT's per-entity tone is GENERAL news tone (products,
politics, executives), a noisy proxy for financial sentiment; the clean version of this effect needs
firm-specific news (RavenPack-grade), not GDELT.

This script PROBES the GDELT DOC 2.0 TimelineTone API to decide whether the "bounded build" is
actually bounded, BEFORE committing to a full universe feed + data module + backtest.

VERDICT: DECLINE the build — feasible as data, NOT deployable as a live signal.
  1. Data exists: "Tesla Inc" -> 92 daily tone points over Jan-Apr 2024, mean -0.08, sd 0.88,
     real range [-4.2, +1.3]. So a usable per-ticker daily tone series IS obtainable.
  2. Rate limits are punishing: 2nd/3rd names 429 (Too Many Requests) even at 6s spacing; the
     throttle is tighter than the stated 5s and appears IP-level (bursts blocked).
  3. DEPLOYABILITY is the blocker: the live signal needs a DAILY tone fetch for the full ~906-name
     universe. At GDELT's limits that is ~2.5h+ of fragile, throttle-prone fetching every day,
     inside a SEQUENTIAL scheduler where any long blocking job freezes all trading jobs. A signal
     that cannot be fed live is worthless regardless of backtest — so the backtest is moot.
  Combined with the low prior (general-news tone != financial sentiment), the bounded build is not
  bounded and the payoff cannot be realized. Declined on cost-vs-prior + deployability grounds.
  Salvage path if ever wanted: a small hand-curated watchlist (~20-30 distinctive names) fetched
  slowly could be tested, but it is statistically underpowered and still non-deployable at scale.

Usage: python scripts/gdelt_tone_probe.py   # prints tone-series stats for a few distinctive names
"""

from __future__ import annotations

import json
import statistics as st
import time
import urllib.parse
import urllib.request

_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_NAMES = {"TSLA": "Tesla Inc", "NVDA": "Nvidia Corp", "CVS": "CVS Health"}


def _fetch_tone(name: str, start: str, end: str, retries: int = 3) -> list[float]:
    q = urllib.parse.urlencode(
        {
            "query": f'"{name}"',
            "mode": "TimelineTone",
            "startdatetime": start,
            "enddatetime": end,
            "format": "json",
        }
    )
    delay = 6.0
    for attempt in range(retries):
        try:
            raw = (
                urllib.request.urlopen(f"{_BASE}?{q}", timeout=40).read().decode("utf-8", "replace")
            )
            d = json.loads(raw)
            return [p["value"] for p in d.get("timeline", [{}])[0].get("data", [])]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return []


def main() -> None:
    print("=== GDELT TimelineTone feasibility probe (2024-01..2024-04) ===")
    for i, (tic, name) in enumerate(_NAMES.items()):
        if i:
            time.sleep(6)
        try:
            vals = _fetch_tone(name, "20240101000000", "20240401000000")
        except Exception as e:  # noqa: BLE001 — probe: report and continue
            print(f"  {tic:5} {name:12} ERR {type(e).__name__}: {str(e)[:60]}")
            continue
        if not vals:
            print(f"  {tic:5} {name:12} empty")
            continue
        print(
            f"  {tic:5} {name:12} n={len(vals):3} tone mean={st.mean(vals):+.2f} "
            f"sd={st.pstdev(vals):.2f} range[{min(vals):+.1f},{max(vals):+.1f}]"
        )
    print("\n  See module docstring VERDICT: DECLINE — data exists but not deployable at scale.")


if __name__ == "__main__":
    main()

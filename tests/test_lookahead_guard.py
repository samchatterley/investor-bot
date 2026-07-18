"""Tests for experiment/lookahead_guard.py — the future-data-poisoning lookahead audit."""

import unittest

import numpy as np
import pandas as pd

from data import market_data
from experiment.lookahead_guard import (
    Leak,
    audit_no_lookahead,
    drop_after,
    leaky_canary,
)

_DATA = [
    {"date": "2026-01-01", "value": 1.0},
    {"date": "2026-01-02", "value": 2.0},
    {"date": "2026-01-03", "value": 3.0},
]
_DATES = ["2026-01-01", "2026-01-02", "2026-01-03"]


def _honest(records, as_of):
    """Control: a correct point-in-time computation — latest value on/before as_of."""
    past = [r for r in records if r["date"] <= as_of]
    return past[-1]["value"] if past else 0.0


def _crashes_on_poison(records, as_of):
    """Leaky by positional read: assumes the row after as_of exists, so it IndexErrors once dropped."""
    ordered = sorted(records, key=lambda r: r["date"])
    idx = next(i for i, r in enumerate(ordered) if r["date"] == as_of)
    return ordered[idx + 1]["value"]  # reads the future -> raises when the future is poisoned away


class TestDropAfter(unittest.TestCase):
    def test_keeps_only_on_or_before(self):
        kept = drop_after(_DATA, "2026-01-02")
        self.assertEqual([r["date"] for r in kept], ["2026-01-01", "2026-01-02"])

    def test_missing_date_key_defaults_to_kept(self):
        kept = drop_after([{"value": 9.0}], "2026-01-01")
        self.assertEqual(kept, [{"value": 9.0}])  # "" <= as_of -> retained


class TestAuditNoLookahead(unittest.TestCase):
    def test_honest_computation_has_no_leaks(self):
        self.assertEqual(audit_no_lookahead(_honest, _DATA, _DATES), [])

    def test_canary_divergence_is_caught(self):
        leaks = audit_no_lookahead(leaky_canary, _DATA, _DATES)
        # leaks on every date that has a future value (the last date has none, so it can't leak)
        self.assertEqual([lk.date for lk in leaks], ["2026-01-01", "2026-01-02"])
        self.assertIsInstance(leaks[0], Leak)
        self.assertIn("output changed", leaks[0].detail)

    def test_crash_only_on_poison_is_caught_as_leak(self):
        leaks = audit_no_lookahead(_crashes_on_poison, _DATA, ["2026-01-01"])
        self.assertEqual(len(leaks), 1)
        self.assertIn("needed future data", leaks[0].detail)

    def test_last_date_canary_does_not_falsely_flag(self):
        # at the final date there is no future to read, so even the canary is clean there
        self.assertEqual(audit_no_lookahead(leaky_canary, _DATA, ["2026-01-03"]), [])


class TestLeakyCanary(unittest.TestCase):
    def test_reads_first_future_value(self):
        self.assertEqual(leaky_canary(_DATA, "2026-01-01"), 2.0)

    def test_no_future_returns_zero(self):
        self.assertEqual(leaky_canary(_DATA, "2026-01-03"), 0.0)


def _synthetic_ohlcv(n: int = 160, seed: int = 7) -> pd.DataFrame:
    """A realistic offline daily OHLCV frame (gentle uptrend + noise) for the real-path audit."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-06-02", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
    )


def _poison_frames(preloaded: dict, as_of: str) -> dict:
    """Poison for the {symbol: OHLCV DataFrame} shape: drop every bar strictly after as_of."""
    ts = pd.Timestamp(as_of)
    return {sym: df[df.index <= ts].copy() for sym, df in preloaded.items()}


class TestRealSignalPathAudit(unittest.TestCase):
    """Audit the deterministic slice -> features -> snapshot path replay.py actually trusts."""

    def setUp(self):
        self.preloaded = {"TEST": _synthetic_ohlcv()}
        self.dates = [d.strftime("%Y-%m-%d") for d in self.preloaded["TEST"].index[-5:]]

    def _snapshot_on(self, preloaded, as_of):
        df = market_data.fetch_stock_data("TEST", 30, preloaded=preloaded, as_of=as_of)
        return market_data.summarise_for_ai("TEST", df, is_preloaded=True)

    def test_real_path_is_lookahead_clean(self):
        # fetch_stock_data slices to <= as_of BEFORE computing indicators, so masking the future is a
        # no-op: the produced snapshot is identical. This is the payoff — replay's PIT claim, proven.
        self.assertEqual(
            audit_no_lookahead(
                self._snapshot_on, self.preloaded, self.dates, poison=_poison_frames
            ),
            [],
        )

    def test_guard_catches_full_sample_normalization_leak(self):
        # the classic bug: normalise against the LATEST close in the full series (which, before slicing,
        # sits in the future) instead of a trailing window. The guard must flag every non-final date.
        def _leaky(preloaded, as_of):
            df = preloaded["TEST"]
            sliced = df[df.index <= pd.Timestamp(as_of)]
            return round(float(sliced["Close"].iloc[-1] / df["Close"].iloc[-1]), 6)

        leaks = audit_no_lookahead(_leaky, self.preloaded, self.dates, poison=_poison_frames)
        self.assertEqual(len(leaks), 4)  # every date but the last has a future close to leak
        self.assertIsInstance(leaks[0], Leak)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

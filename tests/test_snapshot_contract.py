"""Enforce the snapshot seam contract (Fable finding 3) — see signals/snapshot.py.

Two invariants, each turning a silent live/backtest divergence into a CI failure:
  1. Parity: every evaluator-read field that is not declared live-only is produced by the backtest
     long path (no dead-wire-in-backtest).
  2. Fail-closed defaults: no field is read with two different literal defaults unless it is a blessed
     intentional split (the finding-11 regression guard).
Plus hygiene: the two allowlists contain no stale entries.
"""

import os
import re
import unittest
from unittest.mock import patch

import pandas as pd

from signals.snapshot import INTENTIONAL_SPLIT_DEFAULTS, LIVE_ONLY_FIELDS

# Resolve source files against the repo root (this file's parent's parent), not the current working
# directory, so the contract holds no matter where pytest is invoked from (e.g. from a parent dir).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVALUATOR_SRC = "signals/evaluator.py"
_ENGINE_SRC = "backtest/engine.py"


def _read(path: str) -> str:
    with open(os.path.join(_ROOT, path)) as fh:
        return fh.read()


def _evaluator_reads() -> set[str]:
    """Every snapshot field the evaluator consumes, via snapshot.get("x") or snapshot["x"]."""
    txt = _read(_EVALUATOR_SRC)
    got = set(re.findall(r'snapshot\.get\(\s*["\']([a-z0-9_]+)["\']', txt))
    idx = set(re.findall(r'snapshot\[\s*["\']([a-z0-9_]+)["\']\s*\]', txt))
    return got | idx


def _evaluator_defaults() -> dict[str, set[str]]:
    """field -> set of distinct literal defaults it is read with (snapshot.get('x', DEFAULT))."""
    txt = _read(_EVALUATOR_SRC)
    out: dict[str, set[str]] = {}
    for f, d in re.findall(r'snapshot\.get\(\s*["\']([a-z0-9_]+)["\']\s*,\s*([^)]+?)\)', txt):
        out.setdefault(f, set()).add(d.strip())
    return out


def _capture_backtest_long_snapshot() -> set[str]:
    """Drive the REAL backtest long builder (_entry_signal) with a maximal row + every enrichment
    source it accepts, capturing the snapshot dict handed to evaluate_signals."""
    import backtest.engine as eng

    body = _read(_ENGINE_SRC)
    i = body.index("def _row_to_snapshot")
    j = body.index("\ndef ", i + 10)
    fn = body[i:j]
    row_cols = set(re.findall(r'row\.get\(\s*["\']([A-Za-z0-9_]+)["\']', fn)) | set(
        re.findall(r'row\[\s*["\']([A-Za-z0-9_]+)["\']\s*\]', fn)
    )
    fund_keys = set(re.findall(r'fundamentals\.get\(\s*["\']([A-Za-z0-9_]+)["\']', fn)) | {
        "altman_z",
        "piotroski_f",
        "fcf_yield",
        "accruals_ratio",
        "gross_margin_trend",
        "forward_pe",
        "short_pct_float",
        "short_ratio",
        "pead_active",
        "insider_cluster",
        "insider_strong_cluster",
        "insider_comp_ratio",
        "activist_filing",
        "insider_large_buy",
    }
    row = pd.Series(dict.fromkeys(row_cols, 1.0))
    macro = {
        "macro_credit_stress": True,
        "macro_duration_flight": True,
        "macro_copper_gold_positive": True,
        "macro_usd_strong": True,
        "macro_yield_curve": True,
        "macro_yield_curve_inverted_days": 1,
    }

    captured: dict = {}

    def _spy(snap, **kwargs):
        captured.update(snap)
        return []

    with patch.object(eng, "evaluate_signals", _spy):
        eng._entry_signal(
            row,
            fundamentals=dict.fromkeys(fund_keys, 1.0),
            spy_ret_5d=0.0,
            spy_ret_10d=0.0,
            rs_rank_pct=50.0,
            breadth_thrust=True,
            calendar_month=6,
            macro_flags=macro,
        )
    return set(captured.keys())


class TestSnapshotContract(unittest.TestCase):
    def test_backtest_long_path_produces_every_core_field(self):
        """Parity guard: a field the evaluator reads and is not declared live-only MUST be produced by
        the backtest long path — else the signal is dead-wired in the backtest."""
        core = _evaluator_reads() - LIVE_ONLY_FIELDS
        produced = _capture_backtest_long_snapshot()
        missing = sorted(core - produced)
        self.assertEqual(
            missing,
            [],
            f"core evaluator fields not produced by the backtest long path: {missing} — "
            f"either produce them in backtest/engine.py or declare them LIVE_ONLY_FIELDS",
        )

    def test_no_unblessed_inconsistent_defaults(self):
        """Fail-closed guard (finding 11): a field read with two different literal defaults is a
        fail-open risk unless it is a blessed intentional per-site split."""
        split = {f for f, ds in _evaluator_defaults().items() if len(ds) > 1}
        unblessed = sorted(split - INTENTIONAL_SPLIT_DEFAULTS)
        self.assertEqual(
            unblessed,
            [],
            f"fields read with inconsistent defaults and not blessed: {unblessed} — reconcile to one "
            f"fail-closed default (see finding 11 / spread_proxy_20d) or add to INTENTIONAL_SPLIT_DEFAULTS",
        )

    def test_live_only_allowlist_has_no_stale_entries(self):
        reads = _evaluator_reads()
        stale = sorted(LIVE_ONLY_FIELDS - reads)
        self.assertEqual(stale, [], f"LIVE_ONLY_FIELDS the evaluator no longer reads: {stale}")

    def test_intentional_split_allowlist_has_no_stale_entries(self):
        # every blessed field must (a) still be read and (b) still actually have >1 default
        defaults = _evaluator_defaults()
        for f in INTENTIONAL_SPLIT_DEFAULTS:
            self.assertIn(
                f, defaults, f"{f} blessed as a split default but no longer read with a default"
            )
            self.assertGreater(
                len(defaults[f]),
                1,
                f"{f} blessed as a split default but now has a single default — unbless it",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

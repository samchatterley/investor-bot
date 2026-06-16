"""Tests for experiment/collection.py — live point-in-time observation capture."""

import json
import os
import tempfile
import unittest
from decimal import Decimal

from experiment.collection import (
    _features,
    _json_safe,
    append_observations,
    build_observation,
    log_run_observations,
)


class TestJsonSafe(unittest.TestCase):
    def test_finite_float_passthrough(self):
        self.assertEqual(_json_safe(1.5), 1.5)

    def test_nan_and_inf_become_none(self):
        self.assertIsNone(_json_safe(float("nan")))
        self.assertIsNone(_json_safe(float("inf")))

    def test_bool_int_str_none_passthrough(self):
        self.assertIs(_json_safe(True), True)
        self.assertEqual(_json_safe(3), 3)
        self.assertEqual(_json_safe("XLK"), "XLK")
        self.assertIsNone(_json_safe(None))

    def test_coercible_becomes_float(self):
        self.assertEqual(_json_safe(Decimal("2.25")), 2.25)

    def test_coercible_nan_becomes_none(self):
        self.assertIsNone(_json_safe(Decimal("NaN")))

    def test_noncoercible_becomes_str(self):
        class Weird:
            def __str__(self):
                return "weird"

        self.assertEqual(_json_safe(Weird()), "weird")


class TestFeatures(unittest.TestCase):
    def test_only_known_keys_and_coerced(self):
        cand = {"symbol": "AAPL", "rsi_14": 30.0, "vol_ratio": float("nan"), "unknown": 1}
        feats = _features(cand)
        self.assertEqual(feats["rsi_14"], 30.0)
        self.assertIsNone(feats["vol_ratio"])  # NaN coerced
        self.assertNotIn("unknown", feats)
        self.assertNotIn("symbol", feats)  # symbol is not a feature key


class TestBuildObservation(unittest.TestCase):
    def _cand(self, **kw):
        base = {
            "symbol": "AAPL",
            "matched_signals": ["pead"],
            "rsi_14": 35.0,
            "atr": 2.5,
            "current_price": 150.0,
            "ma_event": True,
        }
        base.update(kw)
        return base

    def test_selected_material_candidate(self):
        rec = build_observation(
            self._cand(),
            decision_date="2026-06-16",
            selected=True,
            confidence=0.8,
            deterministic_score=0.6,
            deterministic_rank=2,
            run_id="r1",
            mode="open_buys",
            market_context={"regime": "BULL", "vix": 16.0},
        )
        self.assertEqual(rec["symbol"], "AAPL")
        self.assertEqual(rec["split"], "holdout")  # 2026 > validation_end
        self.assertIn("earnings_surprise_or_drift", rec["material_context"])  # pead
        self.assertIn("ma_event", rec["material_context"])  # flag
        self.assertEqual(rec["fired_signals"], ["pead"])
        self.assertEqual(rec["features"]["rsi_14"], 35.0)
        self.assertIsNone(rec["forward_r"])
        self.assertEqual(rec["expectancy"], {})
        self.assertTrue(rec["extra"]["arm3_ai_selected"])
        self.assertEqual(rec["extra"]["arm3_ai_confidence"], 0.8)
        self.assertEqual(rec["extra"]["arm1_deterministic_rank"], 2)
        self.assertEqual(rec["extra"]["market_context"]["regime"], "BULL")
        # blinded view drops absolute price
        self.assertNotIn("current_price", rec["blinded_features"])

    def test_vetoed_candidate_defaults(self):
        rec = build_observation(
            {"symbol": "T", "signals": ["mean_reversion"]},  # 'signals' fallback, no flags
            decision_date="2022-06-01",
            selected=False,
            confidence=None,
            deterministic_score=None,
            deterministic_rank=None,
            run_id="r2",
            mode="midday",
        )
        self.assertFalse(rec["extra"]["arm3_ai_selected"])
        self.assertIsNone(rec["extra"]["arm3_ai_confidence"])
        self.assertEqual(rec["fired_signals"], ["mean_reversion"])
        self.assertEqual(rec["material_context"], [])
        self.assertEqual(rec["split"], "train")  # 2022 <= train_end
        self.assertEqual(rec["extra"]["market_context"], {})


class TestAppendObservations(unittest.TestCase):
    def test_empty_writes_nothing(self):
        self.assertEqual(append_observations([], "/tmp/_unused_obs.jsonl"), 0)

    def test_appends_json_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "obs.jsonl")
            n = append_observations([{"a": 1}, {"b": 2}], path)
            self.assertEqual(n, 2)
            # append (not overwrite) on a second call
            append_observations([{"c": 3}], path)
            with open(path) as fh:
                lines = [json.loads(line) for line in fh]
            self.assertEqual(lines, [{"a": 1}, {"b": 2}, {"c": 3}])


class TestLogRunObservations(unittest.TestCase):
    def _run(self, path, score_fn=lambda c: 0.5):
        candidates = [
            {"symbol": "AAPL", "matched_signals": ["pead"], "ma_event": True},
            {"symbol": "T", "matched_signals": ["mean_reversion"]},
        ]
        return log_run_observations(
            candidates,
            buy_candidates=[{"symbol": "AAPL", "confidence": 0.9}],  # AI picked AAPL, vetoed T
            ranked=[candidates[0], candidates[1]],
            decision_date="2026-06-16",
            run_id="r1",
            mode="open_buys",
            market_context={"regime": "BULL", "vix": 16.0},
            score_fn=score_fn,
            path=path,
        )

    def test_logs_selected_and_vetoed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "obs.jsonl")
            n = self._run(path)
            self.assertEqual(n, 2)
            with open(path) as fh:
                recs = [json.loads(line) for line in fh]
            by_sym = {r["symbol"]: r for r in recs}
            self.assertTrue(by_sym["AAPL"]["extra"]["arm3_ai_selected"])
            self.assertEqual(by_sym["AAPL"]["extra"]["arm3_ai_confidence"], 0.9)
            self.assertEqual(by_sym["AAPL"]["extra"]["arm1_deterministic_rank"], 1)
            self.assertFalse(by_sym["T"]["extra"]["arm3_ai_selected"])
            self.assertIsNone(by_sym["T"]["extra"]["arm3_ai_confidence"])

    def test_failsafe_swallows_errors(self):
        def boom(_c):
            raise RuntimeError("score blew up")

        # An error anywhere in the build must not propagate (trading must never be blocked).
        self.assertEqual(self._run("/tmp/_unused_failsafe.jsonl", score_fn=boom), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

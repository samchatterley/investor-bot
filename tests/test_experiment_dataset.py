import unittest

from experiment.dataset import (
    AsOfExpectancy,
    assemble_row,
    blind_features,
    forward_r,
    split_tag,
)


class TestForwardR(unittest.TestCase):
    def test_long_return_in_atr_units(self):
        # entry 100, exit 104, atr 2 -> 2.0 R
        self.assertAlmostEqual(forward_r([100, 101, 102, 103, 104, 105], 0, 4, atr=2.0), 2.0)

    def test_short_direction(self):
        self.assertAlmostEqual(
            forward_r([100, 99, 98, 97, 96], 0, 4, atr=2.0, direction="short"), 2.0
        )

    def test_cost_subtracted(self):
        self.assertAlmostEqual(forward_r([100, 104], 0, 1, atr=2.0, cost_r=0.5), 1.5)

    def test_exit_beyond_series_returns_none(self):
        self.assertIsNone(forward_r([100, 101], 0, 5, atr=2.0))

    def test_nonpositive_atr_returns_none(self):
        self.assertIsNone(forward_r([100, 104], 0, 1, atr=0.0))

    def test_negative_entry_index_returns_none(self):
        self.assertIsNone(forward_r([100, 104], -1, 1, atr=2.0))


class TestAsOfExpectancy(unittest.TestCase):
    def _acc(self):
        acc = AsOfExpectancy(decay_window=2)
        # outcomes known on increasing dates
        acc.record("2020-01-01", "pead", "BULL", 1.0)
        acc.record("2020-02-01", "pead", "BULL", 2.0)
        acc.record("2020-03-01", "pead", "CHOP", -1.0)
        acc.record("2020-04-01", "momentum", "BULL", 0.5)
        return acc

    def test_point_in_time_filtering(self):
        acc = self._acc()
        # at 2020-02-15 only the first two pead outcomes are known
        self.assertAlmostEqual(acc.expanding_edge("pead", "2020-02-15"), 1.5)
        # at 2020-01-01 nothing is strictly before it
        self.assertIsNone(acc.expanding_edge("pead", "2020-01-01"))

    def test_expanding_uses_all_prior(self):
        acc = self._acc()
        # at 2020-12-01 all three pead outcomes count: (1+2-1)/3
        self.assertAlmostEqual(acc.expanding_edge("pead", "2020-12-01"), 2.0 / 3.0)

    def test_regime_edge(self):
        acc = self._acc()
        self.assertAlmostEqual(acc.regime_edge("pead", "BULL", "2020-12-01"), 1.5)
        self.assertAlmostEqual(acc.regime_edge("pead", "CHOP", "2020-12-01"), -1.0)

    def test_rolling_window(self):
        acc = self._acc()
        # decay_window=2 -> last two prior pead outcomes at 2020-12-01: (2 + -1)/2 = 0.5
        self.assertAlmostEqual(acc.rolling_edge("pead", "2020-12-01"), 0.5)

    def test_decay_is_rolling_minus_expanding(self):
        acc = self._acc()
        expanding = acc.expanding_edge("pead", "2020-12-01")
        rolling = acc.rolling_edge("pead", "2020-12-01")
        self.assertAlmostEqual(acc.decay("pead", "2020-12-01"), rolling - expanding)

    def test_none_when_no_prior(self):
        acc = self._acc()
        self.assertIsNone(acc.rolling_edge("pead", "2019-01-01"))
        self.assertIsNone(acc.regime_edge("pead", "BULL", "2019-01-01"))
        self.assertIsNone(acc.decay("pead", "2019-01-01"))

    def test_features_bundle(self):
        acc = self._acc()
        f = acc.features("pead", "BULL", "2020-12-01")
        self.assertEqual(
            set(f),
            {"signal_edge_expanding", "signal_edge_rolling", "signal_regime_edge", "signal_decay"},
        )
        self.assertAlmostEqual(f["signal_regime_edge"], 1.5)


class TestBlindFeatures(unittest.TestCase):
    def test_drops_identity_and_price(self):
        blinded = blind_features(
            {
                "symbol": "AAPL",
                "date": "2020-01-01",
                "current_price": 100.0,
                "entry_price": 99.0,
                "rsi_14": 31.0,
                "vol_ratio": 1.8,
            }
        )
        self.assertNotIn("symbol", blinded)
        self.assertNotIn("date", blinded)
        self.assertNotIn("current_price", blinded)
        self.assertNotIn("entry_price", blinded)  # *_price suffix dropped
        self.assertEqual(blinded, {"rsi_14": 31.0, "vol_ratio": 1.8})


class TestSplitTag(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(split_tag("2018-06-01"), "train")
        self.assertEqual(split_tag("2022-12-31"), "train")
        self.assertEqual(split_tag("2023-01-01"), "validation")
        self.assertEqual(split_tag("2023-12-31"), "validation")
        self.assertEqual(split_tag("2024-01-01"), "holdout")
        self.assertEqual(split_tag("2026-06-14"), "holdout")


class TestAssembleRow(unittest.TestCase):
    def test_packages_and_blinds(self):
        row = assemble_row(
            symbol="AAPL",
            date="2024-03-01",
            features={"rsi_14": 30.0, "current_price": 180.0},
            fired_signals=["pead"],
            material_context=["earnings_surprise_or_drift"],
            expectancy={"signal_edge_expanding": 0.2},
            forward_r_value=0.4,
        )
        self.assertEqual(row.split, "holdout")
        self.assertNotIn("symbol", row.blinded_features)
        self.assertNotIn("current_price", row.blinded_features)
        self.assertIn("rsi_14", row.blinded_features)
        self.assertEqual(row.forward_r, 0.4)
        self.assertEqual(row.fired_signals, ["pead"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

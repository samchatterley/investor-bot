"""Tests for data/index_membership.py — index_change detection from news headlines."""

import unittest

from data.index_membership import classify_index_change


class TestClassifyIndexChange(unittest.TestCase):
    def _detected(self, headline: str) -> bool:
        return classify_index_change([headline]) is not None

    # ── positives: membership changes across indices and phrasings ──────────────
    def test_added_to_sp500(self):
        self.assertTrue(self._detected("Acme Corp to be added to the S&P 500"))

    def test_join_sp500(self):
        self.assertTrue(self._detected("Beta Inc set to join the S&P 500 next week"))

    def test_joins_nasdaq100(self):
        self.assertTrue(self._detected("Gamma joins the Nasdaq-100 in latest reshuffle"))

    def test_removed_from_sp500(self):
        self.assertTrue(self._detected("Delta removed from the S&P 500 index"))

    def test_dropped_from_nasdaq100(self):
        self.assertTrue(self._detected("Epsilon dropped from the Nasdaq 100"))

    def test_replace_in_dow_jones(self):
        self.assertTrue(self._detected("Zeta to replace Eta in the Dow Jones Industrial Average"))

    def test_russell_addition(self):
        self.assertTrue(self._detected("Theta added to the Russell 2000 at reconstitution"))

    def test_index_inclusion_noun(self):
        self.assertTrue(self._detected("Index inclusion sends Iota shares higher"))

    def test_sp500_inclusion_noun(self):
        self.assertTrue(self._detected("S&P 500 inclusion expected to lift Kappa demand"))

    def test_returns_matched_headline(self):
        result = classify_index_change(["unrelated", "Lambda joins the S&P 500"])
        self.assertEqual(result, {"detected": True, "headline": "Lambda joins the S&P 500"})

    # ── negatives: generic index mentions must not fire ─────────────────────────
    def test_generic_index_move_not_detected(self):
        self.assertIsNone(classify_index_change(["S&P 500 hits a record high today"]))

    def test_index_as_market_backdrop_not_detected(self):
        self.assertIsNone(classify_index_change(["Mu stock rises as the S&P 500 gains 1%"]))

    def test_dow_falls_not_detected(self):
        self.assertIsNone(classify_index_change(["Dow Jones falls 300 points on inflation data"]))

    def test_company_adds_product_not_detected(self):
        # "adds" (non-membership) near no index, and bare product news, must not fire
        self.assertIsNone(classify_index_change(["Nu Corp adds a new product line this quarter"]))

    def test_empty_headlines_returns_none(self):
        self.assertIsNone(classify_index_change([]))

    def test_no_match_returns_none(self):
        self.assertIsNone(classify_index_change(["Xi beats earnings", "Omicron names new CFO"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

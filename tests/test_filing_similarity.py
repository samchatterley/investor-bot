"""Tests for data/filing_similarity.py — Lazy Prices consecutive-filing similarity."""

import unittest
from unittest.mock import patch

import data.filing_similarity as fs


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_text_is_one(self):
        t = "revenue increased materially because demand accelerated across segments worldwide"
        self.assertAlmostEqual(fs.cosine_similarity(t, t), 1.0, places=6)

    def test_disjoint_text_is_zero(self):
        self.assertEqual(
            fs.cosine_similarity("revenue increased demand accelerated", "litigation goodwill"), 0.0
        )

    def test_empty_text_is_zero(self):
        self.assertEqual(fs.cosine_similarity("", "revenue increased"), 0.0)
        self.assertEqual(fs.cosine_similarity("revenue increased", ""), 0.0)

    def test_all_stopword_text_is_zero(self):
        # Tokens are all stopwords/short → empty counter → 0.0 (covers the not-ca/cb branch).
        self.assertEqual(fs.cosine_similarity("the and for our", "revenue growth margins"), 0.0)

    def test_partial_overlap_between_zero_and_one(self):
        sim = fs.cosine_similarity(
            "revenue growth margins expanded strongly",
            "revenue growth declined sharply downward",
        )
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)


class TestTokens(unittest.TestCase):
    def test_filters_stopwords_and_short_words(self):
        toks = fs._tokens("the revenue is up and margins grew")
        self.assertIn("revenue", toks)
        self.assertIn("margins", toks)
        self.assertNotIn("the", toks)  # stopword
        self.assertNotIn("and", toks)  # stopword
        self.assertNotIn("is", toks)  # < 3 letters
        self.assertNotIn("up", toks)  # < 3 letters


class TestGetFilingChange(unittest.TestCase):
    def _filings(self):
        return [
            {"form": "10-K", "filing_date": "2026-02-15", "accession": "acc2", "doc": "d2.htm"},
            {"form": "10-K", "filing_date": "2025-02-15", "accession": "acc1", "doc": "d1.htm"},
        ]

    def test_none_when_cik_unknown(self):
        with patch.object(fs._ec, "_get_cik_map", return_value={}):
            self.assertIsNone(fs.get_filing_change("NOPE"))

    def test_none_when_fewer_than_two_filings(self):
        with (
            patch.object(fs._ec, "_get_cik_map", return_value={"AAA": "0000000001"}),
            patch.object(fs._ec, "_get_recent_filings", return_value=[self._filings()[0]]),
        ):
            self.assertIsNone(fs.get_filing_change("AAA"))

    def test_none_when_text_missing(self):
        with (
            patch.object(fs._ec, "_get_cik_map", return_value={"AAA": "0000000001"}),
            patch.object(fs._ec, "_get_recent_filings", return_value=self._filings()),
            patch.object(fs._ec, "_fetch_filing_text", side_effect=["", "some text here now"]),
        ):
            self.assertIsNone(fs.get_filing_change("AAA"))

    def test_high_change_flags_bearish(self):
        with (
            patch.object(fs._ec, "_get_cik_map", return_value={"AAA": "0000000001"}),
            patch.object(fs._ec, "_get_recent_filings", return_value=self._filings()),
            patch.object(
                fs._ec,
                "_fetch_filing_text",
                side_effect=[
                    "litigation goodwill impairment restructuring",
                    "revenue margins growth",
                ],
            ),
        ):
            rec = fs.get_filing_change("AAA")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["symbol"], "AAA")
        self.assertEqual(rec["latest_date"], "2026-02-15")
        self.assertEqual(rec["prior_date"], "2025-02-15")
        self.assertTrue(rec["lazy_prices_bearish"])  # disjoint → sim 0 → bearish
        self.assertAlmostEqual(rec["change_score"], 1.0, places=4)

    def test_copy_paste_not_bearish(self):
        same = "revenue grew margins expanded guidance reaffirmed across all operating segments"
        with (
            patch.object(fs._ec, "_get_cik_map", return_value={"AAA": "0000000001"}),
            patch.object(fs._ec, "_get_recent_filings", return_value=self._filings()),
            patch.object(fs._ec, "_fetch_filing_text", side_effect=[same, same]),
        ):
            rec = fs.get_filing_change("AAA")
        self.assertFalse(rec["lazy_prices_bearish"])  # identical → sim 1 → benign
        self.assertEqual(rec["similarity"], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

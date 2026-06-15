"""Tests for utils/symbols.py — ticker normalisation."""

import unittest

from utils.symbols import to_yf_symbol


class TestToYfSymbol(unittest.TestCase):
    def test_dot_class_share_becomes_hyphen(self):
        self.assertEqual(to_yf_symbol("BRK.B"), "BRK-B")
        self.assertEqual(to_yf_symbol("BF.B"), "BF-B")

    def test_plain_ticker_unchanged(self):
        self.assertEqual(to_yf_symbol("AAPL"), "AAPL")

    def test_index_caret_unchanged(self):
        self.assertEqual(to_yf_symbol("^VIX"), "^VIX")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

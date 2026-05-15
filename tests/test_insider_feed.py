"""Tests for data/insider_feed.py — SEC EDGAR Form 4 insider purchase detection."""

import unittest
from unittest.mock import MagicMock, patch

from data.insider_feed import (
    _get_cik_map,
    _parse_form4,
    _recent_form4_filings,
    get_insider_activity,
)

_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

_SUBMISSIONS_JSON = {
    "filings": {
        "recent": {
            "form": ["4", "4", "10-K"],
            "filingDate": ["2026-05-08", "2026-05-07", "2026-01-15"],
            "accessionNumber": [
                "0000320193-26-000001",
                "0000320193-26-000002",
                "0000320193-26-000010",
            ],
            "primaryDocument": ["form4.xml", "form4.xml", "10k.htm"],
        }
    }
}

_FORM4_XML_SINGLE = b"""<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerName>Tim Cook</reportingOwnerName>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-08</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>185.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

_FORM4_XML_SALE = b"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner><reportingOwnerName>Luca Maestri</reportingOwnerName></reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-07</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>186.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

_FORM4_XML_OPTION = b"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner><reportingOwnerName>Jeff Williams</reportingOwnerName></reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-08</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>2000</value></transactionShares>
        <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def _mock_response(content=None, json_data=None, status=200):
    m = MagicMock()
    m.status_code = status
    m.raise_for_status = MagicMock()
    if json_data is not None:
        m.json.return_value = json_data
    if content is not None:
        m.content = content
        m.text = content.decode("utf-8")
    return m


class TestGetCikMap(unittest.TestCase):
    def setUp(self):
        _get_cik_map.cache_clear()

    def tearDown(self):
        _get_cik_map.cache_clear()

    def test_returns_ticker_to_cik_mapping(self):
        with patch("data.insider_feed.requests.get") as mock_get:
            mock_get.return_value = _mock_response(json_data=_TICKERS_JSON)
            result = _get_cik_map()
        self.assertEqual(result["AAPL"], "0000320193")
        self.assertEqual(result["MSFT"], "0000789019")

    def test_returns_empty_on_network_failure(self):
        with patch("data.insider_feed.requests.get", side_effect=Exception("network error")):
            result = _get_cik_map()
        self.assertEqual(result, {})

    def test_cik_zero_padded_to_10_digits(self):
        data = {"0": {"cik_str": 1, "ticker": "X", "title": ""}}
        with patch("data.insider_feed.requests.get") as mock_get:
            mock_get.return_value = _mock_response(json_data=data)
            result = _get_cik_map()
        self.assertEqual(result["X"], "0000000001")


class TestRecentForm4Filings(unittest.TestCase):
    def test_returns_only_form4_within_window(self):
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(json_data=_SUBMISSIONS_JSON)
            result = _recent_form4_filings("0000320193", lookback_days=30)
        # 10-K excluded; 2 Form 4s within 30 days
        self.assertEqual(len(result), 2)
        for f in result:
            self.assertIn("accession", f)
            self.assertIn("doc", f)

    def test_returns_empty_on_network_failure(self):
        with (
            patch("data.insider_feed.requests.get", side_effect=Exception("timeout")),
            patch("data.insider_feed.time.sleep"),
        ):
            result = _recent_form4_filings("0000320193", lookback_days=30)
        self.assertEqual(result, [])

    def test_accession_dashes_stripped(self):
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(json_data=_SUBMISSIONS_JSON)
            result = _recent_form4_filings("0000320193", lookback_days=30)
        for f in result:
            self.assertNotIn("-", f["accession"])


class TestParseForm4(unittest.TestCase):
    def test_extracts_open_market_purchase(self):
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(content=_FORM4_XML_SINGLE)
            txns = _parse_form4("320193", "000032019326000001", "form4.xml")
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0]["reporter"], "Tim Cook")
        self.assertAlmostEqual(txns[0]["shares"], 5000.0)
        self.assertAlmostEqual(txns[0]["price"], 185.50)

    def test_excludes_sales(self):
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(content=_FORM4_XML_SALE)
            txns = _parse_form4("320193", "000032019326000002", "form4.xml")
        self.assertEqual(txns, [])

    def test_excludes_option_exercises(self):
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(content=_FORM4_XML_OPTION)
            txns = _parse_form4("320193", "000032019326000003", "form4.xml")
        self.assertEqual(txns, [])

    def test_returns_empty_on_xml_parse_failure(self):
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(content=b"<not valid xml")
            txns = _parse_form4("320193", "bad", "form4.xml")
        self.assertEqual(txns, [])

    def test_returns_empty_on_network_failure(self):
        with (
            patch("data.insider_feed.requests.get", side_effect=Exception("timeout")),
            patch("data.insider_feed.time.sleep"),
        ):
            txns = _parse_form4("320193", "xxx", "form4.xml")
        self.assertEqual(txns, [])


class TestGetInsiderActivity(unittest.TestCase):
    def setUp(self):
        _get_cik_map.cache_clear()

    def tearDown(self):
        _get_cik_map.cache_clear()

    def _mock_two_insiders(self):
        xml_second = b"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner><reportingOwnerName>Luca Maestri</reportingOwnerName></reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-07</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>3000</value></transactionShares>
        <transactionPricePerShare><value>186.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        return [
            _mock_response(json_data=_TICKERS_JSON),
            _mock_response(json_data=_SUBMISSIONS_JSON),
            _mock_response(content=_FORM4_XML_SINGLE),
            _mock_response(content=xml_second),
        ]

    def test_cluster_true_when_two_insiders(self):
        with (
            patch("data.insider_feed.requests.get", side_effect=self._mock_two_insiders()),
            patch("data.insider_feed.time.sleep"),
        ):
            result = get_insider_activity(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertTrue(result["AAPL"]["insider_cluster"])
        self.assertEqual(result["AAPL"]["insider_unique_insiders"], 2)

    def test_cluster_false_when_single_insider(self):
        single_submissions = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "filingDate": ["2026-05-08"],
                    "accessionNumber": ["0000320193-26-000001"],
                    "primaryDocument": ["form4.xml"],
                }
            }
        }
        with (
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=single_submissions),
                    _mock_response(content=_FORM4_XML_SINGLE),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
        ):
            result = get_insider_activity(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertFalse(result["AAPL"]["insider_cluster"])

    def test_large_buy_flag_set_above_threshold(self):
        # 5000 shares × $185.50 = $927,500 > $100k
        single_submissions = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "filingDate": ["2026-05-08"],
                    "accessionNumber": ["0000320193-26-000001"],
                    "primaryDocument": ["form4.xml"],
                }
            }
        }
        with (
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=single_submissions),
                    _mock_response(content=_FORM4_XML_SINGLE),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
        ):
            result = get_insider_activity(["AAPL"])
        self.assertTrue(result["AAPL"]["insider_large_buy"])

    def test_returns_empty_for_unknown_symbol(self):
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(json_data=_TICKERS_JSON)
            result = get_insider_activity(["UNKNOWN_TICKER_XYZ"])
        self.assertEqual(result, {})

    def test_returns_empty_dict_on_full_network_failure(self):
        with (
            patch("data.insider_feed.requests.get", side_effect=Exception("network down")),
            patch("data.insider_feed.time.sleep"),
        ):
            result = get_insider_activity(["AAPL"])
        self.assertEqual(result, {})

    def test_symbol_absent_when_no_form4_in_window(self):
        no_filings = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2026-01-15"],
                    "accessionNumber": ["0000320193-26-000010"],
                    "primaryDocument": ["10k.htm"],
                }
            }
        }
        with (
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=no_filings),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
        ):
            result = get_insider_activity(["AAPL"])
        self.assertNotIn("AAPL", result)


class TestRecentForm4FilingsDateParseError(unittest.TestCase):
    def test_skips_filing_with_invalid_date(self):
        # Lines 70-71: `except ValueError: continue` when filing_date is not valid ISO
        bad_dates_submissions = {
            "filings": {
                "recent": {
                    "form": ["4", "4"],
                    "filingDate": ["not-a-date", "2026-05-08"],
                    "accessionNumber": [
                        "0000320193-26-000099",
                        "0000320193-26-000001",
                    ],
                    "primaryDocument": ["form4.xml", "form4.xml"],
                }
            }
        }
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(json_data=bad_dates_submissions)
            result = _recent_form4_filings("0000320193", lookback_days=30)
        # The bad-date entry is skipped; only the valid one is included
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["accession"], "000032019326000001")


class TestParseForm4InvalidPrice(unittest.TestCase):
    def test_skips_transaction_with_non_numeric_price(self):
        # Lines 112-113: `except (ValueError, TypeError): continue` when price is not numeric
        xml_bad_price = b"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner><reportingOwnerName>Tim Cook</reportingOwnerName></reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-08</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>not-a-number</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(content=xml_bad_price)
            txns = _parse_form4("320193", "000032019326000001", "form4.xml")
        # Transaction skipped due to ValueError on float("not-a-number")
        self.assertEqual(txns, [])

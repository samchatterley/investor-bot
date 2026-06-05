"""Tests for data/insider_feed.py — SEC EDGAR Form 4 insider purchase detection."""

import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import data.insider_feed as _insider_mod
from data.insider_feed import (
    _edgar_sleep,
    _get_cik_map,
    _load_cache,
    _parse_form4,
    _recent_form4_filings,
    _save_cache,
    get_insider_activity,
    prefetch_insider_activity,
)

# Dynamic dates — always within the 10-day default lookback window
_D1 = (date.today() - timedelta(days=1)).isoformat()
_D2 = (date.today() - timedelta(days=2)).isoformat()

_TODAY = date(2026, 6, 3)

_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

_SUBMISSIONS_JSON = {
    "filings": {
        "recent": {
            "form": ["4", "4", "10-K"],
            "filingDate": [_D1, _D2, "2026-01-15"],
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
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache"),
            patch("data.insider_feed.requests.get", side_effect=self._mock_two_insiders()),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
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
                    "filingDate": [_D1],
                    "accessionNumber": ["0000320193-26-000001"],
                    "primaryDocument": ["form4.xml"],
                }
            }
        }
        with (
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache"),
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=single_submissions),
                    _mock_response(content=_FORM4_XML_SINGLE),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
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
                    "filingDate": [_D1],
                    "accessionNumber": ["0000320193-26-000001"],
                    "primaryDocument": ["form4.xml"],
                }
            }
        }
        with (
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache"),
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=single_submissions),
                    _mock_response(content=_FORM4_XML_SINGLE),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            result = get_insider_activity(["AAPL"])
        self.assertTrue(result["AAPL"]["insider_large_buy"])

    def test_returns_empty_for_unknown_symbol(self):
        with (
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache"),
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            mock_get.return_value = _mock_response(json_data=_TICKERS_JSON)
            result = get_insider_activity(["UNKNOWN_TICKER_XYZ"])
        self.assertEqual(result, {})

    def test_returns_empty_dict_on_full_network_failure(self):
        with (
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache"),
            patch("data.insider_feed.requests.get", side_effect=Exception("network down")),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
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
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache"),
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=no_filings),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            result = get_insider_activity(["AAPL"])
        self.assertNotIn("AAPL", result)


class TestGetInsiderActivityCache(unittest.TestCase):
    """Tests for the same-day caching behaviour of get_insider_activity."""

    def setUp(self):
        _get_cik_map.cache_clear()

    def tearDown(self):
        _get_cik_map.cache_clear()

    def test_cache_hit_returns_without_network_call(self):
        today_key = _TODAY.isoformat()
        cached_data = {
            "insider_cluster": True,
            "insider_unique_insiders": 2,
            "insider_transaction_count": 3,
            "insider_total_shares": 8000.0,
            "insider_large_buy": True,
        }
        warm_cache = {today_key: {"AAPL": cached_data}}
        with (
            patch("data.insider_feed._load_cache", return_value=warm_cache),
            patch("data.insider_feed._save_cache") as mock_save,
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            result = get_insider_activity(["AAPL"])
        mock_get.assert_not_called()
        mock_save.assert_not_called()
        self.assertEqual(result["AAPL"], cached_data)

    def test_null_sentinel_omits_symbol_from_result(self):
        # None in cache means "fetched, no activity" — symbol must not appear in result
        today_key = _TODAY.isoformat()
        warm_cache = {today_key: {"AAPL": None}}
        with (
            patch("data.insider_feed._load_cache", return_value=warm_cache),
            patch("data.insider_feed._save_cache"),
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            result = get_insider_activity(["AAPL"])
        mock_get.assert_not_called()
        self.assertNotIn("AAPL", result)

    def test_partial_cache_hit_only_fetches_missing(self):
        # AAPL already cached (with data), MSFT missing → only MSFT should be fetched
        today_key = _TODAY.isoformat()
        aapl_data = {
            "insider_cluster": False,
            "insider_unique_insiders": 1,
            "insider_transaction_count": 1,
            "insider_total_shares": 1000.0,
            "insider_large_buy": False,
        }
        partial_cache = {today_key: {"AAPL": aapl_data}}
        no_filings = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2026-01-15"],
                    "accessionNumber": ["0000789019-26-000010"],
                    "primaryDocument": ["10k.htm"],
                }
            }
        }
        with (
            patch("data.insider_feed._load_cache", return_value=partial_cache),
            patch("data.insider_feed._save_cache") as mock_save,
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=no_filings),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            result = get_insider_activity(["AAPL", "MSFT"])
        # AAPL returned from cache; MSFT fetched but had no activity
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)
        mock_save.assert_called_once()

    def test_cache_miss_saves_after_fetch(self):
        with (
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache") as mock_save,
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(
                        json_data={
                            "filings": {
                                "recent": {
                                    "form": [],
                                    "filingDate": [],
                                    "accessionNumber": [],
                                    "primaryDocument": [],
                                }
                            }
                        }
                    ),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            get_insider_activity(["AAPL"])
        mock_save.assert_called_once()
        saved_arg = mock_save.call_args[0][0]
        self.assertIn(_TODAY.isoformat(), saved_arg)


class TestPrefetchInsiderActivity(unittest.TestCase):
    def setUp(self):
        _get_cik_map.cache_clear()

    def tearDown(self):
        _get_cik_map.cache_clear()

    def test_returns_count_of_symbols_fetched(self):
        no_filings = {
            "filings": {
                "recent": {
                    "form": [],
                    "filingDate": [],
                    "accessionNumber": [],
                    "primaryDocument": [],
                }
            }
        }
        with (
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache"),
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=no_filings),
                    _mock_response(json_data=no_filings),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            n = prefetch_insider_activity(["AAPL", "MSFT"])
        self.assertEqual(n, 2)

    def test_warm_cache_returns_zero(self):
        today_key = _TODAY.isoformat()
        warm_cache = {today_key: {"AAPL": None, "MSFT": None}}
        with (
            patch("data.insider_feed._load_cache", return_value=warm_cache),
            patch("data.insider_feed._save_cache") as mock_save,
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            n = prefetch_insider_activity(["AAPL", "MSFT"])
        self.assertEqual(n, 0)
        mock_get.assert_not_called()
        mock_save.assert_not_called()

    def test_uses_stock_universe_when_symbols_none(self):
        with (
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache"),
            patch("data.insider_feed._live_fetch", return_value={}) as mock_fetch,
            patch("data.insider_feed.today_et", return_value=_TODAY),
            patch("data.insider_feed.STOCK_UNIVERSE", {"AAPL", "MSFT"}),
        ):
            prefetch_insider_activity()
        fetched_syms = set(mock_fetch.call_args[0][0])
        self.assertEqual(fetched_syms, {"AAPL", "MSFT"})

    def test_saves_today_key_to_cache(self):
        no_filings = {
            "filings": {
                "recent": {
                    "form": [],
                    "filingDate": [],
                    "accessionNumber": [],
                    "primaryDocument": [],
                }
            }
        }
        with (
            patch("data.insider_feed._load_cache", return_value={}),
            patch("data.insider_feed._save_cache") as mock_save,
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=no_filings),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            prefetch_insider_activity(["AAPL"])
        saved_arg = mock_save.call_args[0][0]
        self.assertIn(_TODAY.isoformat(), saved_arg)

    def test_discards_stale_date_keys(self):
        # Previous day's cache should be replaced, not merged
        yesterday_key = (date(2026, 6, 2)).isoformat()
        stale_cache = {yesterday_key: {"AAPL": None}}
        no_filings = {
            "filings": {
                "recent": {
                    "form": [],
                    "filingDate": [],
                    "accessionNumber": [],
                    "primaryDocument": [],
                }
            }
        }
        with (
            patch("data.insider_feed._load_cache", return_value=stale_cache),
            patch("data.insider_feed._save_cache") as mock_save,
            patch(
                "data.insider_feed.requests.get",
                side_effect=[
                    _mock_response(json_data=_TICKERS_JSON),
                    _mock_response(json_data=no_filings),
                ],
            ),
            patch("data.insider_feed.time.sleep"),
            patch("data.insider_feed.today_et", return_value=_TODAY),
        ):
            prefetch_insider_activity(["AAPL"])
        saved_arg = mock_save.call_args[0][0]
        self.assertNotIn(yesterday_key, saved_arg)
        self.assertIn(_TODAY.isoformat(), saved_arg)


class TestLoadSaveCache(unittest.TestCase):
    def test_load_returns_empty_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_returns_empty_on_json_error(self):
        import json as _json
        from unittest.mock import mock_open

        with (
            patch("builtins.open", mock_open(read_data="not valid json")),
            patch("data.insider_feed.json.load", side_effect=_json.JSONDecodeError("err", "", 0)),
        ):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_save_writes_json_on_success(self):
        from unittest.mock import mock_open

        m = mock_open()
        with (
            patch("data.insider_feed.os.makedirs"),
            patch("builtins.open", m),
            patch("data.insider_feed.json.dump") as mock_dump,
        ):
            _save_cache({"2026-06-03": {"AAPL": None}})
        mock_dump.assert_called_once()

    def test_save_logs_warning_on_os_error(self):
        with (
            patch("data.insider_feed.os.makedirs"),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            _save_cache({"2026-06-03": {}})  # should not raise


class TestRecentForm4FilingsCutoff(unittest.TestCase):
    def test_breaks_at_filing_before_cutoff(self):
        # Form 4 with date older than the lookback window → break, not included
        cutoff_submissions = {
            "filings": {
                "recent": {
                    "form": ["4", "4"],
                    "filingDate": [_D1, "2020-01-01"],  # second is ancient
                    "accessionNumber": ["0000320193-26-000001", "0000320193-20-000001"],
                    "primaryDocument": ["form4.xml", "form4.xml"],
                }
            }
        }
        with (
            patch("data.insider_feed.requests.get") as mock_get,
            patch("data.insider_feed.time.sleep"),
        ):
            mock_get.return_value = _mock_response(json_data=cutoff_submissions)
            result = _recent_form4_filings("0000320193", lookback_days=10)
        # Only the recent filing is included; the ancient one triggers break
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["accession"], "000032019326000001")


class TestRecentForm4FilingsDateParseError(unittest.TestCase):
    def test_skips_filing_with_invalid_date(self):
        # Lines 70-71: `except ValueError: continue` when filing_date is not valid ISO
        bad_dates_submissions = {
            "filings": {
                "recent": {
                    "form": ["4", "4"],
                    "filingDate": ["not-a-date", _D1],
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


class TestEdgarSleep(unittest.TestCase):
    def setUp(self):
        _insider_mod._last_req_time = 0.0

    def tearDown(self):
        _insider_mod._last_req_time = 0.0

    def test_sleeps_when_last_request_was_recent(self):
        with (
            patch("data.insider_feed.time.monotonic", side_effect=[100.05, 100.10]),
            patch("data.insider_feed.time.sleep") as mock_sleep,
        ):
            _insider_mod._last_req_time = 100.0
            _edgar_sleep()
        # gap = 0.15 - (100.05 - 100.0) = 0.10 > 0 → sleep called
        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 0.10, places=5)

    def test_no_sleep_when_last_request_was_stale(self):
        with patch("data.insider_feed.time.sleep") as mock_sleep:
            _insider_mod._last_req_time = 0.0  # far in the past
            _edgar_sleep()
        mock_sleep.assert_not_called()

    def test_updates_last_req_time_after_call(self):
        before = _insider_mod._last_req_time
        with patch("data.insider_feed.time.sleep"):
            _edgar_sleep()
        self.assertGreater(_insider_mod._last_req_time, before)

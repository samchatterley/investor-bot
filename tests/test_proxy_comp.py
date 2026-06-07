"""Tests for data/proxy_comp.py — DEF 14A executive compensation fetcher."""

import json
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

import data.proxy_comp as _pc
from data.proxy_comp import (
    _extract_names_and_totals,
    _fetch_compensation,
    _find_def14a,
    _is_fresh,
    _load_cache,
    _normalize_name,
    _parse_comp_table,
    _save_cache,
    get_exec_compensation,
    match_compensation,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_COMP_TABLE_HTML = """
<html><body>
<p>SUMMARY COMPENSATION TABLE</p>
<table>
  <tr><th>Name and Principal Position</th><th>Year</th><th>Salary ($)</th><th>Total ($)</th></tr>
  <tr><td>Timothy D. Cook Chief Executive Officer</td><td>2024</td><td>3,000,000</td><td>63,208,000</td></tr>
  <tr><td>Luca Maestri Senior VP</td><td>2024</td><td>1,000,000</td><td>27,200,000</td></tr>
</table>
</body></html>
"""

_NO_COMP_TABLE_HTML = """
<html><body><p>No compensation table here.</p><table><tr><td>data</td></tr></table></body></html>
"""

_SUBMISSIONS_RESPONSE = {
    "filings": {
        "recent": {
            "form": ["10-K", "DEF 14A", "8-K"],
            "filingDate": ["2025-02-01", "2025-04-15", "2025-05-01"],
            "accessionNumber": [
                "0000320193-25-000010",
                "0000320193-25-000020",
                "0000320193-25-000030",
            ],
            "primaryDocument": ["10k.htm", "proxy2025.htm", "8k.htm"],
        }
    }
}

_CIK = "0000320193"


# ── _edgar_sleep ─────────────────────────────────────────────────────────────


class TestEdgarSleep(unittest.TestCase):
    def test_sleeps_when_gap_positive(self):
        _pc._last_req = 0.0
        with (
            patch("data.proxy_comp.time.monotonic", side_effect=[0.05, 0.20]),
            patch("data.proxy_comp.time.sleep") as mock_sleep,
        ):
            _pc._edgar_sleep()
        mock_sleep.assert_called_once()

    def test_no_sleep_when_gap_not_positive(self):
        _pc._last_req = 0.0
        with (
            patch("data.proxy_comp.time.monotonic", side_effect=[10.0, 10.15]),
            patch("data.proxy_comp.time.sleep") as mock_sleep,
        ):
            _pc._edgar_sleep()
        mock_sleep.assert_not_called()


# ── _normalize_name ───────────────────────────────────────────────────────────


class TestNormalizeName(unittest.TestCase):
    def test_strips_punctuation(self):
        self.assertEqual(_normalize_name("Timothy D. Cook"), "TIMOTHY D COOK")

    def test_uppercases(self):
        self.assertEqual(_normalize_name("luca maestri"), "LUCA MAESTRI")

    def test_strips_titles(self):
        self.assertEqual(_normalize_name("Mr. John Smith Jr."), "JOHN SMITH")

    def test_collapses_whitespace(self):
        self.assertEqual(_normalize_name("  Jane   Doe  "), "JANE DOE")

    def test_already_normalised(self):
        self.assertEqual(_normalize_name("COOK TIMOTHY D"), "COOK TIMOTHY D")

    def test_empty_string(self):
        self.assertEqual(_normalize_name(""), "")


# ── _extract_names_and_totals ─────────────────────────────────────────────────


class TestExtractNamesAndTotals(unittest.TestCase):
    def _make_table(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser").find("table")

    def test_extracts_name_and_total(self):
        soup = self._make_table("""
        <table>
          <tr><th>Name</th><th>Salary</th><th>Total ($)</th></tr>
          <tr><td>Jane Doe</td><td>500,000</td><td>1,200,000</td></tr>
        </table>""")
        result = _extract_names_and_totals(soup)
        self.assertIn("JANE DOE", result)
        self.assertAlmostEqual(result["JANE DOE"], 1200000.0)

    def test_returns_empty_when_no_total_column(self):
        soup = self._make_table("""
        <table>
          <tr><th>Name</th><th>Salary</th></tr>
          <tr><td>Jane Doe</td><td>500,000</td></tr>
        </table>""")
        self.assertEqual(_extract_names_and_totals(soup), {})

    def test_returns_empty_for_empty_table(self):
        soup = self._make_table("<table></table>")
        self.assertEqual(_extract_names_and_totals(soup), {})

    def test_skips_rows_without_amount(self):
        soup = self._make_table("""
        <table>
          <tr><th>Name</th><th>Total ($)</th></tr>
          <tr><td>Jane Doe</td><td>N/A</td></tr>
          <tr><td>John Smith</td><td>500,000</td></tr>
        </table>""")
        result = _extract_names_and_totals(soup)
        self.assertNotIn("JANE DOE", result)
        self.assertIn("JOHN SMITH", result)

    def test_skips_rows_with_zero_total(self):
        soup = self._make_table("""
        <table>
          <tr><th>Name</th><th>Total ($)</th></tr>
          <tr><td>Ghost Exec</td><td>0</td></tr>
        </table>""")
        self.assertEqual(_extract_names_and_totals(soup), {})

    def test_skips_rows_with_empty_name(self):
        soup = self._make_table("""
        <table>
          <tr><th>Name</th><th>Total ($)</th></tr>
          <tr><td></td><td>500,000</td></tr>
          <tr><td>Jane Doe</td><td>800,000</td></tr>
        </table>""")
        result = _extract_names_and_totals(soup)
        self.assertNotIn("", result)
        self.assertIn("JANE DOE", result)

    def test_skips_rows_with_insufficient_columns(self):
        soup = self._make_table("""
        <table>
          <tr><th>Name</th><th>Salary</th><th>Total ($)</th></tr>
          <tr><td>Short</td></tr>
          <tr><td>Jane Doe</td><td>500,000</td><td>1,000,000</td></tr>
        </table>""")
        result = _extract_names_and_totals(soup)
        self.assertIn("JANE DOE", result)
        self.assertNotIn("SHORT", result)

    def test_skips_rows_where_normalised_name_too_short(self):
        soup = self._make_table("""
        <table>
          <tr><th>Name</th><th>Total ($)</th></tr>
          <tr><td>AB</td><td>500,000</td></tr>
          <tr><td>Jane Doe</td><td>800,000</td></tr>
        </table>""")
        result = _extract_names_and_totals(soup)
        self.assertNotIn("AB", result)
        self.assertIn("JANE DOE", result)

    def test_total_col_in_later_header_row(self):
        soup = self._make_table("""
        <table>
          <tr><th>Name</th><th colspan="3">Awards</th></tr>
          <tr><th>Name</th><th>Stock</th><th>Options</th><th>Total ($)</th></tr>
          <tr><td>Jane Doe</td><td>100</td><td>200</td><td>999,000</td></tr>
        </table>""")
        result = _extract_names_and_totals(soup)
        self.assertIn("JANE DOE", result)


# ── _parse_comp_table ─────────────────────────────────────────────────────────


class TestParseCompTable(unittest.TestCase):
    def test_finds_table_after_heading(self):
        soup = BeautifulSoup(_COMP_TABLE_HTML, "html.parser")
        result = _parse_comp_table(soup)
        self.assertIn("TIMOTHY D COOK CHIEF EXECUTIVE OFFICER", result)
        self.assertIn("LUCA MAESTRI SENIOR VP", result)

    def test_returns_empty_when_no_heading(self):
        soup = BeautifulSoup(_NO_COMP_TABLE_HTML, "html.parser")
        self.assertEqual(_parse_comp_table(soup), {})

    def test_tries_next_match_when_first_table_empty(self):
        html = """<html><body>
        <p>Summary Compensation Table</p>
        <table><tr><td>toc entry</td></tr></table>
        <p>Summary Compensation Table</p>
        <table>
          <tr><th>Name</th><th>Total ($)</th></tr>
          <tr><td>Real Exec</td><td>5,000,000</td></tr>
        </table>
        </body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_comp_table(soup)
        self.assertIn("REAL EXEC", result)

    def test_returns_empty_when_no_table_follows_heading(self):
        html = "<html><body><p>Summary Compensation Table</p><p>No table.</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        self.assertEqual(_parse_comp_table(soup), {})

    def test_skips_text_node_with_no_parent(self):
        # A NavigableString with parent=None is edge-case handled by the continue guard
        soup = BeautifulSoup(_COMP_TABLE_HTML, "html.parser")
        strings = soup.find_all(
            string=lambda t: "summary compensation table" in t.lower() if t else False
        )
        if strings:
            strings[0].extract()  # remove the node entirely from the tree
        # Should not raise; result is empty or partial
        result = _parse_comp_table(soup)
        self.assertIsInstance(result, dict)


# ── _find_def14a ──────────────────────────────────────────────────────────────


class TestFindDef14a(unittest.TestCase):
    def test_returns_accession_and_doc(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _SUBMISSIONS_RESPONSE
        with (
            patch("data.proxy_comp.requests.get", return_value=mock_resp),
            patch("data.proxy_comp._edgar_sleep"),
        ):
            result = _find_def14a(_CIK)
        self.assertEqual(result, ("000032019325000020", "proxy2025.htm"))

    def test_returns_none_when_no_def14a(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2025-02-01"],
                    "accessionNumber": ["0000320193-25-000010"],
                    "primaryDocument": ["10k.htm"],
                }
            }
        }
        with (
            patch("data.proxy_comp.requests.get", return_value=mock_resp),
            patch("data.proxy_comp._edgar_sleep"),
        ):
            result = _find_def14a(_CIK)
        self.assertIsNone(result)

    def test_returns_none_on_network_error(self):
        with (
            patch("data.proxy_comp.requests.get", side_effect=Exception("timeout")),
            patch("data.proxy_comp._edgar_sleep"),
        ):
            result = _find_def14a(_CIK)
        self.assertIsNone(result)


# ── _fetch_compensation ───────────────────────────────────────────────────────


class TestFetchCompensation(unittest.TestCase):
    def test_returns_data_on_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _SUBMISSIONS_RESPONSE
        mock_html = MagicMock()
        mock_html.text = _COMP_TABLE_HTML

        def _get(url, **kw):
            if "submissions" in url:
                return mock_resp
            return mock_html

        with (
            patch("data.proxy_comp.requests.get", side_effect=_get),
            patch("data.proxy_comp._edgar_sleep"),
        ):
            result = _fetch_compensation(_CIK)
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)

    def test_returns_empty_when_no_def14a(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
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
            patch("data.proxy_comp.requests.get", return_value=mock_resp),
            patch("data.proxy_comp._edgar_sleep"),
        ):
            result = _fetch_compensation(_CIK)
        self.assertEqual(result, {})

    def test_returns_empty_on_html_download_error(self):
        mock_submissions = MagicMock()
        mock_submissions.json.return_value = _SUBMISSIONS_RESPONSE

        call_count = 0

        def _get(url, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_submissions
            raise Exception("network error")

        with (
            patch("data.proxy_comp.requests.get", side_effect=_get),
            patch("data.proxy_comp._edgar_sleep"),
        ):
            result = _fetch_compensation(_CIK)
        self.assertEqual(result, {})

    def test_returns_empty_on_parse_error(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _SUBMISSIONS_RESPONSE
        mock_html = MagicMock()
        mock_html.text = _COMP_TABLE_HTML

        def _get(url, **kw):
            if "submissions" in url:
                return mock_resp
            return mock_html

        with (
            patch("data.proxy_comp.requests.get", side_effect=_get),
            patch("data.proxy_comp._edgar_sleep"),
            patch("data.proxy_comp.BeautifulSoup", side_effect=Exception("parse crash")),
        ):
            result = _fetch_compensation(_CIK)
        self.assertEqual(result, {})


# ── _load_cache / _save_cache / _is_fresh ─────────────────────────────────────


class TestCacheHelpers(unittest.TestCase):
    def test_load_cache_returns_empty_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            self.assertEqual(_load_cache(), {})

    def test_load_cache_returns_empty_on_bad_json(self):
        with patch("builtins.open", unittest.mock.mock_open(read_data="not json")):
            self.assertEqual(_load_cache(), {})

    def test_load_cache_returns_data(self):
        data = {"cik": {"fetched": "2026-01-01", "data": {}}}
        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(data))):
            self.assertEqual(_load_cache(), data)

    def test_save_cache_writes_json(self):
        m = unittest.mock.mock_open()
        with (
            patch("builtins.open", m),
            patch("os.makedirs"),
        ):
            _save_cache({"k": "v"})
        written = "".join(c.args[0] for c in m().write.call_args_list)
        self.assertIn('"k"', written)

    def test_save_cache_logs_on_os_error(self):
        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            patch("os.makedirs"),
            patch("data.proxy_comp.logger") as mock_log,
        ):
            _save_cache({})
        mock_log.warning.assert_called_once()

    def test_is_fresh_true_within_ttl(self):
        recent = (date.today() - timedelta(days=10)).isoformat()
        self.assertTrue(_is_fresh({"fetched": recent}))

    def test_is_fresh_false_beyond_ttl(self):
        old = (date.today() - timedelta(days=100)).isoformat()
        self.assertFalse(_is_fresh({"fetched": old}))

    def test_is_fresh_false_on_missing_key(self):
        self.assertFalse(_is_fresh({}))

    def test_is_fresh_false_on_bad_date(self):
        self.assertFalse(_is_fresh({"fetched": "not-a-date"}))


# ── get_exec_compensation ─────────────────────────────────────────────────────


class TestGetExecCompensation(unittest.TestCase):
    def test_returns_cached_data_when_fresh(self):
        recent = date.today().isoformat()
        cached = {_CIK: {"fetched": recent, "data": {"TIMOTHY D COOK": 63208000.0}}}
        with patch("data.proxy_comp._load_cache", return_value=cached):
            result = get_exec_compensation(_CIK)
        self.assertEqual(result, {"TIMOTHY D COOK": 63208000.0})

    def test_fetches_and_caches_when_stale(self):
        old = (date.today() - timedelta(days=200)).isoformat()
        stale_cache = {_CIK: {"fetched": old, "data": {}}}
        comp_data = {"JANE DOE": 5000000.0}
        with (
            patch("data.proxy_comp._load_cache", return_value=stale_cache),
            patch("data.proxy_comp._fetch_compensation", return_value=comp_data),
            patch("data.proxy_comp._save_cache") as mock_save,
        ):
            result = get_exec_compensation(_CIK)
        self.assertEqual(result, comp_data)
        mock_save.assert_called_once()

    def test_caches_empty_dict_when_fetch_fails(self):
        with (
            patch("data.proxy_comp._load_cache", return_value={}),
            patch("data.proxy_comp._fetch_compensation", return_value={}),
            patch("data.proxy_comp._save_cache") as mock_save,
        ):
            result = get_exec_compensation(_CIK)
        self.assertEqual(result, {})
        mock_save.assert_called_once()

    def test_returns_empty_for_missing_cik_in_cache(self):
        recent = date.today().isoformat()
        cached = {"OTHER": {"fetched": recent, "data": {"NAME": 1.0}}}
        with (
            patch("data.proxy_comp._load_cache", return_value=cached),
            patch("data.proxy_comp._fetch_compensation", return_value={}),
            patch("data.proxy_comp._save_cache"),
        ):
            result = get_exec_compensation(_CIK)
        self.assertEqual(result, {})


# ── match_compensation ────────────────────────────────────────────────────────


class TestMatchCompensation(unittest.TestCase):
    _COMP = {"TIMOTHY D COOK": 63208000.0, "LUCA MAESTRI": 27200000.0}

    def test_exact_token_match(self):
        result = match_compensation("COOK TIMOTHY D", self._COMP)
        self.assertAlmostEqual(result, 63208000.0)

    def test_partial_match_above_threshold(self):
        result = match_compensation("LUCA MAESTRI CFO", self._COMP)
        self.assertAlmostEqual(result, 27200000.0)

    def test_no_match_below_threshold(self):
        result = match_compensation("JOHN SMITH BOB", self._COMP)
        self.assertIsNone(result)

    def test_returns_none_for_empty_comp_map(self):
        self.assertIsNone(match_compensation("TIMOTHY COOK", {}))

    def test_returns_none_for_empty_reporter(self):
        self.assertIsNone(match_compensation("", self._COMP))

    def test_returns_none_for_reporter_with_only_punctuation(self):
        self.assertIsNone(match_compensation("...", self._COMP))

    def test_picks_best_match(self):
        comp = {"TIMOTHY D COOK": 63208000.0, "TIMOTHY SMITH": 1000000.0}
        result = match_compensation("TIMOTHY D COOK", comp)
        self.assertAlmostEqual(result, 63208000.0)

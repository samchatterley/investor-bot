"""Tests for data/edgar_client.py — 100% coverage."""

from __future__ import annotations

import contextlib
import os
from datetime import date, timedelta
from unittest import TestCase
from unittest.mock import MagicMock, patch

from data.edgar_client import (
    _classify_guidance,
    _fetch_8k_exhibit_text,
    _fetch_8k_guidance,
    _fetch_13d_activist,
    _fetch_accounting_concern,
    _fetch_filing_text,
    _fetch_ma_event,
    _fetch_recent_filings,
    _fetch_regulatory_event,
    _fetch_secondary_offering,
    _get_cik_map,
    _get_recent_filings,
    _is_stale,
    _live_fetch,
    _load_cache,
    _save_cache,
    _today_entry,
    classify_guidance_text,
    get_accounting_concern,
    get_activist_filing,
    get_edgar_signals,
    get_edgar_signals_batch,
    get_guidance_sentiment,
    get_ma_event,
    get_regulatory_event,
    get_secondary_offering,
    prefetch_edgar_data,
)

# ── Cache helpers ─────────────────────────────────────────────────────────────


class TestCacheIO(TestCase):
    def test_save_load_roundtrip(self):
        data = {"2026-06-04": {"AAPL": {"guidance": None}}}
        with patch("data.edgar_client._CACHE_PATH", "/tmp/_edgar_cache.json"):
            _save_cache(data)
            loaded = _load_cache()
        self.assertIn("2026-06-04", loaded)
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_edgar_cache.json")

    def test_load_missing(self):
        with patch("data.edgar_client._CACHE_PATH", "/tmp/_no_edgar.json"):
            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_corrupt(self):
        with patch("data.edgar_client._CACHE_PATH", "/tmp/_corrupt_edgar.json"):
            with open("/tmp/_corrupt_edgar.json", "w") as f:
                f.write("{bad")
            result = _load_cache()
        self.assertEqual(result, {})
        with contextlib.suppress(FileNotFoundError):
            os.remove("/tmp/_corrupt_edgar.json")

    def test_save_oserror(self):
        with (
            patch("data.edgar_client._CACHE_PATH", "/no_dir/x.json"),
            patch("data.edgar_client.os.makedirs", side_effect=OSError),
        ):
            _save_cache({})  # must not raise

    def test_is_stale_true(self):
        entry = {"_date": "2026-01-01"}
        with patch("data.edgar_client.today_et") as m:
            m.return_value = date(2026, 6, 4)
            self.assertTrue(_is_stale(entry))

    def test_is_stale_false(self):
        entry = {"_date": "2026-06-04"}
        with patch("data.edgar_client.today_et") as m:
            m.return_value = date(2026, 6, 4)
            self.assertFalse(_is_stale(entry))


# ── CIK map ───────────────────────────────────────────────────────────────────


class TestGetCikMap(TestCase):
    def setUp(self):
        # Clear lru_cache between tests
        _get_cik_map.cache_clear()

    def tearDown(self):
        _get_cik_map.cache_clear()

    def test_successful_fetch(self):
        fake_data = {
            "0": {"ticker": "AAPL", "cik_str": 320193},
            "1": {"ticker": "MSFT", "cik_str": 789019},
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = fake_data
        with patch("data.edgar_client.requests.get", return_value=mock_resp):
            result = _get_cik_map()
        self.assertEqual(result["AAPL"], "0000320193")
        self.assertEqual(result["MSFT"], "0000789019")

    def test_network_failure(self):
        with patch("data.edgar_client.requests.get", side_effect=RuntimeError("timeout")):
            result = _get_cik_map()
        self.assertEqual(result, {})


# ── Recent filings ────────────────────────────────────────────────────────────


def _make_submissions_response(
    forms: list[str],
    dates: list[str],
    accessions: list[str],
    docs: list[str],
    items: list[str] | None = None,
) -> MagicMock:
    if items is None:
        items = [""] * len(forms)
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "filings": {
            "recent": {
                "form": forms,
                "filingDate": dates,
                "accessionNumber": accessions,
                "primaryDocument": docs,
                "items": items,
            }
        }
    }
    return mock


class TestGetRecentFilings(TestCase):
    def setUp(self):
        _fetch_recent_filings.cache_clear()

    def tearDown(self):
        _fetch_recent_filings.cache_clear()

    def test_shared_fetch_dedupes_submissions(self):
        # The per-symbol form lookups (8-K / 13D / 424B) must share ONE network fetch.
        today = date.today().isoformat()
        resp = _make_submissions_response(
            forms=["8-K", "SC 13D"],
            dates=[today, today],
            accessions=["0001-23-456789", "0002-34-567890"],
            docs=["a.htm", "b.htm"],
            items=["2.02", ""],
        )
        with (
            patch("data.edgar_client.requests.get", return_value=resp) as mock_get,
            patch("data.edgar_client.time.sleep"),
        ):
            _get_recent_filings("0000320193", ["8-K"], lookback_days=30)
            _get_recent_filings("0000320193", ["SC 13D"], lookback_days=30)
            _get_recent_filings("0000320193", ["424B4"], lookback_days=30)
        self.assertEqual(mock_get.call_count, 1)

    def test_returns_matching_forms(self):
        today = date.today().isoformat()
        resp = _make_submissions_response(
            forms=["8-K", "10-Q", "8-K"],
            dates=[today, today, today],
            accessions=["0001-23-456789", "0002-34-567890", "0003-45-678901"],
            docs=["doc1.htm", "doc2.htm", "doc3.htm"],
            items=["2.02", "N/A", "7.01"],
        )
        with (
            patch("data.edgar_client.requests.get", return_value=resp),
            patch("data.edgar_client.time.sleep"),
        ):
            result = _get_recent_filings("0000320193", ["8-K"], lookback_days=30)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["form"], "8-K")

    def test_stops_at_cutoff(self):
        old_date = (date.today() - timedelta(days=60)).isoformat()
        resp = _make_submissions_response(
            forms=["8-K"],
            dates=[old_date],
            accessions=["0001-23-456789"],
            docs=["doc.htm"],
        )
        with (
            patch("data.edgar_client.requests.get", return_value=resp),
            patch("data.edgar_client.time.sleep"),
        ):
            result = _get_recent_filings("0000320193", ["8-K"], lookback_days=30)
        self.assertEqual(result, [])

    def test_bad_date_skipped(self):
        resp = _make_submissions_response(
            forms=["8-K"],
            dates=["not-a-date"],
            accessions=["0001"],
            docs=["doc.htm"],
        )
        with (
            patch("data.edgar_client.requests.get", return_value=resp),
            patch("data.edgar_client.time.sleep"),
        ):
            result = _get_recent_filings("0000320193", ["8-K"], lookback_days=30)
        self.assertEqual(result, [])

    def test_network_failure(self):
        with (
            patch("data.edgar_client.requests.get", side_effect=RuntimeError),
            patch("data.edgar_client.time.sleep"),
        ):
            result = _get_recent_filings("0000320193", ["8-K"], lookback_days=30)
        self.assertEqual(result, [])


# ── Filing text fetch ─────────────────────────────────────────────────────────


class TestFetchFilingText(TestCase):
    def test_fetches_and_strips_html(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = "<html><body><p>Raises guidance for full-year revenue.</p></body></html>"
        with (
            patch("data.edgar_client.requests.get", return_value=mock_resp),
            patch("data.edgar_client.time.sleep"),
        ):
            result = _fetch_filing_text("320193", "000032019326000001", "doc.htm")
        self.assertIn("raises guidance", result)
        self.assertNotIn("<", result)

    def test_network_failure_returns_empty(self):
        with (
            patch("data.edgar_client.requests.get", side_effect=RuntimeError),
            patch("data.edgar_client.time.sleep"),
        ):
            result = _fetch_filing_text("320193", "000032019326000001", "doc.htm")
        self.assertEqual(result, "")

    def test_truncates_at_max_chars(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = "A" * 20000
        with (
            patch("data.edgar_client.requests.get", return_value=mock_resp),
            patch("data.edgar_client.time.sleep"),
        ):
            result = _fetch_filing_text("320193", "000032019326000001", "doc.htm", max_chars=100)
        self.assertLessEqual(len(result), 200)  # stripped whitespace may be shorter


# ── Keyword classifier ────────────────────────────────────────────────────────


class TestClassifyGuidance(TestCase):
    def test_positive_keywords(self):
        self.assertEqual(_classify_guidance("raises guidance for the year"), "positive")
        self.assertEqual(_classify_guidance("above consensus estimates"), "positive")
        self.assertEqual(_classify_guidance("record revenue achieved"), "positive")

    def test_negative_keywords(self):
        self.assertEqual(_classify_guidance("lowers guidance due to headwinds"), "negative")
        self.assertEqual(_classify_guidance("revenue shortfall disappointing"), "negative")

    def test_neutral_tie(self):
        # No keywords → neutral
        self.assertEqual(_classify_guidance("quarterly results announced"), "neutral")

    def test_public_wrapper(self):
        result = classify_guidance_text("RAISES GUIDANCE FOR FULL-YEAR")
        self.assertEqual(result, "positive")

    def test_tie_is_neutral(self):
        # Equal positive and negative hits → neutral
        text = "raises guidance but lowers guidance"
        self.assertEqual(_classify_guidance(text), "neutral")

    def test_enriched_positive_language(self):
        text = "the company exceeded expectations with record revenue and raised its outlook"
        self.assertEqual(_classify_guidance(text), "positive")

    def test_enriched_negative_language(self):
        text = "net loss widened amid an impairment charge and softening demand"
        self.assertEqual(_classify_guidance(text), "negative")

    def test_non_contiguous_guidance_phrase_caught(self):
        # The "raised" stem catches phrasing the old contiguous "raised guidance" missed.
        self.assertEqual(
            _classify_guidance("management raised its full-year revenue guidance"), "positive"
        )

    def test_word_boundary_avoids_substring_false_positive(self):
        # "praised" contains "raised", "flowered" contains "lowered" — neither should count.
        self.assertEqual(_classify_guidance("analysts praised the new strategy"), "neutral")
        self.assertEqual(_classify_guidance("the garden flowered nicely"), "neutral")


# ── 8-K guidance detection ────────────────────────────────────────────────────


class TestFetch8kGuidance(TestCase):
    def _filing(self, items="2.02") -> dict:
        return {
            "form": "8-K",
            "filing_date": date.today().isoformat(),
            "accession": "000032019326000001",
            "doc": "doc.htm",
            "items": items,
        }

    def test_positive_guidance_detected(self):
        with (
            patch("data.edgar_client._get_recent_filings", return_value=[self._filing("2.02")]),
            patch(
                "data.edgar_client._fetch_8k_exhibit_text",
                return_value="company raises guidance for full-year revenue",
            ),
        ):
            result = _fetch_8k_guidance("AAPL", "0000320193", 30)
        self.assertIsNotNone(result)
        self.assertEqual(result["sentiment"], "positive")
        self.assertTrue(result["guidance_positive"])
        self.assertFalse(result["guidance_negative"])

    def test_negative_guidance_detected(self):
        with (
            patch("data.edgar_client._get_recent_filings", return_value=[self._filing("7.01")]),
            patch(
                "data.edgar_client._fetch_8k_exhibit_text",
                return_value="company lowers guidance headwinds softening demand",
            ),
        ):
            result = _fetch_8k_guidance("AAPL", "0000320193", 30)
        self.assertIsNotNone(result)
        self.assertEqual(result["sentiment"], "negative")
        self.assertTrue(result["guidance_negative"])

    def test_no_guidance_filings(self):
        # 8-K with item 1.01 only (not 2.02 or 7.01)
        with patch(
            "data.edgar_client._get_recent_filings",
            return_value=[{**self._filing(), "items": "1.01"}],
        ):
            result = _fetch_8k_guidance("AAPL", "0000320193", 30)
        self.assertIsNone(result)

    def test_empty_text_returns_none(self):
        with (
            patch("data.edgar_client._get_recent_filings", return_value=[self._filing("2.02")]),
            patch("data.edgar_client._fetch_8k_exhibit_text", return_value=""),
        ):
            result = _fetch_8k_guidance("AAPL", "0000320193", 30)
        self.assertIsNone(result)

    def test_no_recent_filings(self):
        with patch("data.edgar_client._get_recent_filings", return_value=[]):
            result = _fetch_8k_guidance("AAPL", "0000320193", 30)
        self.assertIsNone(result)


def _f(items: str, accession: str = "000032019326000001", doc: str = "d.htm") -> dict:
    """Minimal 8-K filing record for the narrative-event detectors."""
    return {
        "form": "8-K",
        "filing_date": date.today().isoformat(),
        "accession": accession,
        "doc": doc,
        "items": items,
    }


class TestFetchAccountingConcern(TestCase):
    def test_item_402_restatement_detected(self):
        result = _fetch_accounting_concern("AAPL", [_f("4.02,9.01")])
        self.assertIsNotNone(result)
        self.assertTrue(result["detected"])
        self.assertEqual(result["items"], "4.02,9.01")

    def test_item_401_auditor_change_detected(self):
        result = _fetch_accounting_concern("AAPL", [_f("4.01")])
        self.assertTrue(result["detected"])

    def test_unrelated_items_returns_none(self):
        self.assertIsNone(_fetch_accounting_concern("AAPL", [_f("5.02"), _f("2.02")]))

    def test_empty_filings_returns_none(self):
        self.assertIsNone(_fetch_accounting_concern("AAPL", []))


class TestFetchMaEvent(TestCase):
    def test_item_201_completion_detected_without_text_fetch(self):
        with patch("data.edgar_client._fetch_8k_exhibit_text") as exhibit:
            result = _fetch_ma_event("AAPL", "0000320193", [_f("2.01,9.01")])
        self.assertTrue(result["detected"])
        self.assertEqual(result["trigger"], "2.01")
        exhibit.assert_not_called()  # high-precision item needs no text confirmation

    def test_item_101_with_ma_keywords_detected(self):
        with patch(
            "data.edgar_client._fetch_8k_exhibit_text",
            return_value="entered into an agreement and plan of merger to be acquired",
        ):
            result = _fetch_ma_event("AAPL", "0000320193", [_f("1.01")])
        self.assertTrue(result["detected"])
        self.assertEqual(result["trigger"], "1.01+kw")

    def test_item_101_without_keywords_returns_none(self):
        with patch(
            "data.edgar_client._fetch_8k_exhibit_text",
            return_value="entered into a new revolving credit facility with its lenders",
        ):
            self.assertIsNone(_fetch_ma_event("AAPL", "0000320193", [_f("1.01")]))

    def test_item_101_empty_text_returns_none(self):
        with patch("data.edgar_client._fetch_8k_exhibit_text", return_value=""):
            self.assertIsNone(_fetch_ma_event("AAPL", "0000320193", [_f("1.01")]))

    def test_item_101_debt_covenant_boilerplate_not_detected(self):
        # Live false positive from Ford: bare "merger" in indenture covenant language on a debt 8-K.
        with patch(
            "data.edgar_client._fetch_8k_exhibit_text",
            return_value="in the event of a merger, disposition or transfer of substantially all assets",
        ):
            self.assertIsNone(_fetch_ma_event("F", "0000037996", [_f("1.01,2.03")]))

    def test_unrelated_items_returns_none(self):
        self.assertIsNone(_fetch_ma_event("AAPL", "0000320193", [_f("5.02")]))


class TestFetchRegulatoryEvent(TestCase):
    def test_item_301_delisting_detected_without_text_fetch(self):
        with patch("data.edgar_client._fetch_8k_exhibit_text") as exhibit:
            result = _fetch_regulatory_event("AAPL", "0000320193", [_f("3.01")])
        self.assertTrue(result["detected"])
        self.assertEqual(result["trigger"], "3.01")
        exhibit.assert_not_called()

    def test_item_801_with_regulatory_keywords_detected(self):
        with patch(
            "data.edgar_client._fetch_8k_exhibit_text",
            return_value="received an fda complete response letter for its lead candidate",
        ):
            result = _fetch_regulatory_event("AAPL", "0000320193", [_f("8.01")])
        self.assertTrue(result["detected"])
        self.assertEqual(result["trigger"], "8.01+kw")

    def test_item_801_without_keywords_returns_none(self):
        with patch(
            "data.edgar_client._fetch_8k_exhibit_text",
            return_value="announced the date of its next quarterly earnings call",
        ):
            self.assertIsNone(_fetch_regulatory_event("AAPL", "0000320193", [_f("8.01")]))

    def test_item_801_empty_text_returns_none(self):
        with patch("data.edgar_client._fetch_8k_exhibit_text", return_value=""):
            self.assertIsNone(_fetch_regulatory_event("AAPL", "0000320193", [_f("8.01")]))

    def test_item_801_fda_action_verb_detected(self):
        with patch(
            "data.edgar_client._fetch_8k_exhibit_text",
            return_value="the fda approved the company's new drug application this morning",
        ):
            result = _fetch_regulatory_event("MRK", "0000310158", [_f("8.01")])
        self.assertTrue(result["detected"])

    def test_item_801_fda_approval_noun_boilerplate_not_detected(self):
        # "FDA approval" as a noun is risk-factor boilerplate; only the action verb counts.
        with patch(
            "data.edgar_client._fetch_8k_exhibit_text",
            return_value="our products require fda approval before they can be marketed",
        ):
            self.assertIsNone(_fetch_regulatory_event("MRK", "0000310158", [_f("8.01")]))

    def test_item_801_sec_filing_boilerplate_not_detected(self):
        # Live false positive from JPM/GILD: "securities and exchange commission" is in nearly every
        # 8-K's standard filing language and must not count as a regulatory event.
        with patch(
            "data.edgar_client._fetch_8k_exhibit_text",
            return_value="this report was filed by the company with the securities and exchange commission",
        ):
            self.assertIsNone(_fetch_regulatory_event("JPM", "0000019617", [_f("8.01")]))

    def test_unrelated_items_returns_none(self):
        self.assertIsNone(_fetch_regulatory_event("AAPL", "0000320193", [_f("5.02")]))


class TestNarrativeGetters(TestCase):
    def test_getters_read_today_entry(self):
        entry = {
            "ma_event": {"detected": True, "trigger": "2.01"},
            "accounting_concern": {"detected": True},
            "regulatory_event": {"detected": True, "trigger": "3.01"},
        }
        with patch("data.edgar_client._today_entry", return_value=entry):
            self.assertEqual(get_ma_event("AAPL")["trigger"], "2.01")
            self.assertTrue(get_accounting_concern("AAPL")["detected"])
            self.assertEqual(get_regulatory_event("AAPL")["trigger"], "3.01")

    def test_getters_return_none_when_absent(self):
        with patch("data.edgar_client._today_entry", return_value={}):
            self.assertIsNone(get_ma_event("AAPL"))
            self.assertIsNone(get_accounting_concern("AAPL"))
            self.assertIsNone(get_regulatory_event("AAPL"))


class TestFetch8kExhibitText(TestCase):
    def _index_resp(self, names):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"directory": {"item": [{"name": n} for n in names]}}
        return r

    def _doc_resp(self, text):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.text = text
        return r

    def test_combines_exhibits_excluding_cover_index_xbrl(self):
        idx = self._index_resp(
            [
                "x-index.html",
                "FilingSummary.xml",
                "cover.htm",
                "R1.htm",
                "pr.htm",
                "cfo.htm",
                "logo.jpg",
            ]
        )
        pr = self._doc_resp("<p>raises guidance, record revenue</p>")
        cfo = self._doc_resp("<p>strong full-year outlook</p>")
        with (
            patch("data.edgar_client.requests.get", side_effect=[idx, pr, cfo]),
            patch("data.edgar_client.time.sleep"),
        ):
            txt = _fetch_8k_exhibit_text("320193", "000032019326000001", "cover.htm")
        self.assertIn("raises guidance", txt)
        self.assertIn("strong full-year outlook", txt)
        self.assertNotIn("cover", txt)  # cover page not included

    def test_falls_back_to_primary_when_no_exhibits(self):
        idx = self._index_resp(["x-index.html", "cover.htm", "R2.htm"])
        cover = self._doc_resp("<p>cover page only</p>")
        with (
            patch("data.edgar_client.requests.get", side_effect=[idx, cover]),
            patch("data.edgar_client.time.sleep"),
        ):
            txt = _fetch_8k_exhibit_text("320193", "000032019326000001", "cover.htm")
        self.assertIn("cover page only", txt)

    def test_falls_back_when_index_fetch_fails(self):
        cover = self._doc_resp("<p>cover fallback body</p>")
        with (
            patch(
                "data.edgar_client.requests.get", side_effect=[RuntimeError("index down"), cover]
            ),
            patch("data.edgar_client.time.sleep"),
        ):
            txt = _fetch_8k_exhibit_text("320193", "000032019326000001", "cover.htm")
        self.assertIn("cover fallback body", txt)

    def test_empty_exhibits_fall_back_to_primary(self):
        idx = self._index_resp(["cover.htm", "pr.htm"])
        empty = self._doc_resp("")
        cover = self._doc_resp("<p>primary body text</p>")
        with (
            patch("data.edgar_client.requests.get", side_effect=[idx, empty, cover]),
            patch("data.edgar_client.time.sleep"),
        ):
            txt = _fetch_8k_exhibit_text("320193", "000032019326000001", "cover.htm")
        self.assertIn("primary body text", txt)

    def test_stops_after_reaching_size_cap(self):
        # a (50) + b (50) >= 2*max_chars(50) -> break before fetching c
        idx = self._index_resp(["cover.htm", "a.htm", "b.htm", "c.htm"])
        a = self._doc_resp("A" * 100)
        b = self._doc_resp("B" * 100)
        with (
            patch("data.edgar_client.requests.get", side_effect=[idx, a, b]),
            patch("data.edgar_client.time.sleep"),
        ):
            txt = _fetch_8k_exhibit_text("320193", "000032019326000001", "cover.htm", max_chars=50)
        self.assertIn("a" * 50, txt)
        self.assertIn("b" * 50, txt)
        self.assertNotIn("c", txt)  # third exhibit never fetched


# ── 13D activist detection ────────────────────────────────────────────────────


class TestFetch13dActivist(TestCase):
    def _filing(self, form="SC 13D") -> dict:
        return {
            "form": form,
            "filing_date": date.today().isoformat(),
            "accession": "000032019326000001",
            "doc": "sc13d.htm",
            "items": "",
        }

    def test_known_activist_detected(self):
        with (
            patch("data.edgar_client._get_recent_filings", return_value=[self._filing()]),
            patch(
                "data.edgar_client._fetch_filing_text",
                return_value="elliott investment management has acquired a 5% stake",
            ),
        ):
            result = _fetch_13d_activist("AAPL", "0000320193", 30)
        self.assertIsNotNone(result)
        self.assertTrue(result["known_activist"])
        self.assertEqual(result["activist_name"], "Elliott Investment Management")

    def test_unknown_activist_returned(self):
        with (
            patch("data.edgar_client._get_recent_filings", return_value=[self._filing()]),
            patch(
                "data.edgar_client._fetch_filing_text",
                return_value="some unknown investor has acquired a 5.5% stake",
            ),
        ):
            result = _fetch_13d_activist("AAPL", "0000320193", 30)
        self.assertIsNotNone(result)
        self.assertFalse(result["known_activist"])

    def test_empty_text_skips_then_returns_generic(self):
        # filing exists but text is empty → loop through all filings → fallback
        with (
            patch("data.edgar_client._get_recent_filings", return_value=[self._filing()]),
            patch("data.edgar_client._fetch_filing_text", return_value=""),
        ):
            result = _fetch_13d_activist("AAPL", "0000320193", 30)
        self.assertIsNotNone(result)
        self.assertFalse(result["known_activist"])

    def test_no_filings(self):
        with patch("data.edgar_client._get_recent_filings", return_value=[]):
            result = _fetch_13d_activist("AAPL", "0000320193", 30)
        self.assertIsNone(result)


# ── Secondary offering detection ──────────────────────────────────────────────


class TestFetchSecondaryOffering(TestCase):
    def test_424b4_detected(self):
        filing = {
            "form": "424B4",
            "filing_date": date.today().isoformat(),
            "accession": "000032019326000002",
            "doc": "424b4.htm",
            "items": "",
        }
        with patch("data.edgar_client._get_recent_filings", return_value=[filing]):
            result = _fetch_secondary_offering("AAPL", "0000320193", 30)
        self.assertIsNotNone(result)
        self.assertTrue(result["offering_detected"])
        self.assertEqual(result["form"], "424B4")

    def test_s3_detected(self):
        filing = {
            "form": "S-3",
            "filing_date": date.today().isoformat(),
            "accession": "000032019326000002",
            "doc": "s3.htm",
            "items": "",
        }
        with patch("data.edgar_client._get_recent_filings", return_value=[filing]):
            result = _fetch_secondary_offering("AAPL", "0000320193", 30)
        self.assertIsNotNone(result)

    def test_no_filings(self):
        with patch("data.edgar_client._get_recent_filings", return_value=[]):
            result = _fetch_secondary_offering("AAPL", "0000320193", 30)
        self.assertIsNone(result)


# ── _live_fetch ───────────────────────────────────────────────────────────────


class TestLiveFetch(TestCase):
    def setUp(self):
        _get_cik_map.cache_clear()

    def tearDown(self):
        _get_cik_map.cache_clear()

    def test_cik_not_found_returns_empty(self):
        with patch("data.edgar_client._get_cik_map", return_value={}):
            result = _live_fetch("AAPL", 30)
        self.assertEqual(result, {})

    def test_full_fetch_with_all_events(self):
        guidance = {
            "sentiment": "positive",
            "filing_date": "2026-06-01",
            "items": "2.02",
            "guidance_positive": True,
            "guidance_negative": False,
        }
        activist = {
            "activist_name": "Starboard Value",
            "filing_date": "2026-06-01",
            "known_activist": True,
            "form": "SC 13D",
        }
        offering = {"form": "424B4", "filing_date": "2026-06-01", "offering_detected": True}
        with (
            patch("data.edgar_client._get_cik_map", return_value={"AAPL": "0000320193"}),
            patch("data.edgar_client._fetch_8k_guidance", return_value=guidance),
            patch("data.edgar_client._fetch_13d_activist", return_value=activist),
            patch("data.edgar_client._fetch_secondary_offering", return_value=offering),
            patch("data.edgar_client._get_recent_filings", return_value=[]),
            patch("data.edgar_client._fetch_ma_event", return_value={"detected": True}),
            patch("data.edgar_client._fetch_accounting_concern", return_value={"detected": True}),
            patch("data.edgar_client._fetch_regulatory_event", return_value={"detected": True}),
        ):
            result = _live_fetch("AAPL", 30)
        self.assertIn("guidance", result)
        self.assertIn("activist", result)
        self.assertIn("secondary_offering", result)
        self.assertIn("ma_event", result)
        self.assertIn("accounting_concern", result)
        self.assertIn("regulatory_event", result)

    def test_no_events_returns_empty_dict(self):
        with (
            patch("data.edgar_client._get_cik_map", return_value={"AAPL": "0000320193"}),
            patch("data.edgar_client._fetch_8k_guidance", return_value=None),
            patch("data.edgar_client._fetch_13d_activist", return_value=None),
            patch("data.edgar_client._fetch_secondary_offering", return_value=None),
            patch("data.edgar_client._get_recent_filings", return_value=[]),
        ):
            result = _live_fetch("AAPL", 30)
        self.assertEqual(result, {})


# ── Prefetch ──────────────────────────────────────────────────────────────────


class TestPrefetchEdgarData(TestCase):
    def test_already_warm_skips(self):
        today = "2026-06-04"
        cache = {today: {"AAPL": {}}}
        with (
            patch("data.edgar_client._load_cache", return_value=cache),
            patch("data.edgar_client.today_et") as m,
        ):
            m.return_value = date(2026, 6, 4)
            n = prefetch_edgar_data(symbols=["AAPL"])
        self.assertEqual(n, 0)

    def test_fetches_missing_symbols(self):
        with (
            patch("data.edgar_client._load_cache", return_value={}),
            patch("data.edgar_client._save_cache") as mock_save,
            patch("data.edgar_client._live_fetch", return_value={}),
            patch("data.edgar_client.today_et") as m,
        ):
            m.return_value = date(2026, 6, 4)
            n = prefetch_edgar_data(symbols=["AAPL", "MSFT"])
        self.assertEqual(n, 2)
        mock_save.assert_called_once()

    def test_default_universe(self):
        with (
            patch("data.edgar_client._load_cache", return_value={}),
            patch("data.edgar_client._save_cache"),
            patch("data.edgar_client._live_fetch", return_value={}),
            patch("data.edgar_client.today_et") as m,
            patch("data.edgar_client.STOCK_UNIVERSE", ["AAPL"]),
        ):
            m.return_value = date(2026, 6, 4)
            n = prefetch_edgar_data()
        self.assertEqual(n, 1)


# ── _today_entry and public getters ──────────────────────────────────────────


class TestTodayEntry(TestCase):
    def test_cache_hit(self):
        today = "2026-06-04"
        entry = {"guidance": {"sentiment": "positive", "filing_date": today}}
        cache = {today: {"AAPL": entry}}
        with (
            patch("data.edgar_client._load_cache", return_value=cache),
            patch("data.edgar_client.today_et") as m,
        ):
            m.return_value = date(2026, 6, 4)
            result = _today_entry("AAPL", 30)
        self.assertEqual(result, entry)

    def test_cache_miss_fetches_live(self):
        with (
            patch("data.edgar_client._load_cache", return_value={}),
            patch("data.edgar_client._save_cache"),
            patch("data.edgar_client._live_fetch", return_value={"activist": {}}),
            patch("data.edgar_client.today_et") as m,
        ):
            m.return_value = date(2026, 6, 4)
            result = _today_entry("AAPL", 30)
        self.assertIn("activist", result)


class TestPublicGetters(TestCase):
    def test_get_guidance_sentiment_present(self):
        guidance = {"sentiment": "positive", "guidance_positive": True}
        with patch("data.edgar_client._today_entry", return_value={"guidance": guidance}):
            result = get_guidance_sentiment("AAPL")
        self.assertEqual(result["sentiment"], "positive")

    def test_get_guidance_sentiment_absent(self):
        with patch("data.edgar_client._today_entry", return_value={}):
            result = get_guidance_sentiment("AAPL")
        self.assertIsNone(result)

    def test_get_activist_filing_present(self):
        activist = {"activist_name": "Starboard", "known_activist": True}
        with patch("data.edgar_client._today_entry", return_value={"activist": activist}):
            result = get_activist_filing("AAPL")
        self.assertEqual(result["activist_name"], "Starboard")

    def test_get_activist_filing_absent(self):
        with patch("data.edgar_client._today_entry", return_value={}):
            result = get_activist_filing("AAPL")
        self.assertIsNone(result)

    def test_get_secondary_offering_present(self):
        offering = {"form": "424B4", "offering_detected": True}
        with patch("data.edgar_client._today_entry", return_value={"secondary_offering": offering}):
            result = get_secondary_offering("AAPL")
        self.assertEqual(result["form"], "424B4")

    def test_get_secondary_offering_absent(self):
        with patch("data.edgar_client._today_entry", return_value={}):
            result = get_secondary_offering("AAPL")
        self.assertIsNone(result)

    def test_get_edgar_signals_all_present(self):
        entry = {
            "guidance": {"sentiment": "positive"},
            "activist": {"known_activist": True},
            "secondary_offering": {"offering_detected": True},
        }
        with patch("data.edgar_client._today_entry", return_value=entry):
            result = get_edgar_signals("AAPL")
        self.assertIn("guidance", result)
        self.assertIn("activist", result)
        self.assertIn("secondary_offering", result)

    def test_get_edgar_signals_empty(self):
        with patch("data.edgar_client._today_entry", return_value={}):
            result = get_edgar_signals("AAPL")
        self.assertEqual(result, {})


# ── get_edgar_signals_batch ───────────────────────────────────────────────────


class TestGetEdgarSignalsBatch(TestCase):
    def test_all_cache_hits_no_live_fetch(self):
        today = "2026-06-04"
        entry = {"guidance": {"guidance_positive": True}}
        cache = {today: {"AAPL": entry, "MSFT": entry}}
        with (
            patch("data.edgar_client._load_cache", return_value=cache),
            patch("data.edgar_client._save_cache") as mock_save,
            patch("data.edgar_client._live_fetch") as mock_live,
            patch("data.edgar_client.today_et") as m,
        ):
            m.return_value = date(2026, 6, 4)
            result = get_edgar_signals_batch(["AAPL", "MSFT"])
        self.assertEqual(result["AAPL"], entry)
        self.assertEqual(result["MSFT"], entry)
        mock_live.assert_not_called()
        mock_save.assert_not_called()

    def test_cache_miss_triggers_live_fetch(self):
        live_entry = {"activist": {"known_activist": True}}
        with (
            patch("data.edgar_client._load_cache", return_value={}),
            patch("data.edgar_client._save_cache") as mock_save,
            patch("data.edgar_client._live_fetch", return_value=live_entry),
            patch("data.edgar_client.today_et") as m,
        ):
            m.return_value = date(2026, 6, 4)
            result = get_edgar_signals_batch(["AAPL"])
        self.assertEqual(result["AAPL"], live_entry)
        mock_save.assert_called_once()

    def test_mixed_hit_miss(self):
        today = "2026-06-04"
        cached_entry = {"guidance": {"guidance_positive": False}}
        live_entry = {"secondary_offering": {"offering_detected": True}}
        cache = {today: {"AAPL": cached_entry}}
        with (
            patch("data.edgar_client._load_cache", return_value=cache),
            patch("data.edgar_client._save_cache"),
            patch("data.edgar_client._live_fetch", return_value=live_entry),
            patch("data.edgar_client.today_et") as m,
        ):
            m.return_value = date(2026, 6, 4)
            result = get_edgar_signals_batch(["AAPL", "TSLA"])
        self.assertEqual(result["AAPL"], cached_entry)
        self.assertEqual(result["TSLA"], live_entry)

    def test_empty_symbols_returns_empty_dict(self):
        with (
            patch("data.edgar_client._load_cache", return_value={}),
            patch("data.edgar_client.today_et") as m,
        ):
            m.return_value = date(2026, 6, 4)
            result = get_edgar_signals_batch([])
        self.assertEqual(result, {})

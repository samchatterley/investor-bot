"""Tests for data/edgar_event_history.py — the historical EDGAR filing-event feed."""

import os
import tempfile
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import data.edgar_event_history as eh

_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["424B5", "8-K", "4", "SC 13D", "10-K", "8-K"],
            "filingDate": [
                "2024-03-01",
                "2024-02-15",
                "2024-02-10",
                "2024-01-05",
                "2023-12-01",
                "not-a-date",
            ],
            "accessionNumber": ["0001-23", "0002-24", "0003", "0004", "0005", "0006"],
            "items": ["", "4.02", "", "", "", ""],
        }
    }
}


class TestParseEvents(unittest.TestCase):
    def test_all_events_bad_date_skipped(self):
        ev = eh._parse_events(_SUBMISSIONS, None)
        self.assertEqual(len(ev), 5)  # the "not-a-date" row is dropped
        self.assertEqual(
            ev[0], {"date": "2024-03-01", "form": "424B5", "accession": "000123", "items": ""}
        )

    def test_form_prefix_filter(self):
        ev = eh._parse_events(_SUBMISSIONS, ("424B", "S-3"))
        self.assertEqual([e["form"] for e in ev], ["424B5"])

    def test_8k_item_codes_preserved(self):
        ev = eh._parse_events(_SUBMISSIONS, ("8-K",))
        self.assertEqual(ev[0]["items"], "4.02")

    def test_empty_submissions(self):
        self.assertEqual(eh._parse_events({}, None), [])

    def test_missing_items_defaults_to_blank(self):
        subs = {
            "filings": {
                "recent": {
                    "form": ["424B5"],
                    "filingDate": ["2024-01-01"],
                    "accessionNumber": ["a"],
                }
            }
        }
        self.assertEqual(eh._parse_events(subs, None)[0]["items"], "")


class TestFetchSubmissions(unittest.TestCase):
    @patch("data.edgar_event_history.requests.get")
    def test_fetches_and_returns_json(self, mget):
        resp = MagicMock()
        resp.json.return_value = _SUBMISSIONS
        mget.return_value = resp
        with patch("time.sleep"):
            out = eh._fetch_submissions("0000000001")
        self.assertIn("filings", out)
        resp.raise_for_status.assert_called_once()


class TestCacheAndFetchEvents(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._patch = patch.object(eh, "_CACHE_PATH", os.path.join(self._tmp, "cache.json"))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_load_cache_missing_returns_empty(self):
        self.assertEqual(eh._load_cache(), {})

    def test_load_cache_corrupt_returns_empty(self):
        with open(eh._CACHE_PATH, "w") as f:
            f.write("{not json")
        self.assertEqual(eh._load_cache(), {})

    def test_save_then_load_roundtrip(self):
        eh._save_cache({"AAA": {"day": "2024-01-01", "events": []}})
        self.assertEqual(eh._load_cache()["AAA"]["day"], "2024-01-01")

    def test_cache_hit_skips_network(self):
        eh._save_cache(
            {
                "AAA": {
                    "day": date.today().isoformat(),
                    "events": [
                        {"date": "2024-03-01", "form": "424B5", "accession": "x", "items": ""}
                    ],
                }
            }
        )
        with patch.object(eh, "_fetch_submissions") as mfetch:
            out = eh.fetch_events("aaa", eh.OFFERING_FORMS)
        mfetch.assert_not_called()
        self.assertEqual(len(out), 1)

    def test_cache_hit_forms_none_returns_all(self):
        eh._save_cache(
            {
                "AAA": {
                    "day": date.today().isoformat(),
                    "events": [
                        {"date": "2024-03-01", "form": "424B5", "accession": "x", "items": ""},
                        {"date": "2024-02-01", "form": "8-K", "accession": "y", "items": ""},
                    ],
                }
            }
        )
        with patch.object(eh, "_fetch_submissions"):
            self.assertEqual(len(eh.fetch_events("AAA")), 2)

    def test_cache_miss_fetches_parses_and_saves(self):
        with (
            patch.object(eh._ec, "_get_cik_map", return_value={"AAA": "0000000001"}),
            patch.object(eh, "_fetch_submissions", return_value=_SUBMISSIONS),
        ):
            out = eh.fetch_events("AAA", eh.OFFERING_FORMS)
        self.assertEqual([e["form"] for e in out], ["424B5"])
        # persisted for next time
        self.assertIn("AAA", eh._load_cache())

    def test_unknown_ticker_returns_empty(self):
        with patch.object(eh._ec, "_get_cik_map", return_value={}):
            self.assertEqual(eh.fetch_events("NOPE"), [])

    def test_network_failure_returns_empty(self):
        with (
            patch.object(eh._ec, "_get_cik_map", return_value={"AAA": "0000000001"}),
            patch.object(eh, "_fetch_submissions", side_effect=RuntimeError("boom")),
        ):
            self.assertEqual(eh.fetch_events("AAA"), [])

    def test_no_cache_bypasses_disk(self):
        with (
            patch.object(eh._ec, "_get_cik_map", return_value={"AAA": "0000000001"}),
            patch.object(eh, "_fetch_submissions", return_value=_SUBMISSIONS),
        ):
            out = eh.fetch_events("AAA", use_cache=False)
        self.assertEqual(len(out), 5)
        self.assertEqual(eh._load_cache(), {})  # nothing written


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""Tests for data/analyst_revisions.py."""

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

_TODAY_STR = "2026-06-12"
_TODAY_KEY = _TODAY_STR


def _today():
    from datetime import date

    return date.fromisoformat(_TODAY_STR)


class TestLoadSaveCache(unittest.TestCase):
    def test_load_returns_empty_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            from data.analyst_revisions import _load_cache

            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_returns_empty_on_json_error(self):
        with (
            patch("builtins.open", mock_open(read_data="bad json")),
            patch(
                "data.analyst_revisions.json.load",
                side_effect=json.JSONDecodeError("err", "", 0),
            ),
        ):
            from data.analyst_revisions import _load_cache

            result = _load_cache()
        self.assertEqual(result, {})

    def test_load_returns_dict_on_success(self):
        data = {_TODAY_KEY: {"AAPL": {"analyst_upgrade": True, "analyst_downgrade": False}}}
        with (
            patch("builtins.open", mock_open(read_data=json.dumps(data))),
            patch("data.analyst_revisions.json.load", return_value=data),
        ):
            from data.analyst_revisions import _load_cache

            result = _load_cache()
        self.assertEqual(result, data)

    def test_save_writes_json_on_success(self):
        m = mock_open()
        with (
            patch("data.analyst_revisions.os.makedirs"),
            patch("builtins.open", m),
            patch("data.analyst_revisions.json.dump") as mock_dump,
        ):
            from data.analyst_revisions import _save_cache

            _save_cache({_TODAY_KEY: {}})
        mock_dump.assert_called_once()

    def test_save_logs_warning_on_os_error(self):
        with (
            patch("data.analyst_revisions.os.makedirs"),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            from data.analyst_revisions import _save_cache

            _save_cache({_TODAY_KEY: {}})  # should not raise


class TestLiveFetchRevisions(unittest.TestCase):
    def _rec_df(self, cur: dict, prev: dict) -> pd.DataFrame:
        """Build a fake recommendations_summary DataFrame."""
        data = {"0m": cur, "1m": prev}
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.name = "period"
        return df

    def test_etf_symbols_skipped(self):
        with patch("data.analyst_revisions.ETF_SYMBOLS", {"SPY"}):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["SPY"])
        self.assertIsNone(result["SPY"])

    def test_exception_during_fetch_returns_none(self):
        # Make yf.Ticker() itself raise so the except block at line 68-71 is hit
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", side_effect=RuntimeError("network error")),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNone(result["AAPL"])

    def test_empty_recommendations_returns_none(self):
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = pd.DataFrame()
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNone(result["AAPL"])

    def test_none_recommendations_returns_none(self):
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = None
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNone(result["AAPL"])

    def test_only_one_period_returns_none(self):
        df = pd.DataFrame(
            [{"strongBuy": 5, "buy": 3, "hold": 2, "sell": 0, "strongSell": 0}],
            index=pd.Index(["0m"]),
        )
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = df
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNone(result["AAPL"])

    def test_too_few_analysts_returns_none(self):
        # total = 2, less than minimum of 3
        df = self._rec_df(
            cur={"strongBuy": 1, "buy": 0, "hold": 1, "sell": 0, "strongSell": 0},
            prev={"strongBuy": 1, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0},
        )
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = df
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNone(result["AAPL"])

    def test_detects_upgrade(self):
        # buy% rose from 30% to 50% (>10pp)
        df = self._rec_df(
            cur={"strongBuy": 4, "buy": 1, "hold": 5, "sell": 0, "strongSell": 0},  # 50% buy
            prev={"strongBuy": 2, "buy": 1, "hold": 7, "sell": 0, "strongSell": 0},  # 30% buy
        )
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = df
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNotNone(result["AAPL"])
        self.assertTrue(result["AAPL"]["analyst_upgrade"])
        self.assertFalse(result["AAPL"]["analyst_downgrade"])

    def test_detects_downgrade_by_sell_rise(self):
        # sell% rose by >10pp
        df = self._rec_df(
            cur={"strongBuy": 2, "buy": 1, "hold": 3, "sell": 2, "strongSell": 2},  # 40% sell
            prev={"strongBuy": 3, "buy": 2, "hold": 4, "sell": 0, "strongSell": 1},  # 10% sell
        )
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = df
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNotNone(result["AAPL"])
        self.assertTrue(result["AAPL"]["analyst_downgrade"])

    def test_detects_downgrade_by_buy_drop(self):
        # buy% fell by >10pp
        df = self._rec_df(
            cur={"strongBuy": 2, "buy": 1, "hold": 7, "sell": 0, "strongSell": 0},  # 30% buy
            prev={"strongBuy": 3, "buy": 2, "hold": 5, "sell": 0, "strongSell": 0},  # 50% buy
        )
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = df
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNotNone(result["AAPL"])
        self.assertTrue(result["AAPL"]["analyst_downgrade"])

    def test_no_change_neutral(self):
        # No significant shift in buy/sell %
        df = self._rec_df(
            cur={"strongBuy": 3, "buy": 2, "hold": 5, "sell": 0, "strongSell": 0},  # 50% buy
            prev={"strongBuy": 3, "buy": 2, "hold": 5, "sell": 0, "strongSell": 0},  # 50% buy
        )
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = df
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNotNone(result["AAPL"])
        self.assertFalse(result["AAPL"]["analyst_upgrade"])
        self.assertFalse(result["AAPL"]["analyst_downgrade"])

    def test_parse_exception_returns_none(self):
        # Malformed df — to_dict raises
        mock_rec = MagicMock()
        mock_rec.empty = False
        mock_rec.__len__ = lambda self: 2
        mock_rec.to_dict = MagicMock(side_effect=TypeError("bad data"))
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = mock_rec
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNone(result["AAPL"])


class TestPrefetchAnalystRevisions(unittest.TestCase):
    def test_returns_zero_when_already_warm(self):
        cached = {_TODAY_KEY: {"AAPL": {"analyst_upgrade": False, "analyst_downgrade": False}}}
        with (
            patch("data.analyst_revisions.today_et", return_value=_today()),
            patch("data.analyst_revisions._load_cache", return_value=cached),
            patch("data.analyst_revisions._save_cache") as mock_save,
        ):
            from data.analyst_revisions import prefetch_analyst_revisions

            result = prefetch_analyst_revisions(["AAPL"])
        self.assertEqual(result, 0)
        mock_save.assert_not_called()

    def test_fetches_missing_and_returns_count(self):
        cached = {_TODAY_KEY: {}}
        with (
            patch("data.analyst_revisions.today_et", return_value=_today()),
            patch("data.analyst_revisions._load_cache", return_value=cached),
            patch(
                "data.analyst_revisions._live_fetch_revisions",
                return_value={"AAPL": {"analyst_upgrade": True, "analyst_downgrade": False}},
            ),
            patch("data.analyst_revisions._save_cache") as mock_save,
        ):
            from data.analyst_revisions import prefetch_analyst_revisions

            result = prefetch_analyst_revisions(["AAPL"])
        self.assertEqual(result, 1)
        mock_save.assert_called_once()

    def test_uses_stock_universe_when_symbols_none(self):
        with (
            patch("data.analyst_revisions.STOCK_UNIVERSE", {"AAPL", "MSFT"}),
            patch("data.analyst_revisions.today_et", return_value=_today()),
            patch("data.analyst_revisions._load_cache", return_value={_TODAY_KEY: {}}),
            patch(
                "data.analyst_revisions._live_fetch_revisions",
                return_value={"AAPL": None, "MSFT": None},
            ),
            patch("data.analyst_revisions._save_cache"),
        ):
            from data.analyst_revisions import prefetch_analyst_revisions

            result = prefetch_analyst_revisions(None)
        self.assertEqual(result, 2)


class TestGetAnalystRevisions(unittest.TestCase):
    def test_returns_only_non_none_from_cache(self):
        cached = {
            _TODAY_KEY: {
                "AAPL": {"analyst_upgrade": True, "analyst_downgrade": False},
                "MSFT": None,
            }
        }
        with (
            patch("data.analyst_revisions.today_et", return_value=_today()),
            patch("data.analyst_revisions._load_cache", return_value=cached),
            patch("data.analyst_revisions._save_cache"),
        ):
            from data.analyst_revisions import get_analyst_revisions

            result = get_analyst_revisions(["AAPL", "MSFT"])
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)

    def test_fetches_on_cache_miss_and_saves(self):
        cached = {_TODAY_KEY: {}}
        with (
            patch("data.analyst_revisions.today_et", return_value=_today()),
            patch("data.analyst_revisions._load_cache", return_value=cached),
            patch(
                "data.analyst_revisions._live_fetch_revisions",
                return_value={"AAPL": {"analyst_upgrade": True, "analyst_downgrade": False}},
            ),
            patch("data.analyst_revisions._save_cache") as mock_save,
        ):
            from data.analyst_revisions import get_analyst_revisions

            result = get_analyst_revisions(["AAPL"])
        self.assertIn("AAPL", result)
        mock_save.assert_called_once()

    def test_returns_empty_when_all_none(self):
        cached = {_TODAY_KEY: {"AAPL": None}}
        with (
            patch("data.analyst_revisions.today_et", return_value=_today()),
            patch("data.analyst_revisions._load_cache", return_value=cached),
            patch("data.analyst_revisions._save_cache"),
        ):
            from data.analyst_revisions import get_analyst_revisions

            result = get_analyst_revisions(["AAPL"])
        self.assertEqual(result, {})


class TestParseEpsRevisions(unittest.TestCase):
    """_parse_eps_revisions — downward current-quarter EPS estimate momentum."""

    def _ticker(self, up, down, period="0q"):
        df = pd.DataFrame({"upLast30days": [up], "downLast30days": [down]}, index=[period])
        tk = MagicMock()
        tk.eps_revisions = df
        return tk

    def test_fires_when_downs_cluster_and_exceed_ups(self):
        from data.analyst_revisions import _parse_eps_revisions

        self.assertTrue(_parse_eps_revisions(self._ticker(up=1, down=4)))

    def test_no_fire_below_min_downs(self):
        from data.analyst_revisions import _parse_eps_revisions

        self.assertFalse(_parse_eps_revisions(self._ticker(up=0, down=2)))

    def test_no_fire_when_downs_not_exceed_ups(self):
        from data.analyst_revisions import _parse_eps_revisions

        self.assertFalse(_parse_eps_revisions(self._ticker(up=5, down=3)))

    def test_no_fire_when_no_current_quarter_row(self):
        from data.analyst_revisions import _parse_eps_revisions

        self.assertFalse(_parse_eps_revisions(self._ticker(up=0, down=5, period="+1q")))

    def test_no_fire_when_table_none(self):
        from data.analyst_revisions import _parse_eps_revisions

        tk = MagicMock()
        tk.eps_revisions = None
        self.assertFalse(_parse_eps_revisions(tk))

    def test_no_fire_when_access_raises(self):
        from data.analyst_revisions import _parse_eps_revisions

        class _Raising:
            @property
            def eps_revisions(self):
                raise RuntimeError("boom")

        self.assertFalse(_parse_eps_revisions(_Raising()))

    def test_no_fire_when_row_values_unparseable(self):
        from data.analyst_revisions import _parse_eps_revisions

        df = pd.DataFrame({"upLast30days": ["x"], "downLast30days": ["y"]}, index=["0q"])
        tk = MagicMock()
        tk.eps_revisions = df
        self.assertFalse(_parse_eps_revisions(tk))


class TestLiveFetchEpsPath(unittest.TestCase):
    """_live_fetch_revisions surfaces eps_estimate_cut even with no ratings table."""

    def _eps_df(self, up, down):
        return pd.DataFrame({"upLast30days": [up], "downLast30days": [down]}, index=["0q"])

    def test_eps_only_when_no_recommendations(self):
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = None
        mock_ticker.eps_revisions = self._eps_df(up=0, down=4)
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNotNone(result["AAPL"])
        self.assertTrue(result["AAPL"]["eps_estimate_cut"])
        self.assertFalse(result["AAPL"]["analyst_downgrade"])

    def test_none_when_neither_ratings_nor_eps(self):
        mock_ticker = MagicMock()
        mock_ticker.recommendations_summary = None
        mock_ticker.eps_revisions = self._eps_df(up=2, down=1)  # no cut
        with (
            patch("data.analyst_revisions.ETF_SYMBOLS", set()),
            patch("data.analyst_revisions.yf.Ticker", return_value=mock_ticker),
            patch("data.analyst_revisions.time.sleep"),
        ):
            from data.analyst_revisions import _live_fetch_revisions

            result = _live_fetch_revisions(["AAPL"])
        self.assertIsNone(result["AAPL"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

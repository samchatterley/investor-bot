"""Tests for data/fundamental_cache.py — 100% branch coverage."""

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.fundamental_cache import (
    _compute_altman_z,
    _compute_fcf_yield,
    _compute_gross_margin_trend,
    _compute_piotroski,
    _fetch_symbol,
    _get_field,
    _is_stale,
    _load_cache,
    _row,
    _save_cache,
    _val,
    get_altman_z,
    get_fcf_yield,
    get_forward_pe,
    get_gross_margin_current,
    get_gross_margin_trend,
    get_market_cap,
    get_piotroski_f,
    get_shares_outstanding,
    refresh_fundamental_cache,
)

# ── helpers ────────────────────────────────────────────────────────────────────

TODAY = date(2026, 6, 2)
OLD_DATE = (TODAY - timedelta(days=10)).isoformat()
FRESH_DATE = TODAY.isoformat()


def _df2(cur: dict, pri: dict) -> pd.DataFrame:
    return pd.DataFrame({0: cur, 1: pri})


def _df_multi(cols: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(dict(enumerate(cols)))


def _ticker(fin, bs, cf, info=None, qfin=None, qcf=None):
    t = MagicMock()
    t.financials = fin
    t.balance_sheet = bs
    t.cashflow = cf
    t.info = info or {}
    t.quarterly_financials = qfin if qfin is not None else pd.DataFrame()
    t.quarterly_cashflow = qcf if qcf is not None else pd.DataFrame()
    return t


# Good-company DataFrames for Piotroski all-pass
_GOOD_FIN = _df2(
    {"Net Income": 100.0, "Gross Profit": 400.0, "Total Revenue": 500.0, "EBIT": 150.0},
    {"Net Income": 50.0, "Gross Profit": 280.0, "Total Revenue": 400.0, "EBIT": 80.0},
)
_GOOD_BS = _df2(
    {
        "Total Assets": 1000.0,
        "Long Term Debt": 100.0,
        "Current Assets": 300.0,
        "Current Liabilities": 100.0,
        "Share Issued": 100.0,
        "Working Capital": 200.0,
        "Retained Earnings": 500.0,
        "Total Liabilities Net Minority Interest": 300.0,
    },
    {
        "Total Assets": 900.0,
        "Long Term Debt": 200.0,
        "Current Assets": 200.0,
        "Current Liabilities": 150.0,
        "Share Issued": 100.0,
        "Working Capital": 150.0,
        "Retained Earnings": 400.0,
        "Total Liabilities Net Minority Interest": 350.0,
    },
)
_GOOD_CF = _df2(
    {"Operating Cash Flow": 120.0, "Free Cash Flow": 100.0},
    {"Operating Cash Flow": 60.0, "Free Cash Flow": 50.0},
)

# Bad-company DataFrames (all outer conditions True, all inner comparisons False)
_BAD_FIN = _df2(
    {"Net Income": -100.0, "Gross Profit": 200.0, "Total Revenue": 400.0},
    {"Net Income": -50.0, "Gross Profit": 300.0, "Total Revenue": 500.0},
)
_BAD_BS = _df2(
    {
        "Total Assets": 1000.0,
        "Long Term Debt": 200.0,
        "Current Assets": 100.0,
        "Current Liabilities": 300.0,
        "Share Issued": 200.0,
        "Working Capital": -200.0,
        "Retained Earnings": -500.0,
        "Total Liabilities Net Minority Interest": 800.0,
    },
    {
        "Total Assets": 900.0,
        "Long Term Debt": 100.0,
        "Current Assets": 200.0,
        "Current Liabilities": 100.0,
        "Share Issued": 100.0,
        "Working Capital": 100.0,
        "Retained Earnings": -200.0,
        "Total Liabilities Net Minority Interest": 400.0,
    },
)
_BAD_CF = _df2(
    {"Operating Cash Flow": -150.0, "Free Cash Flow": -120.0},
    {"Operating Cash Flow": -60.0, "Free Cash Flow": -50.0},
)


# ── _load_cache / _save_cache ─────────────────────────────────────────────────


class TestLoadSaveCache:
    def test_load_file_not_found(self, tmp_path):
        with patch("data.fundamental_cache._CACHE_PATH", str(tmp_path / "missing.json")):
            assert _load_cache() == {}

    def test_load_json_decode_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with patch("data.fundamental_cache._CACHE_PATH", str(p)):
            assert _load_cache() == {}

    def test_load_success(self, tmp_path):
        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"AAPL": {"piotroski_f": 7}}))
        with patch("data.fundamental_cache._CACHE_PATH", str(p)):
            assert _load_cache() == {"AAPL": {"piotroski_f": 7}}

    def test_save_success(self, tmp_path):
        p = tmp_path / "cache.json"
        with patch("data.fundamental_cache._CACHE_PATH", str(p)):
            _save_cache({"AAPL": {"piotroski_f": 7}})
            assert json.loads(p.read_text()) == {"AAPL": {"piotroski_f": 7}}

    def test_save_os_error(self, tmp_path):
        with (
            patch("data.fundamental_cache.LOG_DIR", "/nonexistent/path/xyz"),
            patch("data.fundamental_cache._CACHE_PATH", "/nonexistent/path/xyz/f.json"),
        ):
            _save_cache({"x": 1})  # must not raise


# ── _is_stale ─────────────────────────────────────────────────────────────────


class TestIsStale:
    def test_fresh_not_stale(self):
        with patch("data.fundamental_cache.today_et", return_value=TODAY):
            assert _is_stale({"last_updated": FRESH_DATE}) is False

    def test_old_is_stale(self):
        with patch("data.fundamental_cache.today_et", return_value=TODAY):
            assert _is_stale({"last_updated": OLD_DATE}) is True

    def test_missing_key_is_stale(self):
        assert _is_stale({}) is True

    def test_bad_date_format_is_stale(self):
        assert _is_stale({"last_updated": "not-a-date"}) is True


# ── _row / _val ───────────────────────────────────────────────────────────────


class TestRowVal:
    def test_row_finds_first_name(self):
        df = pd.DataFrame({0: {"Alpha": 1.0, "Beta": 2.0}})
        s = _row(df, "Alpha", "Beta")
        assert s is not None
        assert float(s.iloc[0]) == 1.0

    def test_row_finds_second_name(self):
        df = pd.DataFrame({0: {"Beta": 2.0}})
        s = _row(df, "Alpha", "Beta")
        assert s is not None
        assert float(s.iloc[0]) == 2.0

    def test_row_not_found_returns_none(self):
        df = pd.DataFrame({0: {"Gamma": 3.0}})
        assert _row(df, "Alpha", "Beta") is None

    def test_val_none_series_returns_none(self):
        assert _val(None) is None

    def test_val_empty_series_returns_none(self):
        assert _val(pd.Series([], dtype=float)) is None

    def test_val_normal_returns_float(self):
        s = pd.Series([42.0, 10.0])
        assert _val(s, 0) == 42.0

    def test_val_nan_returns_none(self):
        s = pd.Series([float("nan")])
        assert _val(s) is None

    def test_val_non_numeric_returns_none(self):
        s = pd.Series(["hello"])
        assert _val(s) is None


# ── _compute_piotroski ────────────────────────────────────────────────────────


class TestComputePiotroski:
    def test_all_criteria_pass_returns_9(self):
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF)
        assert _compute_piotroski(t) == 9

    def test_all_inner_comparisons_fail_returns_0(self):
        t = _ticker(_BAD_FIN, _BAD_BS, _BAD_CF)
        assert _compute_piotroski(t) == 0

    def test_none_field_values_outer_conditions_skip(self):
        # fin/bs/cf present but only Total Assets rows — all F-score outer conditions False
        fin = _df2({"Total Revenue": 400.0}, {"Total Revenue": 500.0})
        bs = _df2({"Total Assets": 1000.0}, {"Total Assets": 900.0})
        cf = _df2({"Unrelated Row": 50.0}, {"Unrelated Row": 40.0})
        t = _ticker(fin, bs, cf)
        assert _compute_piotroski(t) == 0

    def test_empty_fin_returns_none(self):
        t = _ticker(pd.DataFrame(), _GOOD_BS, _GOOD_CF)
        assert _compute_piotroski(t) is None

    def test_insufficient_columns_returns_none(self):
        # fin has only 1 column → shape[1] < 2
        fin_1col = pd.DataFrame({0: {"Net Income": 100.0}})
        t = _ticker(fin_1col, _GOOD_BS, _GOOD_CF)
        assert _compute_piotroski(t) is None

    def test_zero_total_assets_returns_none(self):
        bs_zero = _df2(
            {"Total Assets": 0.0},
            {"Total Assets": 900.0},
        )
        t = _ticker(_GOOD_FIN, bs_zero, _GOOD_CF)
        assert _compute_piotroski(t) is None

    def test_nan_total_assets_pri_returns_none(self):
        bs_nan = _df2(
            {"Total Assets": 1000.0},
            {"Total Assets": float("nan")},
        )
        t = _ticker(_GOOD_FIN, bs_nan, _GOOD_CF)
        assert _compute_piotroski(t) is None

    def test_f9_revenue_none_skips_f9(self):
        # revenue absent → F9 outer condition False → covers arc 164→168
        fin = _df2({"Net Income": 100.0}, {"Net Income": 50.0})
        bs = _df2({"Total Assets": 1000.0}, {"Total Assets": 900.0})
        cf = _df2({"Unrelated": 50.0}, {"Unrelated": 40.0})
        t = _ticker(fin, bs, cf)
        result = _compute_piotroski(t)
        assert isinstance(result, int)  # F1 and F3 pass, others None/skip

    def test_exception_returns_none(self):
        class Broken:
            @property
            def financials(self):
                raise RuntimeError("boom")

            balance_sheet = pd.DataFrame()
            cashflow = pd.DataFrame()

        assert _compute_piotroski(Broken()) is None


# ── _compute_altman_z ─────────────────────────────────────────────────────────


class TestComputeAltmanZ:
    def test_normal_returns_float(self):
        info = {"marketCap": 2_000_000.0}
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, info=info)
        result = _compute_altman_z(t)
        assert isinstance(result, float)

    def test_empty_bs_returns_none(self):
        t = _ticker(_GOOD_FIN, pd.DataFrame(), _GOOD_CF)
        assert _compute_altman_z(t) is None

    def test_zero_total_assets_returns_none(self):
        bs_zero = _df2({"Total Assets": 0.0}, {"Total Assets": 900.0})
        t = _ticker(_GOOD_FIN, bs_zero, _GOOD_CF, info={"marketCap": 1_000_000})
        assert _compute_altman_z(t) is None

    def test_missing_market_cap_returns_none(self):
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, info={})
        assert _compute_altman_z(t) is None

    def test_missing_retained_earnings_returns_none(self):
        bs_no_re = _df2(
            {
                "Total Assets": 1000.0,
                "Working Capital": 200.0,
                "Total Liabilities Net Minority Interest": 300.0,
            },
            {"Total Assets": 900.0},
        )
        t = _ticker(_GOOD_FIN, bs_no_re, _GOOD_CF, info={"marketCap": 1_000_000})
        assert _compute_altman_z(t) is None

    def test_zero_total_liab_returns_none(self):
        bs_zero_liab = _df2(
            {
                "Total Assets": 1000.0,
                "Working Capital": 200.0,
                "Retained Earnings": 500.0,
                "Total Liabilities Net Minority Interest": 0.0,
            },
            {"Total Assets": 900.0},
        )
        t = _ticker(_GOOD_FIN, bs_zero_liab, _GOOD_CF, info={"marketCap": 1_000_000})
        assert _compute_altman_z(t) is None

    def test_exception_returns_none(self):
        class Broken:
            @property
            def balance_sheet(self):
                raise RuntimeError("boom")

            financials = pd.DataFrame()
            cashflow = pd.DataFrame()
            info = {}

        assert _compute_altman_z(Broken()) is None


# ── _compute_fcf_yield ────────────────────────────────────────────────────────


class TestComputeFcfYield:
    def _make_qcf(self, fcf_values: list[float]) -> pd.DataFrame:
        return _df_multi([{"Free Cash Flow": v} for v in fcf_values])

    def test_quarterly_4_periods_returns_value(self):
        qcf = self._make_qcf([25.0, 30.0, 20.0, 25.0])
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, info={"marketCap": 1_000.0}, qcf=qcf)
        result = _compute_fcf_yield(t)
        assert isinstance(result, float)
        assert result == round(100.0 / 1_000.0, 4)

    def test_no_market_cap_returns_none(self):
        qcf = self._make_qcf([25.0, 30.0, 20.0, 25.0])
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, info={}, qcf=qcf)
        assert _compute_fcf_yield(t) is None

    def test_quarterly_with_nan_returns_none(self):
        qcf = self._make_qcf([25.0, float("nan"), 20.0, 25.0])
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, info={"marketCap": 1000.0}, qcf=qcf)
        assert _compute_fcf_yield(t) is None

    def test_fewer_than_4_quarters_falls_back_to_annual(self):
        qcf = self._make_qcf([25.0, 30.0])  # only 2 quarters
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, info={"marketCap": 1_000.0}, qcf=qcf)
        result = _compute_fcf_yield(t)
        assert isinstance(result, float)  # uses annual FCF = 100.0 → 0.1

    def test_fewer_than_4_quarters_annual_cf_empty_returns_none(self):
        qcf = self._make_qcf([25.0])
        t = _ticker(_GOOD_FIN, _GOOD_BS, pd.DataFrame(), info={"marketCap": 1_000.0}, qcf=qcf)
        assert _compute_fcf_yield(t) is None

    def test_fewer_than_4_quarters_annual_fcf_none_returns_none(self):
        qcf = self._make_qcf([25.0])
        cf_no_fcf = _df2({"Operating Cash Flow": 100.0}, {"Operating Cash Flow": 80.0})
        t = _ticker(_GOOD_FIN, _GOOD_BS, cf_no_fcf, info={"marketCap": 1_000.0}, qcf=qcf)
        assert _compute_fcf_yield(t) is None

    def test_exception_returns_none(self):
        class Broken:
            @property
            def info(self):
                raise RuntimeError("boom")

            financials = pd.DataFrame()
            balance_sheet = pd.DataFrame()
            cashflow = pd.DataFrame()
            quarterly_cashflow = pd.DataFrame()

        assert _compute_fcf_yield(Broken()) is None


# ── _compute_gross_margin_trend ───────────────────────────────────────────────


class TestComputeGrossMarginTrend:
    def _make_qfin(self, gp_vals: list, rev_vals: list) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        for i, (gp, rev) in enumerate(zip(gp_vals, rev_vals, strict=False)):
            rows[i] = {"Gross Profit": gp, "Total Revenue": rev}
        return pd.DataFrame(rows)

    def test_normal_returns_tuple(self):
        qfin = self._make_qfin(
            [400.0, 380.0, 360.0, 350.0],
            [500.0, 480.0, 460.0, 450.0],
        )
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, qfin=qfin)
        cur, avg, trend = _compute_gross_margin_trend(t)
        assert cur is not None
        assert avg is not None
        assert isinstance(trend, float)

    def test_gp_row_missing_returns_nones(self):
        qfin = _df_multi([{"Total Revenue": 500.0}] * 4)
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, qfin=qfin)
        assert _compute_gross_margin_trend(t) == (None, None, None)

    def test_insufficient_periods_returns_nones(self):
        qfin = self._make_qfin([400.0, 380.0], [500.0, 480.0])  # only 2 periods
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, qfin=qfin)
        assert _compute_gross_margin_trend(t) == (None, None, None)

    def test_too_many_nan_margins_returns_nones(self):
        # 4 periods but 2 have zero revenue → margin is None → valid < 4
        qfin = self._make_qfin(
            [400.0, 0.0, 0.0, 350.0],
            [500.0, 0.0, 0.0, 450.0],  # zero revenue → division skipped → None margin
        )
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, qfin=qfin)
        assert _compute_gross_margin_trend(t) == (None, None, None)

    def test_exception_returns_nones(self):
        class Broken:
            @property
            def quarterly_financials(self):
                raise RuntimeError("boom")

            financials = pd.DataFrame()
            balance_sheet = pd.DataFrame()
            cashflow = pd.DataFrame()

        assert _compute_gross_margin_trend(Broken()) == (None, None, None)


# ── _fetch_symbol ─────────────────────────────────────────────────────────────


class TestFetchSymbol:
    def test_normal_returns_dict_with_all_keys(self):
        qcf = _df_multi([{"Free Cash Flow": 25.0}] * 4)
        qfin_data = _df_multi([{"Gross Profit": 400.0, "Total Revenue": 500.0}] * 5)
        info = {"forwardPE": 25.0, "sharesOutstanding": 1_000, "marketCap": 10_000.0}
        t = _ticker(_GOOD_FIN, _GOOD_BS, _GOOD_CF, info=info, qfin=qfin_data, qcf=qcf)

        with (
            patch("data.fundamental_cache.yf.Ticker", return_value=t),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            result = _fetch_symbol("AAPL")

        assert "piotroski_f" in result
        assert "altman_z" in result
        assert "fcf_yield" in result
        assert "gross_margin_trend" in result
        assert "forward_pe" in result
        assert result["last_updated"] == TODAY.isoformat()

    def test_exception_returns_empty_dict(self):
        with patch("data.fundamental_cache.yf.Ticker", side_effect=RuntimeError("boom")):
            assert _fetch_symbol("AAPL") == {}


# ── refresh_fundamental_cache ─────────────────────────────────────────────────


class TestRefreshFundamentalCache:
    def test_symbols_none_uses_stock_universe(self):
        with (
            patch("data.fundamental_cache._load_cache", return_value={}),
            patch("data.fundamental_cache._save_cache") as mock_save,
            patch("data.fundamental_cache._fetch_symbol", return_value={}),
            patch("data.fundamental_cache.STOCK_UNIVERSE", ["AAPL", "MSFT"]),
        ):
            n = refresh_fundamental_cache(symbols=None)
        assert n == 0  # fetch returns {} so nothing saved
        mock_save.assert_not_called()

    def test_skips_fresh_entry(self):
        cache = {"AAPL": {"last_updated": FRESH_DATE, "piotroski_f": 7}}
        with (
            patch("data.fundamental_cache._load_cache", return_value=cache),
            patch("data.fundamental_cache._save_cache") as mock_save,
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            n = refresh_fundamental_cache(["AAPL"])
        assert n == 0
        mock_save.assert_not_called()

    def test_refreshes_stale_entry_with_data(self):
        cache = {"AAPL": {"last_updated": OLD_DATE}}
        new_data = {"piotroski_f": 6, "last_updated": FRESH_DATE}
        with (
            patch("data.fundamental_cache._load_cache", return_value=cache),
            patch("data.fundamental_cache._save_cache") as mock_save,
            patch("data.fundamental_cache._fetch_symbol", return_value=new_data),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            n = refresh_fundamental_cache(["AAPL"])
        assert n == 1
        mock_save.assert_called_once()

    def test_stale_entry_no_data_not_counted(self):
        cache = {"AAPL": {"last_updated": OLD_DATE}}
        with (
            patch("data.fundamental_cache._load_cache", return_value=cache),
            patch("data.fundamental_cache._save_cache") as mock_save,
            patch("data.fundamental_cache._fetch_symbol", return_value={}),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            n = refresh_fundamental_cache(["AAPL"])
        assert n == 0
        mock_save.assert_not_called()

    def test_force_refreshes_fresh_entry(self):
        cache = {"AAPL": {"last_updated": FRESH_DATE, "piotroski_f": 7}}
        new_data = {"piotroski_f": 8, "last_updated": FRESH_DATE}
        with (
            patch("data.fundamental_cache._load_cache", return_value=cache),
            patch("data.fundamental_cache._save_cache") as mock_save,
            patch("data.fundamental_cache._fetch_symbol", return_value=new_data),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            n = refresh_fundamental_cache(["AAPL"], force=True)
        assert n == 1
        mock_save.assert_called_once()


# ── _get_field ────────────────────────────────────────────────────────────────


class TestGetField:
    def test_fresh_cache_returns_field(self):
        cache = {"AAPL": {"last_updated": FRESH_DATE, "piotroski_f": 7}}
        with (
            patch("data.fundamental_cache._load_cache", return_value=cache),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            result = _get_field("AAPL", "piotroski_f")
        assert result == 7

    def test_stale_entry_triggers_refresh_and_saves(self):
        cache = {"AAPL": {"last_updated": OLD_DATE, "piotroski_f": 5}}
        new_data = {"piotroski_f": 8, "last_updated": FRESH_DATE}
        with (
            patch("data.fundamental_cache._load_cache", return_value=cache),
            patch("data.fundamental_cache._save_cache") as mock_save,
            patch("data.fundamental_cache._fetch_symbol", return_value=new_data),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            result = _get_field("AAPL", "piotroski_f")
        assert result == 8
        mock_save.assert_called_once()

    def test_sym_not_in_cache_fetches_and_saves(self):
        new_data = {"piotroski_f": 6, "last_updated": FRESH_DATE}
        with (
            patch("data.fundamental_cache._load_cache", return_value={}),
            patch("data.fundamental_cache._save_cache") as mock_save,
            patch("data.fundamental_cache._fetch_symbol", return_value=new_data),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            result = _get_field("AAPL", "piotroski_f")
        assert result == 6
        mock_save.assert_called_once()

    def test_fetch_empty_returns_none(self):
        with (
            patch("data.fundamental_cache._load_cache", return_value={}),
            patch("data.fundamental_cache._save_cache"),
            patch("data.fundamental_cache._fetch_symbol", return_value={}),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            result = _get_field("AAPL", "piotroski_f")
        assert result is None

    def test_field_missing_from_entry_returns_none(self):
        cache = {"AAPL": {"last_updated": FRESH_DATE, "altman_z": 3.5}}
        with (
            patch("data.fundamental_cache._load_cache", return_value=cache),
            patch("data.fundamental_cache.today_et", return_value=TODAY),
        ):
            result = _get_field("AAPL", "piotroski_f")
        assert result is None


# ── public getters ────────────────────────────────────────────────────────────


class TestPublicGetters:
    def _patched(self, field: str, value):
        return patch("data.fundamental_cache._get_field", return_value=value)

    def test_get_piotroski_f_returns_int(self):
        with self._patched("piotroski_f", 7):
            assert get_piotroski_f("AAPL") == 7
            assert isinstance(get_piotroski_f("AAPL"), int)

    def test_get_piotroski_f_none(self):
        with self._patched("piotroski_f", None):
            assert get_piotroski_f("AAPL") is None

    def test_get_altman_z_returns_float(self):
        with self._patched("altman_z", 3.5):
            assert get_altman_z("AAPL") == 3.5

    def test_get_altman_z_none(self):
        with self._patched("altman_z", None):
            assert get_altman_z("AAPL") is None

    def test_get_fcf_yield_returns_float(self):
        with self._patched("fcf_yield", 0.04):
            assert get_fcf_yield("AAPL") == pytest.approx(0.04)

    def test_get_fcf_yield_none(self):
        with self._patched("fcf_yield", None):
            assert get_fcf_yield("AAPL") is None

    def test_get_gross_margin_trend_returns_float(self):
        with self._patched("gross_margin_trend", 0.02):
            assert get_gross_margin_trend("AAPL") == pytest.approx(0.02)

    def test_get_gross_margin_trend_none(self):
        with self._patched("gross_margin_trend", None):
            assert get_gross_margin_trend("AAPL") is None

    def test_get_gross_margin_current_returns_float(self):
        with self._patched("gross_margin_current", 0.44):
            assert get_gross_margin_current("AAPL") == pytest.approx(0.44)

    def test_get_gross_margin_current_none(self):
        with self._patched("gross_margin_current", None):
            assert get_gross_margin_current("AAPL") is None

    def test_get_forward_pe_returns_float(self):
        with self._patched("forward_pe", 28.5):
            assert get_forward_pe("AAPL") == pytest.approx(28.5)

    def test_get_forward_pe_none(self):
        with self._patched("forward_pe", None):
            assert get_forward_pe("AAPL") is None

    def test_get_shares_outstanding_returns_int(self):
        with self._patched("shares_outstanding", 1_000_000):
            assert get_shares_outstanding("AAPL") == 1_000_000
            assert isinstance(get_shares_outstanding("AAPL"), int)

    def test_get_shares_outstanding_none(self):
        with self._patched("shares_outstanding", None):
            assert get_shares_outstanding("AAPL") is None

    def test_get_market_cap_returns_float(self):
        with self._patched("market_cap", 3_000_000_000.0):
            assert get_market_cap("AAPL") == 3_000_000_000.0

    def test_get_market_cap_none(self):
        with self._patched("market_cap", None):
            assert get_market_cap("AAPL") is None

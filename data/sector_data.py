import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Technology",
    "NVDA": "Technology",
    "TSLA": "Consumer Discretionary",
    "AMD": "Technology",
    "NFLX": "Consumer Discretionary",
    "CRM": "Technology",
    "ADBE": "Technology",
    "UBER": "Industrials",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "SPY": "ETF",
    "QQQ": "ETF",
    "IWM": "ETF",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "V": "Financials",
    "MA": "Financials",
    "PYPL": "Financials",
    "SHOP": "Technology",
    "COIN": "Financials",
    "SNAP": "Technology",
    "PINS": "Technology",
    "RBLX": "Technology",
    "DIS": "Consumer Discretionary",
    "PFE": "Healthcare",
    "MRNA": "Healthcare",
    "ABBV": "Healthcare",
    "LLY": "Healthcare",
    "BA": "Industrials",
    "CAT": "Industrials",
    "GE": "Industrials",
    "LMT": "Industrials",
    "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary",
    "PLTR": "Technology",
    "SNOW": "Technology",
    "NET": "Technology",
    "DDOG": "Technology",
    "CRWD": "Technology",
    "INTC": "Technology",
    "QCOM": "Technology",
    "AVGO": "Technology",
    "TXN": "Technology",
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
}


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Unknown")


def get_sector_performance(days: int = 5) -> dict[str, float]:
    """Return {sector_name: 5d_return_pct} for major sectors, sorted best to worst."""
    try:
        etfs = list(SECTOR_ETFS.values())
        data = yf.download(
            etfs, period=f"{days + 10}d", interval="1d", progress=False, auto_adjust=True
        )
        if data.empty or len(data) < 2:
            return {}

        close = data["Close"]
        perf = {}
        for sector, etf in SECTOR_ETFS.items():
            if etf in close.columns:
                ret = (close[etf].iloc[-1] / close[etf].iloc[-days] - 1) * 100
                if not pd.isna(ret):
                    perf[sector] = round(float(ret), 2)

        return dict(sorted(perf.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        logger.error(f"Sector performance fetch failed: {e}")
        return {}


def get_leading_sectors(top_n: int = 3) -> list[str]:
    perf = get_sector_performance()
    return list(perf.keys())[:top_n]


def check_sector_concentration(symbols: list[str], max_per_sector: int = 2) -> list[str]:
    """
    Return symbols that would breach the sector concentration cap.
    Call with current held + proposed new symbols.
    """
    sector_counts: dict[str, list[str]] = {}
    for sym in symbols:
        s = get_sector(sym)
        sector_counts.setdefault(s, []).append(sym)

    breaches = []
    for sector, syms in sector_counts.items():
        if sector == "ETF":
            continue
        if len(syms) > max_per_sector:
            # Flag the excess symbols (keep the first max_per_sector)
            breaches.extend(syms[max_per_sector:])
    return breaches

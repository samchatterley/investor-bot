"""Ticker symbol normalisation helpers."""

from __future__ import annotations


def to_yf_symbol(symbol: str) -> str:
    """Normalise a ticker to Yahoo Finance's convention for class/preferred shares.

    The bot's universe and Alpaca use a dot for share classes (BRK.B, BF.B); Yahoo Finance uses a
    hyphen (BRK-B, BF-B) and returns *zero rows* for the dot form. Passing the dot ticker straight to
    yfinance therefore silently drops those names from every yfinance-backed feed (prices, earnings,
    short interest, news). Apply this only at the yfinance query boundary and key the results back to
    the original symbol — the rest of the system continues to use the dot form.
    """
    return symbol.replace(".", "-")

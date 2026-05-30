import logging

import yfinance as yf

from config import ETF_SYMBOLS, MIN_VOLUME
from data.market_regime import fetch_spy_vix_history, load_regime_state, save_regime_state
from data.market_regime import get_market_regime as _compute_regime
from signals.evaluator import REGIME_BLOCKED, SHORT_ALLOWED_REGIMES, evaluate_signals

logger = logging.getLogger(__name__)

# Broader universe used only for the daily top-movers scan.
# Intentionally sector-diverse and mid-cap-heavy so the momentum×volume scorer
# can surface "sleeper" moves beyond mega-caps.
EXTENDED_UNIVERSE = list(
    dict.fromkeys(
        [
            # Mega-cap tech (anchor names)
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "AMD",
            "NFLX",
            "CRM",
            "ADBE",
            "UBER",
            # Semiconductors — mid/large
            "INTC",
            "QCOM",
            "AVGO",
            "TXN",
            "MU",
            "SMCI",
            "MRVL",
            "ON",
            "MPWR",
            "LRCX",
            "KLAC",
            "AMAT",
            "SWKS",
            # Software / SaaS growth
            "PLTR",
            "SNOW",
            "NET",
            "DDOG",
            "CRWD",
            "BILL",
            "APP",
            "GTLB",
            "PCOR",
            "MNDY",
            "HUBS",
            "TEAM",
            "ZS",
            "PANW",
            # Fintech & payments
            "V",
            "MA",
            "PYPL",
            "SHOP",
            "COIN",
            "AFRM",
            "SOFI",
            "HOOD",
            # Financials
            "JPM",
            "BAC",
            "GS",
            "MS",
            "C",
            "SCHW",
            # Energy — oil & clean
            "XOM",
            "CVX",
            "OXY",
            "ENPH",
            "SEDG",
            "FSLR",
            "RUN",
            "PLUG",
            "BE",
            # Healthcare & biotech
            "PFE",
            "MRNA",
            "ABBV",
            "LLY",
            "JNJ",
            "BMY",
            "NBIX",
            "BMRN",
            "INCY",
            "SRPT",
            "ISRG",
            "DXCM",
            "PODD",
            # Defense — large & mid
            "LMT",
            "RTX",
            "NOC",
            "BA",
            "HII",
            "KTOS",
            "RKLB",
            "LDOS",
            # Industrials & infrastructure
            "CAT",
            "GE",
            "HON",
            "EMR",
            "GNRC",
            "AXON",
            "SAIA",
            "XYL",
            # Consumer discretionary
            "NKE",
            "MCD",
            "SBUX",
            "TGT",
            "COST",
            "WMT",
            "HD",
            "DKNG",
            "LYFT",
            "DASH",
            "CHWY",
            "W",
            # Entertainment & media
            "DIS",
            "SNAP",
            "PINS",
            "RBLX",
            # Commodities & materials
            "FCX",
            "AA",
            "MP",
            "CLF",
            # Crypto-adjacent (high volatility movers)
            "MSTR",
            "MARA",
            "RIOT",
            # Broad market ETFs (baseline comparison)
            "SPY",
            "QQQ",
            "IWM",
        ]
    )
)


def get_market_regime(threshold_pct: float = -1.5, vix: float | None = None) -> dict:
    """5-state regime classifier backed by data.market_regime.

    threshold_pct and vix are accepted for backward-compatible call sites
    but the shared module uses its own RegimeThresholds defaults.
    Returns a backward-compatible dict (is_bearish, spy_change_pct, spy_5d_pct, regime).
    """
    try:
        spy_df, vix_df = fetch_spy_vix_history()
        previous = load_regime_state()
        snapshot = _compute_regime(spy_df, vix_df, previous=previous)
        save_regime_state(snapshot)
        result = snapshot.to_dict()
        logger.info(
            f"SPY 1d: {result['spy_change_pct']:+.2f}%  5d: {result['spy_5d_pct']:+.2f}%"
            f"  Regime: {result['regime']}  ({result['data_quality']})"
        )
        return result
    except Exception as e:
        logger.error(f"Market regime check failed: {e}")
        return {"is_bearish": False, "spy_change_pct": 0.0, "spy_5d_pct": 0.0, "regime": "UNKNOWN"}


def _passes_quality_screen(snapshot: dict) -> bool:
    """Fundamental quality gate. Permissive when fields are absent (ETFs, backtest mode).

    Rejects stocks with: negative ROE, negative profit margins, or extreme leverage (D/E > 300).
    D/E threshold of 300 is intentionally loose to accommodate financials and REITs.
    """
    roe = snapshot.get("roe")
    profit_margin = snapshot.get("profit_margin")
    debt_to_equity = snapshot.get("debt_to_equity")

    if roe is None and profit_margin is None:
        return True

    if roe is not None and roe < 0:
        return False

    if profit_margin is not None and profit_margin < 0:
        return False

    return not (debt_to_equity is not None and debt_to_equity > 300)


def prefilter_candidates(snapshots: list[dict], regime: str | None = None) -> list[dict]:
    """
    Rule-based screen applied before Claude analysis.
    Signal logic delegated to signals.evaluator — single canonical implementation
    shared with the backtest engine to prevent live/backtest divergence.
    Stocks trading against their weekly trend are blocked unless deeply oversold.
    """
    _blocked: frozenset[str] = (
        REGIME_BLOCKED.get(regime or "", frozenset()) if regime else frozenset()
    )
    qualified = []
    for s in snapshots:
        if s.get("avg_volume", 0) < MIN_VOLUME:
            continue

        if not _passes_quality_screen(s):
            continue

        matched = evaluate_signals(s, blocked=_blocked)

        if not matched:
            continue

        # Weekly trend filter: block buys counter-trend unless deeply oversold or
        # fundamental conviction present (insider cluster / post-earnings drift).
        rsi = s.get("rsi_14", 50)
        bb = s.get("bb_pct", 0.5)
        weekly_up = s.get("weekly_trend_up", True)
        if (
            not weekly_up
            and not (rsi < 30 and bb < 0.15)
            and "insider_buying" not in matched
            and "pead" not in matched
        ):
            continue

        qualified.append({**s, "matched_signals": matched})

    return qualified


def score_candidate(snapshot: dict) -> float:
    """Deterministic score for baseline comparison against Claude's selections.

    Higher score = stronger setup by technical rules alone.
    Used to compare which candidates Claude selected vs. the rule-ranked order.
    """
    rsi = snapshot.get("rsi_14", 50)
    bb = snapshot.get("bb_pct", 0.5)
    vol = snapshot.get("vol_ratio", 1.0)
    rel = snapshot.get("rel_strength_5d", 0.0)
    n_signals = len(snapshot.get("matched_signals", []))

    rsi_dist = abs(rsi - 50) / 50
    bb_dist = abs(bb - 0.5) * 2
    vol_score = min((vol - 1.0) / 1.0, 1.0) if vol > 1.0 else 0.0
    rel_score = min(max(rel, 0.0), 5.0) / 5.0
    sig_score = n_signals / 8

    return round(
        rsi_dist * 0.25 + bb_dist * 0.20 + vol_score * 0.25 + rel_score * 0.20 + sig_score * 0.10,
        4,
    )


def get_top_movers(n: int = 10) -> list[str]:
    """Return top n symbols from the extended universe by momentum × volume score today."""
    try:
        data = yf.download(
            EXTENDED_UNIVERSE,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty or len(data) < 2:
            return []

        close = data["Close"]
        volume = data["Volume"]

        ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        vol_avg = volume.iloc[:-1].mean()
        vol_ratio = volume.iloc[-1] / vol_avg.replace(0, 1)

        # Score rewards large moves with above-average volume
        score = ret_1d.abs() * vol_ratio.clip(lower=0.5)
        top = score.dropna().nlargest(n).index.tolist()
        logger.info(f"Top movers: {top}")
        return top
    except Exception as e:
        logger.error(f"Top movers scan failed: {e}")
        return []


def scan_short_candidates(
    snapshots: list[dict],
    regime: str | None,
    held_symbols: set[str],
) -> list[dict]:
    """Return short candidates via two distinct paths.

    Regime gate: only runs in SHORT_ALLOWED_REGIMES (stress/downtrend regimes).

    Path A — Reversal (rs_rank_pct >= 65): recently-strong stocks showing exhaustion.
      Gate: not ETF, min volume, rs_rank >= 65.
      Signals checked: failed_breakout, high_vol_reversal.

    Path B — Fundamental deterioration (rs_rank_pct < 25): bottom-quartile laggards
      with an earnings-miss catalyst. No technical overlap with reversal setup.
      Gate: not ETF, min volume, rs_rank < 25, price below EMA21.
      Signals checked: earnings_miss.

    Returns candidates sorted by signal count descending, then rs_rank_pct ascending
    (for path A: highest RS first; for path B: lowest RS first — both handled by the
    sort key which puts most-signals first regardless of direction).
    """
    from signals.evaluator import SHORT_SIGNAL_PRIORITY, evaluate_short_signals

    if regime not in SHORT_ALLOWED_REGIMES:
        return []

    candidates = []
    for s in snapshots:
        symbol = s.get("symbol", "")
        if symbol in held_symbols:
            continue
        if symbol in ETF_SYMBOLS:
            continue
        if s.get("avg_volume", 0) < MIN_VOLUME:
            continue

        rs_rank = s.get("rs_rank_pct")

        # Path A: reversal — recently strong, now showing exhaustion
        if rs_rank is not None and rs_rank >= 65.0:
            rev_signals = evaluate_short_signals(
                s, blocked=frozenset({"earnings_miss", "high_short_interest"})
            )
            if rev_signals:
                confidence = int(len(rev_signals) / len(SHORT_SIGNAL_PRIORITY) * 10)
                candidates.append(
                    {
                        **s,
                        "matched_signals": rev_signals,
                        "key_signal": rev_signals[0],
                        "confidence": confidence,
                    }
                )
            continue

        # Path B: fundamental — bottom-quartile laggard with earnings miss catalyst
        if rs_rank is None or rs_rank >= 25.0:
            continue
        if s.get("price_vs_ema21_pct", 0.0) >= 0:
            continue
        fund_signals = evaluate_short_signals(
            s, blocked=frozenset({"failed_breakout", "high_vol_reversal", "high_short_interest"})
        )
        if fund_signals:
            confidence = int(len(fund_signals) / len(SHORT_SIGNAL_PRIORITY) * 10)
            candidates.append(
                {
                    **s,
                    "matched_signals": fund_signals,
                    "key_signal": fund_signals[0],
                    "confidence": confidence,
                }
            )

    return sorted(
        candidates,
        key=lambda x: (-len(x.get("matched_signals", [])), x.get("rs_rank_pct", 0.0)),
    )

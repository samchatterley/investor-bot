import contextlib
import json
import os
from datetime import date, datetime

import pytz
from dotenv import load_dotenv

# Final holdout period — never used for parameter tuning.
# Walk-forward and signal analysis must use dates strictly before this.
# Call run_holdout_evaluation() only once per strategy version to preserve validity.
HOLDOUT_START_DATE: date = date(2024, 1, 1)

# Canonical start for full-history backtest runs (2015 includes two full market cycles:
# 2018 Q4 drawdown, 2020 COVID crash, 2022 bear market).
BACKTEST_DEFAULT_START: str = "2015-01-01"

# Adaptive prompt: when True the AI prompt includes the outcome-derived blocks (weekly-review lessons
# and performance feedback) — the bot's self-learning loop. The experiment FREEZES this (sets it
# False) during the core context-measurement window so the contextual arm is stationary; the loop's
# own value is then measured as a separate, pre-registered ablation (lessons on vs off). Each decision
# logs which mode was in effect (experiment/collection.py). See docs/EXPERIMENT.md.
ADAPTIVE_PROMPT_ENABLED: bool = True

load_dotenv()


def today_et() -> date:
    """Return today's date in US Eastern time — use this for all market-date comparisons."""
    return datetime.now(pytz.timezone(MARKET_TIMEZONE)).date()


ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Explicit trading mode — set TRADING_MODE=live to enable live trading.
# If TRADING_MODE is set, the URL is validated against the expected Alpaca endpoint.
# If unset, falls back to substring detection for backward compatibility.
_TRADING_MODE = os.getenv("TRADING_MODE", "").lower()


def _validate_trading_mode(mode: str, url: str) -> bool:
    """Validate TRADING_MODE against ALPACA_BASE_URL. Returns True for paper, False for live.

    Raises ValueError if the mode/URL combination is invalid.
    """
    if mode == "paper":
        _expected = "https://paper-api.alpaca.markets"
        if url.rstrip("/") != _expected:
            raise ValueError(
                f"ALPACA_BASE_URL={url!r} does not match TRADING_MODE=paper. Expected {_expected!r}"
            )
        return True
    elif mode == "live":
        _expected = "https://api.alpaca.markets"
        if url.rstrip("/") != _expected:
            raise ValueError(
                f"ALPACA_BASE_URL={url!r} does not match TRADING_MODE=live. Expected {_expected!r}"
            )
        return False
    elif mode:
        raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got {mode!r}")
    else:
        return "paper" in url


IS_PAPER = _validate_trading_mode(_TRADING_MODE, ALPACA_BASE_URL)

# Small-account experiment mode — set SMALL_ACCOUNT_MODE=true in .env for a £150-scale live run.
# When active, caps and risk parameters default to values appropriate for a <$200 account.
# Any individual env var still overrides the small-account default.
# Defined here (before position sizing) so _SAM is available for all inline defaults below.
SMALL_ACCOUNT_MODE = os.getenv("SMALL_ACCOUNT_MODE", "false").lower() == "true"
_SAM = SMALL_ACCOUNT_MODE  # shorthand for inline defaults below

# Position sizing
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "2" if _SAM else "5"))
MAX_POSITION_PCT = 0.45  # Deprecated — legacy Kelly cap; kept only for config.validate(). Superseded by MAX_POSITION_WEIGHT.
CASH_RESERVE_PCT = 0.10  # Always keep 10% as cash buffer

# Short selling
MAX_SHORT_POSITIONS = int(os.getenv("MAX_SHORT_POSITIONS", "3"))
SHORT_SIZE_SCALE = float(os.getenv("SHORT_SIZE_SCALE", "0.5"))  # fraction of standard long size
MAX_SHORT_HEDGE_RATIO = float(
    os.getenv("MAX_SHORT_HEDGE_RATIO", "0.5")
)  # short notional / long notional ceiling
MAX_SHORT_HOLD_DAYS = int(os.getenv("MAX_SHORT_HOLD_DAYS", "3"))
MAX_SHORT_STANDALONE_RATIO = float(
    os.getenv("MAX_SHORT_STANDALONE_RATIO", "0.3")
)  # max short notional as fraction of portfolio when no longs held (bear regimes only)

# Index regime hedge — short an index ETF in confirmed bear regimes as a portfolio-level
# hedge (v1.99). Index ETFs are cheap-to-borrow, deep, and carry no single-name squeeze
# risk — a structurally cleaner short than crowded single names. DISABLED by default: this
# is a new live order path and must be explicitly opted into.
INDEX_HEDGE_ENABLED = os.getenv("INDEX_HEDGE_ENABLED", "false").lower() == "true"
INDEX_HEDGE_SYMBOL = os.getenv("INDEX_HEDGE_SYMBOL", "SPY")
INDEX_HEDGE_WEIGHT = float(os.getenv("INDEX_HEDGE_WEIGHT", "0.10"))  # portfolio fraction to short
INDEX_HEDGE_REGIMES = frozenset(
    s.strip()
    for s in os.getenv("INDEX_HEDGE_REGIMES", "STRESS_RISK_OFF,HIGH_VOL_DOWNTREND").split(",")
    if s.strip()
)

# Risk-budget sizing (replaces Kelly)
RISK_PER_TRADE_PCT = 0.006  # 0.6% of equity risked per trade
MAX_POSITION_WEIGHT = 0.15  # 15% of portfolio per position (hard cap)

# Risk management
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.07" if _SAM else "0.04"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.20" if _SAM else "0.15"))
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "7.0" if _SAM else "4.0"))
SLIPPAGE_BPS = 5  # one-way market impact estimate (basis points)
SPREAD_BPS = 3  # half-spread applied to each side (basis points)
KELLY_MULTIPLIER = 0.5  # half-Kelly — kept for research telemetry only

# AI decision threshold
MIN_CONFIDENCE = 7  # Min confidence score (1-10) to open a position

# Churn guard (audit F4): a *discretionary* exit of a position opened the SAME day is allowed only
# on very-high conviction (>= this) or a hard new negative catalyst — otherwise the AI must HOLD.
# Stop-losses, trailing stops, stale-age, adverse-volume and regime exits are unaffected.
SAME_DAY_SELL_MIN_CONFIDENCE = 9

# Position hold limit — auto-exit after this many trading days
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "5" if _SAM else "3"))

# Per-signal hold limits override MAX_HOLD_DAYS for specific entry types.
# Momentum and trend trades need room to develop; mean-reversion and news
# catalysts play out faster and should be exited sooner.
#
# F2: only ACTIVE long signals + "unknown" are listed. Disabled signals (rs_leader,
# breakout_52w, vix_fear_reversion, momentum_12_1, range_reversion, volume_climax_reversal,
# obv_divergence, obv_acceleration, tax_loss_reversal) and never-defined ones (rsi_oversold,
# news_catalyst, trend_continuation) were pruned — lookups use .get(signal, MAX_HOLD_DAYS).
SIGNAL_MAX_HOLD_DAYS: dict[str, int] = {
    "mean_reversion": 2,
    "macd_crossover": 4,
    "momentum": 5,
    "bb_squeeze": 4,  # volatility squeeze → expansion; hold for the move
    "inside_day_breakout": 3,  # short-duration coil play
    "trend_pullback": 3,  # quick bounce off EMA in uptrend
    "vwap_reclaim": 1,  # intraday flow signal — exit same day or next open
    "orb_breakout": 1,  # intraday breakout — hold expires at next open
    "intraday_momentum": 1,  # intraday continuation — exit same day or next open
    "gap_and_go": 2,  # confirmed gap continuation — typically resolves in 1–2 days
    "insider_buying": 5,  # cluster insider purchases — drift plays out over days-weeks
    "pead": 3,  # post-earnings drift — capture the initial repricing window
    "iv_compression": 4,  # vol squeeze → expansion; hold for the directional move
    # ── Batch 1: OHLCV technical signals ────────────────────────────────────
    "golden_cross": 5,  # 50d/200d trend-following; needs room to develop
    "candle_exhaustion": 3,  # reversal at exhaustion candle; typically 2-3 day
    # ── Batch 2: universe-level signals ──────────────────────────────────────
    "breadth_thrust": 4,  # broad-market thrust; hold for the continuation move
    # ── Batch 4: fundamental quality signals ─────────────────────────────────
    "fcf_yield_signal": 5,  # fundamental value; drift plays out over days-weeks
    # ── Batch 5: options-derived signals ─────────────────────────────────────
    "options_skew_signal": 3,  # skew normalisation usually 1-3 days
    "unusual_options_activity": 3,  # OTM call build-up resolves within 1-3 days
    "put_call_contrarian": 3,  # contrarian signal; relief rally 2-3 days
    "iv_vs_rv_spread": 4,  # vol-regime normalisation; 3-4 day window
    # ── Batch 6: short-squeeze signals ───────────────────────────────────────
    "squeeze_setup_long": 5,  # pre-squeeze dormant; hold for the catalyst
    "squeeze_momentum_long": 4,  # active squeeze; move resolves in 3-4 days
    "short_interest_trend_long": 5,  # short-cover drift; multi-day momentum
    # ── Batch 7: analyst signals ─────────────────────────────────────────────
    "analyst_upgrade_signal": 5,  # consensus revision drift; plays out over days
    # ── Batch 8: sentiment signals ────────────────────────────────────────────
    "aaii_extreme_fear_long": 3,  # contrarian; typically resolves in 2-3 days
    "fear_greed_extreme_fear": 3,  # composite fear reversal; 2-3 day window
    # ── Batch 9: cross-asset signals ─────────────────────────────────────────
    "sector_pair_mean_reversion": 5,  # pairs spread reversion; 3-5 day window
    # ── Batch 10: alternative data ───────────────────────────────────────────
    "google_trends_bullish": 3,  # attention-driven; typically short-lived
    # ── Fundamental conviction (new) ─────────────────────────────────────────
    "activist_13d_signal": 5,  # activist catalyst; medium-term hold
    "guidance_raise_signal": 3,  # guidance event; catalyst captures initial drift
    "unknown": 3,  # conservative default
}

# Bear market filter — skip new buys when SPY drops more than this % in a single day
BEAR_MARKET_SPY_THRESHOLD = -1.5

# How many top movers to add to the daily scan universe
TOP_MOVERS_COUNT = 15

# Partial profit taking — sell half position when unrealised gain hits this %
PARTIAL_PROFIT_PCT = float(os.getenv("PARTIAL_PROFIT_PCT", "15.0" if _SAM else "8.0"))

# Earnings guard — exit positions with earnings within this many calendar days
EARNINGS_WARNING_DAYS = 2

# VIX thresholds for stop adjustment
VIX_HIGH = 25.0  # above this, widen stops

# Max positions per sector
MAX_SECTOR_POSITIONS = 2

# How many days of historical data to feed to Claude
LOOKBACK_DAYS = 30

# Claude model
CLAUDE_MODEL = "claude-sonnet-4-6"

# Market schedule (US Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 31  # Run 1 min after open
MARKET_TIMEZONE = "America/New_York"

# S&P 500 + S&P 400 constituents (large + mid cap, sourced from Wikipedia 2026-06-16)
# plus broad-market ETFs retained for market-regime and momentum signals.
# Edge check (scripts/universe_edge_check.py) validated positive edge on large+mid; small-caps
# (S&P 600) excluded from live trading (negative edge under the full pipeline).
# BRK.B / BF.B use dot notation; execution/universe.py normalises to hyphens for Alpaca.
STOCK_UNIVERSE = [
    "A",
    "AA",
    "AAL",
    "AAON",
    "AAPL",
    "ABBV",
    "ABNB",
    "ABT",
    "ACGL",
    "ACI",
    "ACM",
    "ACN",
    "ADBE",
    "ADC",
    "ADI",
    "ADM",
    "ADP",
    "ADSK",
    "AEE",
    "AEIS",
    "AEP",
    "AES",
    "AFG",
    "AFL",
    "AGCO",
    "AHR",
    "AIG",
    "AIT",
    "AIZ",
    "AJG",
    "AKAM",
    "ALB",
    "ALGM",
    "ALGN",
    "ALK",
    "ALL",
    "ALLE",
    "ALLY",
    "ALV",
    "AM",
    "AMAT",
    "AMCR",
    "AMD",
    "AME",
    "AMG",
    "AMGN",
    "AMH",
    "AMKR",
    "AMP",
    "AMT",
    "AMZN",
    "AN",
    "ANET",
    "ANF",
    "AON",
    "AOS",
    "APA",
    "APD",
    "APG",
    "APH",
    "APO",
    "APP",
    "APPF",
    "APTV",
    "AR",
    "ARE",
    "ARES",
    "ARMK",
    "ARW",
    "ARWR",
    "ASB",
    "ASH",
    "ATI",
    "ATO",
    "ATR",
    "AVAV",
    "AVB",
    "AVGO",
    "AVNT",
    "AVT",
    "AVTR",
    "AVY",
    "AWK",
    "AXON",
    "AXP",
    "AXTA",
    "AYI",
    "AZO",
    "BA",
    "BAC",
    "BAH",
    "BALL",
    "BAX",
    "BBWI",
    "BBY",
    "BC",
    "BCO",
    "BDC",
    "BDX",
    "BEN",
    "BF.B",
    "BG",
    "BHF",
    "BIIB",
    "BILL",
    "BIO",
    "BJ",
    "BKH",
    "BKNG",
    "BKR",
    "BLD",
    "BLDR",
    "BLK",
    "BLKB",
    "BMRN",
    "BMY",
    "BNY",
    "BR",
    "BRBR",
    "BRK.B",
    "BRKR",
    "BRO",
    "BROS",
    "BRX",
    "BSX",
    "BSY",
    "BURL",
    "BWA",
    "BWXT",
    "BX",
    "BXP",
    "BYD",
    "C",
    "CACI",
    "CAG",
    "CAH",
    "CAR",
    "CARR",
    "CART",
    "CASY",
    "CAT",
    "CAVA",
    "CB",
    "CBOE",
    "CBRE",
    "CBSH",
    "CBT",
    "CCI",
    "CCK",
    "CCL",
    "CDNS",
    "CDP",
    "CDW",
    "CEG",
    "CELH",
    "CF",
    "CFG",
    "CFR",
    "CG",
    "CGNX",
    "CHD",
    "CHDN",
    "CHE",
    "CHH",
    "CHRD",
    "CHRW",
    "CHTR",
    "CHWY",
    "CI",
    "CIEN",
    "CINF",
    "CL",
    "CLF",
    "CLH",
    "CLX",
    "CMC",
    "CMCSA",
    "CME",
    "CMG",
    "CMI",
    "CMS",
    "CNC",
    "CNH",
    "CNM",
    "CNO",
    "CNP",
    "CNX",
    "CNXC",
    "COF",
    "COHR",
    "COIN",
    "COKE",
    "COLB",
    "COLM",
    "COO",
    "COP",
    "COR",
    "COST",
    "COTY",
    "CPAY",
    "CPB",
    "CPRI",
    "CPRT",
    "CPT",
    "CR",
    "CRBG",
    "CRH",
    "CRL",
    "CRM",
    "CROX",
    "CRS",
    "CRUS",
    "CRWD",
    "CSCO",
    "CSGP",
    "CSL",
    "CSX",
    "CTAS",
    "CTRE",
    "CTSH",
    "CTVA",
    "CUBE",
    "CUZ",
    "CVLT",
    "CVNA",
    "CVS",
    "CVX",
    "CW",
    "CXT",
    "CYTK",
    "D",
    "DAL",
    "DAR",
    "DASH",
    "DBX",
    "DCI",
    "DD",
    "DDOG",
    "DE",
    "DECK",
    "DELL",
    "DG",
    "DGX",
    "DHI",
    "DHR",
    "DINO",
    "DIS",
    "DKS",
    "DLB",
    "DLR",
    "DLTR",
    "DOC",
    "DOCN",
    "DOCS",
    "DOCU",
    "DOV",
    "DOW",
    "DPZ",
    "DRI",
    "DT",
    "DTE",
    "DTM",
    "DUK",
    "DUOL",
    "DVA",
    "DVN",
    "DXCM",
    "DY",
    "EA",
    "EBAY",
    "ECL",
    "ED",
    "EEFT",
    "EFX",
    "EG",
    "EGP",
    "EHC",
    "EIX",
    "EL",
    "ELAN",
    "ELF",
    "ELS",
    "ELV",
    "EME",
    "EMR",
    "ENS",
    "ENSG",
    "ENTG",
    "EOG",
    "EPAM",
    "EPR",
    "EQH",
    "EQIX",
    "EQR",
    "EQT",
    "ERIE",
    "ES",
    "ESAB",
    "ESNT",
    "ESS",
    "ETN",
    "ETR",
    "EVR",
    "EVRG",
    "EW",
    "EWBC",
    "EXC",
    "EXE",
    "EXEL",
    "EXLS",
    "EXP",
    "EXPD",
    "EXPE",
    "EXPO",
    "EXR",
    "F",
    "FAF",
    "FANG",
    "FAST",
    "FBIN",
    "FCFS",
    "FCN",
    "FCX",
    "FDS",
    "FDX",
    "FE",
    "FFIN",
    "FFIV",
    "FHI",
    "FHN",
    "FICO",
    "FIS",
    "FISV",
    "FITB",
    "FIVE",
    "FIX",
    "FLEX",
    "FLG",
    "FLR",
    "FLS",
    "FN",
    "FNB",
    "FND",
    "FNF",
    "FOUR",
    "FOX",
    "FOXA",
    "FR",
    "FRT",
    "FSLR",
    "FTI",
    "FTNT",
    "FTV",
    "G",
    "GAP",
    "GATX",
    "GBCI",
    "GD",
    "GDDY",
    "GE",
    "GEF",
    "GEHC",
    "GEN",
    "GEV",
    "GGG",
    "GHC",
    "GILD",
    "GIS",
    "GL",
    "GLPI",
    "GLW",
    "GM",
    "GME",
    "GMED",
    "GNRC",
    "GNTX",
    "GOOG",
    "GOOGL",
    "GPC",
    "GPK",
    "GPN",
    "GRMN",
    "GS",
    "GT",
    "GTLS",
    "GWRE",
    "GWW",
    "GXO",
    "H",
    "HAE",
    "HAL",
    "HALO",
    "HAS",
    "HBAN",
    "HCA",
    "HD",
    "HGV",
    "HIG",
    "HII",
    "HIMS",
    "HL",
    "HLI",
    "HLNE",
    "HLT",
    "HOG",
    "HOMB",
    "HON",
    "HOOD",
    "HPE",
    "HPQ",
    "HQY",
    "HR",
    "HRB",
    "HRL",
    "HSIC",
    "HST",
    "HSY",
    "HUBB",
    "HUM",
    "HWC",
    "HWM",
    "HXL",
    "IBKR",
    "IBM",
    "IBOC",
    "ICE",
    "IDA",
    "IDCC",
    "IDXX",
    "IEX",
    "IFF",
    "ILMN",
    "INCY",
    "INGR",
    "INTC",
    "INTU",
    "INVH",
    "IP",
    "IPGP",
    "IQV",
    "IR",
    "IRM",
    "IRT",
    "ISRG",
    "IT",
    "ITT",
    "ITW",
    "IVZ",
    "J",
    "JAZZ",
    "JBHT",
    "JBL",
    "JCI",
    "JEF",
    "JHG",
    "JKHY",
    "JLL",
    "JNJ",
    "JPM",
    "KBH",
    "KBR",
    "KD",
    "KDP",
    "KEX",
    "KEY",
    "KEYS",
    "KHC",
    "KIM",
    "KKR",
    "KLAC",
    "KMB",
    "KMI",
    "KNF",
    "KNSL",
    "KNX",
    "KO",
    "KR",
    "KRC",
    "KRG",
    "KTOS",
    "KVUE",
    "L",
    "LAD",
    "LAMR",
    "LDOS",
    "LEA",
    "LECO",
    "LEN",
    "LFUS",
    "LH",
    "LHX",
    "LII",
    "LIN",
    "LITE",
    "LIVN",
    "LLY",
    "LMT",
    "LNT",
    "LNTH",
    "LOPE",
    "LOW",
    "LPX",
    "LRCX",
    "LSCC",
    "LSTR",
    "LULU",
    "LUV",
    "LVS",
    "LYB",
    "LYV",
    "M",
    "MA",
    "MAA",
    "MANH",
    "MAR",
    "MAS",
    "MAT",
    "MCD",
    "MCHP",
    "MCK",
    "MCO",
    "MDLZ",
    "MDT",
    "MEDP",
    "MET",
    "META",
    "MGM",
    "MIDD",
    "MKC",
    "MKSI",
    "MLI",
    "MLM",
    "MMM",
    "MMS",
    "MNST",
    "MO",
    "MOG.A",
    "MORN",
    "MOS",
    "MP",
    "MPC",
    "MPWR",
    "MRK",
    "MRNA",
    "MS",
    "MSA",
    "MSCI",
    "MSFT",
    "MSI",
    "MSM",
    "MTB",
    "MTD",
    "MTDR",
    "MTG",
    "MTN",
    "MTSI",
    "MTZ",
    "MU",
    "MUR",
    "MUSA",
    "MZTI",
    "NBIX",
    "NCLH",
    "NDAQ",
    "NDSN",
    "NEE",
    "NEM",
    "NEU",
    "NFG",
    "NFLX",
    "NI",
    "NJR",
    "NKE",
    "NLY",
    "NNN",
    "NOC",
    "NOV",
    "NOVT",
    "NOW",
    "NRG",
    "NSA",
    "NSC",
    "NTAP",
    "NTNX",
    "NTRS",
    "NUE",
    "NVDA",
    "NVR",
    "NVST",
    "NVT",
    "NWE",
    "NWS",
    "NWSA",
    "NXPI",
    "NXST",
    "NXT",
    "NYT",
    "O",
    "OC",
    "ODFL",
    "OGE",
    "OGS",
    "OHI",
    "OKE",
    "OKTA",
    "OLED",
    "OLLI",
    "OLN",
    "OMC",
    "ON",
    "ONB",
    "ONTO",
    "OPCH",
    "ORA",
    "ORCL",
    "ORI",
    "ORLY",
    "OSK",
    "OTIS",
    "OVV",
    "OXY",
    "OZK",
    "P",
    "PAG",
    "PANW",
    "PATH",
    "PAYX",
    "PB",
    "PBF",
    "PCAR",
    "PCG",
    "PCTY",
    "PEG",
    "PEGA",
    "PEN",
    "PEP",
    "PFE",
    "PFG",
    "PFGC",
    "PG",
    "PGR",
    "PH",
    "PHM",
    "PII",
    "PINS",
    "PK",
    "PKG",
    "PLD",
    "PLNT",
    "PLTR",
    "PM",
    "PNC",
    "PNFP",
    "PNR",
    "PNW",
    "PODD",
    "POOL",
    "POR",
    "POST",
    "PPC",
    "PPG",
    "PPL",
    "PR",
    "PRI",
    "PRU",
    "PSA",
    "PSKY",
    "PSN",
    "PSX",
    "PTC",
    "PVH",
    "PWR",
    "PYPL",
    "QCOM",
    "QLYS",
    "R",
    "RBA",
    "RBC",
    "RCL",
    "REG",
    "REGN",
    "REXR",
    "RF",
    "RGA",
    "RGEN",
    "RGLD",
    "RH",
    "RJF",
    "RL",
    "RLI",
    "RMBS",
    "RMD",
    "RNR",
    "ROIV",
    "ROK",
    "ROL",
    "ROP",
    "ROST",
    "RPM",
    "RRC",
    "RRX",
    "RS",
    "RSG",
    "RTX",
    "RVTY",
    "RYAN",
    "RYN",
    "SAIA",
    "SAIC",
    "SAM",
    "SARO",
    "SATS",
    "SBAC",
    "SBRA",
    "SBUX",
    "SCHW",
    "SCI",
    "SEIC",
    "SF",
    "SFM",
    "SGI",
    "SHC",
    "SHW",
    "SIGI",
    "SIRI",
    "SITM",
    "SJM",
    "SLAB",
    "SLB",
    "SLGN",
    "SLM",
    "SMCI",
    "SMG",
    "SN",
    "SNA",
    "SNDK",
    "SNPS",
    "SNX",
    "SO",
    "SOLS",
    "SOLV",
    "SON",
    "SPG",
    "SPGI",
    "SPXC",
    "SR",
    "SRE",
    "SSB",
    "SSD",
    "ST",
    "STAG",
    "STE",
    "STLD",
    "STRL",
    "STT",
    "STWD",
    "STX",
    "STZ",
    "SW",
    "SWK",
    "SWKS",
    "SWX",
    "SYF",
    "SYK",
    "SYNA",
    "SYY",
    "T",
    "TAP",
    "TCBI",
    "TDG",
    "TDY",
    "TECH",
    "TEL",
    "TER",
    "TEX",
    "TFC",
    "TGT",
    "THC",
    "THG",
    "THO",
    "TJX",
    "TKO",
    "TKR",
    "TLN",
    "TMHC",
    "TMO",
    "TMUS",
    "TNL",
    "TOL",
    "TPL",
    "TPR",
    "TREX",
    "TRGP",
    "TRMB",
    "TROW",
    "TRU",
    "TRV",
    "TSCO",
    "TSLA",
    "TSN",
    "TT",
    "TTC",
    "TTD",
    "TTEK",
    "TTMI",
    "TTWO",
    "TWLO",
    "TXN",
    "TXNM",
    "TXRH",
    "TXT",
    "TYL",
    "UAL",
    "UBER",
    "UBSI",
    "UDR",
    "UFPI",
    "UGI",
    "UHS",
    "ULS",
    "ULTA",
    "UMBF",
    "UNH",
    "UNM",
    "UNP",
    "UPS",
    "URI",
    "USB",
    "USFD",
    "UTHR",
    "V",
    "VAL",
    "VC",
    "VEEV",
    "VFC",
    "VICI",
    "VICR",
    "VLO",
    "VLTO",
    "VLY",
    "VMC",
    "VMI",
    "VNO",
    "VNOM",
    "VNT",
    "VOYA",
    "VRSK",
    "VRSN",
    "VRT",
    "VRTX",
    "VST",
    "VTR",
    "VTRS",
    "VVV",
    "VZ",
    "WAB",
    "WAL",
    "WAT",
    "WBD",
    "WBS",
    "WCC",
    "WDAY",
    "WDC",
    "WEC",
    "WELL",
    "WEX",
    "WFC",
    "WFRD",
    "WH",
    "WHR",
    "WING",
    "WLK",
    "WM",
    "WMB",
    "WMG",
    "WMS",
    "WMT",
    "WPC",
    "WRB",
    "WSM",
    "WSO",
    "WST",
    "WTFC",
    "WTRG",
    "WTS",
    "WTW",
    "WWD",
    "WY",
    "WYNN",
    "XEL",
    "XOM",
    "XPO",
    "XRAY",
    "XYL",
    "XYZ",
    "YETI",
    "YUM",
    "ZBH",
    "ZBRA",
    "ZION",
    "ZTS",
    # Broad-market & sector ETFs (retained for momentum / regime signals)
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLE",
    "XLF",
]

# ETFs that have no individual earnings — skip earnings lookups for these symbols.
# Crypto ETFs (BITO, GBTC) and international ETFs (KWEB, EEM) may appear as top movers;
# keeping this set broad avoids false "symbol may be delisted" errors from yfinance.
ETF_SYMBOLS: frozenset[str] = frozenset(
    {
        "SPY",
        "QQQ",
        "IWM",
        "XLK",
        "XLE",
        "XLF",
        "XLV",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "DIA",
        "VTI",
        "VTV",
        "VUG",
        "VOO",
        "VXX",
        "GLD",
        "SLV",
        "GDX",
        "USO",
        "TLT",
        "HYG",
        "LQD",
        "BITO",
        "GBTC",
        "IBIT",
        "FBTC",
        "KWEB",
        "EEM",
        "EFA",
        "FXI",
        "MCHI",
        "SQQQ",
        "TQQQ",
        "SPXU",
        "SPXL",
        "UVXY",
    }
)

# Email notifications
EMAIL_FROM = os.getenv("EMAIL_FROM")  # Your Gmail address
EMAIL_TO = os.getenv("EMAIL_TO")  # Owner address — emergency alerts only
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")  # Gmail App Password (not your login password)
# Named recipient list for daily summary + weekly review emails.
# Format: "FirstName:email,FirstName:email,..."
# Example: "Sam:sam@gmail.com,Harri:harri@outlook.com,Jess:jess@gmail.com"
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "")
# Legacy fallback — used if EMAIL_RECIPIENTS is not set
EMAIL_CC = os.getenv("EMAIL_CC", "")

# Log file path
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Pre-trade controls (MiFID II Article 17)
# Fat-finger guard: reject any single order above this USD value
MAX_SINGLE_ORDER_USD = float(os.getenv("MAX_SINGLE_ORDER_USD", "55.0" if _SAM else "50000.0"))
# Runaway algorithm guard: halt new buys once this much notional has been deployed today
MAX_DAILY_NOTIONAL_USD = float(os.getenv("MAX_DAILY_NOTIONAL_USD", "75.0" if _SAM else "150000.0"))
# Maximum total open position exposure at any one time (0 = disabled)
MAX_DEPLOYED_USD = float(os.getenv("MAX_DEPLOYED_USD", "125.0" if _SAM else "0.0"))
# Maximum intraday loss in USD before all positions are closed (0 = use % only)
MAX_DAILY_LOSS_USD = float(os.getenv("MAX_DAILY_LOSS_USD", "20.0" if _SAM else "0.0"))
# Maximum experiment drawdown from starting capital before halt (0 = disabled)
MAX_EXPERIMENT_DRAWDOWN_USD = float(
    os.getenv("MAX_EXPERIMENT_DRAWDOWN_USD", "50.0" if _SAM else "0.0")
)

# Account type assertions — verified against broker at startup in live mode
LONG_ONLY = os.getenv("LONG_ONLY", "true").lower() == "true"
ALLOW_MARGIN = os.getenv("ALLOW_MARGIN", "false").lower() == "true"

# Universe price filter (0 = disabled). In small account mode, restricts to names
# where a single whole share can be stop-protected within the per-order cap.
MIN_PRICE_USD = float(os.getenv("MIN_PRICE_USD", "5.0" if _SAM else "0.0"))
MAX_PRICE_USD = float(os.getenv("MAX_PRICE_USD", "60.0" if _SAM else "0.0"))

# Operations
# Kill switch creates this file; bot refuses to run while it exists.
# To resume: python main.py --clear-halt
HALT_FILE = os.path.join(LOG_DIR, ".HALTED")

# Max new buy orders placed in a single run — guards against runaway loops
MAX_ORDERS_PER_RUN = int(os.getenv("MAX_ORDERS_PER_RUN", "1" if _SAM else "3"))

# Minimum average daily volume — filters out illiquid stocks
MIN_VOLUME = 500_000

# Set to "I-ACCEPT-REAL-MONEY-RISK" to enable live trading without interactive prompt.
# Must be set in .env (or exported) before any live run — the scheduler does not set this automatically.
LIVE_CONFIRM = os.getenv("LIVE_CONFIRM", "")

# Path for runtime parameter overrides set by the weekly self-review.
# Values here take precedence over the defaults above at startup.
_RUNTIME_CONFIG_PATH = os.path.join(LOG_DIR, "runtime_config.json")

# Explicit allowlist of keys the weekly self-review may modify at runtime.
# Everything else — API keys, trading mode, stock universe — is immutable.
RUNTIME_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "MIN_CONFIDENCE",
        "TRAILING_STOP_PCT",
        "MAX_HOLD_DAYS",
        "MAX_ORDERS_PER_RUN",
        "PARTIAL_PROFIT_PCT",
    }
)

# (type, min_inclusive, max_inclusive) — applied only to runtime overrides.
# Tighter than the static validate() bounds to constrain AI self-modification.
RUNTIME_OVERRIDE_BOUNDS: dict[str, tuple] = {
    "MIN_CONFIDENCE": (int, 7, 10),
    "TRAILING_STOP_PCT": (float, 2.0, 10.0),
    "MAX_HOLD_DAYS": (int, 1, 10),
    "MAX_ORDERS_PER_RUN": (int, 1, 5),
    "PARTIAL_PROFIT_PCT": (float, 3.0, 20.0),
}


def _load_runtime_overrides() -> None:
    """Apply allowlisted, bounds-checked parameter overrides from runtime_config.json.

    Unknown keys are rejected. Values outside the declared type or bounds are
    rejected. Every decision is written to the audit log for full observability.
    Audit failures never prevent startup.
    """
    try:
        with open(_RUNTIME_CONFIG_PATH) as f:
            overrides = json.load(f)
    except FileNotFoundError:
        return
    except json.JSONDecodeError as exc:
        import logging as _logging

        _logging.getLogger(__name__).warning(f"runtime_config.json is malformed: {exc}")
        return

    import sys

    module = sys.modules[__name__]

    for key, raw_value in overrides.items():
        if key not in RUNTIME_OVERRIDE_KEYS:
            with contextlib.suppress(Exception):
                _audit_config_event(
                    "CONFIG_OVERRIDE_REJECTED",
                    {"key": key, "value": raw_value, "reason": "not in allowlist"},
                )
            continue

        expected_type, min_val, max_val = RUNTIME_OVERRIDE_BOUNDS[key]

        try:
            coerced = expected_type(raw_value)
        except (TypeError, ValueError):
            with contextlib.suppress(Exception):
                _audit_config_event(
                    "CONFIG_OVERRIDE_REJECTED",
                    {
                        "key": key,
                        "value": raw_value,
                        "reason": f"cannot coerce to {expected_type.__name__}",
                    },
                )
            continue

        if not (min_val <= coerced <= max_val):
            with contextlib.suppress(Exception):
                _audit_config_event(
                    "CONFIG_OVERRIDE_REJECTED",
                    {
                        "key": key,
                        "value": coerced,
                        "reason": f"out of bounds [{min_val}, {max_val}]",
                    },
                )
            continue

        setattr(module, key, coerced)
        with contextlib.suppress(Exception):
            _audit_config_event("CONFIG_OVERRIDE_APPLIED", {"key": key, "value": coerced})


def _audit_config_event(event_type: str, payload: dict) -> None:
    """Lazy audit emit — avoids circular import (audit_log imports config)."""
    try:
        from utils import audit_log

        audit_log._write(event_type, payload)
    except Exception:
        pass


_load_runtime_overrides()


def validate():
    """Raise ValueError if any config value is outside its safe operating range."""
    errors = []
    if not (0 < MAX_POSITION_PCT <= 1.0):
        errors.append(f"MAX_POSITION_PCT={MAX_POSITION_PCT} must be between 0 and 1")
    if not (0 < CASH_RESERVE_PCT < 1.0):
        errors.append(f"CASH_RESERVE_PCT={CASH_RESERVE_PCT} must be between 0 and 1")
    if not (1 <= MIN_CONFIDENCE <= 10):
        errors.append(f"MIN_CONFIDENCE={MIN_CONFIDENCE} must be between 1 and 10")
    if TRAILING_STOP_PCT <= 0:
        errors.append(f"TRAILING_STOP_PCT={TRAILING_STOP_PCT} must be positive")
    if MAX_HOLD_DAYS < 1:
        errors.append(f"MAX_HOLD_DAYS={MAX_HOLD_DAYS} must be >= 1")
    if MAX_POSITIONS < 1:
        errors.append(f"MAX_POSITIONS={MAX_POSITIONS} must be >= 1")
    if MAX_SINGLE_ORDER_USD <= 0:
        errors.append("MAX_SINGLE_ORDER_USD must be positive")
    if MAX_DAILY_NOTIONAL_USD <= 0:
        errors.append("MAX_DAILY_NOTIONAL_USD must be positive")
    if MAX_DEPLOYED_USD < 0:
        errors.append("MAX_DEPLOYED_USD must be >= 0")
    if MAX_DAILY_LOSS_USD < 0:
        errors.append("MAX_DAILY_LOSS_USD must be >= 0")
    if MAX_EXPERIMENT_DRAWDOWN_USD < 0:
        errors.append("MAX_EXPERIMENT_DRAWDOWN_USD must be >= 0")
    if SMALL_ACCOUNT_MODE and MAX_SINGLE_ORDER_USD > MAX_DAILY_NOTIONAL_USD:
        errors.append(
            f"MAX_SINGLE_ORDER_USD ({MAX_SINGLE_ORDER_USD}) > MAX_DAILY_NOTIONAL_USD "
            f"({MAX_DAILY_NOTIONAL_USD}) in SMALL_ACCOUNT_MODE"
        )
    # Short-side sizing ratios (F1) — guard against a mis-set env var sizing an oversized short.
    if not (0 < INDEX_HEDGE_WEIGHT <= 0.5):
        errors.append(f"INDEX_HEDGE_WEIGHT={INDEX_HEDGE_WEIGHT} must be in (0, 0.5]")
    if not (0 < SHORT_SIZE_SCALE <= 1.0):
        errors.append(f"SHORT_SIZE_SCALE={SHORT_SIZE_SCALE} must be in (0, 1]")
    if not (0 < MAX_SHORT_STANDALONE_RATIO <= 1.0):
        errors.append(f"MAX_SHORT_STANDALONE_RATIO={MAX_SHORT_STANDALONE_RATIO} must be in (0, 1]")
    if not (0 < MAX_SHORT_HEDGE_RATIO <= 1.0):
        errors.append(f"MAX_SHORT_HEDGE_RATIO={MAX_SHORT_HEDGE_RATIO} must be in (0, 1]")
    if errors:
        raise ValueError("Config errors:\n" + "\n".join(f"  - {e}" for e in errors))

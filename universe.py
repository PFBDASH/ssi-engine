# universe.py

# ---------------------------------------------------
# SSI DEFAULT UNIVERSES
# ---------------------------------------------------
# These are the fast, always-on default scan universes.
# Users can later manually score ANY symbol on-demand (handled in app.py).
# ---------------------------------------------------

# Lane 1 — CRYPTO (spot pairs)
CRYPTO = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "ADA-USD",
]

# Lane 2 — FOREX (majors)
FOREX = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCHF=X",
]

# ---------------------------------------------------
# Lane 3 — OPTIONS UNDERLYINGS (keep SMALL + ultra-liquid)
# ---------------------------------------------------
# Purpose: only underlyings with consistently tight spreads + deep OI.
OPTIONS_UNDERLYINGS = [
    # Index proxies
    "SPY", "QQQ", "IWM", "DIA",

    # Vol / hedges
    "TLT", "GLD",

    # Mega-cap liquidity magnets
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",

    # High-volume “retail magnets”
    "AMD", "NFLX",

    # Financial bellwethers
    "JPM",

    # Energy bellwether
    "XOM",
]

# ---------------------------------------------------
# Lane 4 — LC (Long-Cycle / Phase-4)
# US EQUITIES ONLY — broad discovery universe (runnable today)
# ---------------------------------------------------
# Goal: broad enough to find real Phase-4 candidates,
# but small enough to run reliably on your current stack.
#
# This list is intentionally diversified across sectors.
# You can expand later once you add caching / better data sources.
LC_UNIVERSE = [
    # Benchmarks / context
    "SPY", "QQQ", "IWM", "DIA",

    # Mega-cap / Platforms
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",

    # Semis / AI infrastructure
    "AMD", "AVGO", "QCOM", "MU", "AMAT", "LRCX", "ASML", "INTC", "ON", "TXN",

    # Cyber / software (often base + re-rate)
    "CRWD", "NET", "DDOG", "SNOW", "MDB", "NOW", "ADBE", "ORCL", "PANW", "ZS", "OKTA",

    # Cloud / comms / networks
    "CSCO", "ANET", "VZ", "T", "TMUS",

    # Consumer / retail / travel
    "COST", "WMT", "HD", "LOW", "NKE", "SBUX", "MCD", "DIS", "BKNG", "ABNB", "UBER",

    # Financials / capital markets
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA",

    # Healthcare / biotech / medtech
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "DHR", "ISRG", "VRTX", "REGN",

    # Industrials / defense / automation
    "CAT", "DE", "HON", "GE", "LMT", "NOC", "GD", "BA", "RTX", "ETN",

    # Energy (long-cycle bases happen here)
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY",

    # Materials / miners (classic cycle bases)
    "LIN", "APD", "FCX", "NUE", "STLD", "AA", "MP",

    # Utilities / infrastructure (slow but important regime)
    "NEE", "DUK", "SO",

    # REITs / housing signal
    "AMT", "PLD", "O",

    # ETFs for sector regime context
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLC", "XLU", "XLB",

    # “Long-cycle candidate” high beta cluster
    "PLTR", "SHOP", "SQ", "PYPL", "ROKU", "TWLO", "COIN", "HOOD", "SOFI",
    "RIVN", "LCID", "RKLB",

    # Quality compounders
    "PG", "KO", "PEP", "PM", "MO", "CL", "MDLZ", "TGT",

    # Transport / logistics / rails (cycle tells)
    "UPS", "FDX", "UNP", "CSX", "NSC",

    # Chips + hardware supply chain
    "DELL", "HPQ", "IBM",
]

# Benchmark used inside LC relative strength checks
LC_BENCHMARK = "SPY"
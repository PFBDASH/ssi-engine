# universe.py

# Keep symbols simple + compatible with yfinance.
# Crypto uses the "-USD" format.
# FX uses "=X" format.
# Options lane uses US equity/ETF underlyings.

CRYPTO = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "ADA-USD",
]

FOREX = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCHF=X",
]

OPTIONS_UNDERLYINGS = [
    "SPY",
    "QQQ",
    "IWM",
    "NVDA",
    "TSLA",
]
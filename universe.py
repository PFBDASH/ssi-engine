# universe.py

CRYPTO_SYMBOLS = [
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "XRPUSD",
    "ADAUSD",
]

# Stooq FX daily symbols (no suffix)
FOREX_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
]

# Underlyings to score for “Options Lane”
# (we’re scoring the underlying, not pulling option chains)
OPTIONS_UNDERLYINGS = [
    "SPY",
    "QQQ",
    "IWM",
    "NVDA",
    "TSLA",
]
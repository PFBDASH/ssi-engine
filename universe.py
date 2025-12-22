# universe.py

CRYPTO = [
    ("BTCUSD", "BTC-USD"),
    ("ETHUSD", "ETH-USD"),
    ("SOLUSD", "SOL-USD"),
    ("XRPUSD", "XRP-USD"),
    ("ADAUSD", "ADA-USD"),
]

FOREX = [
    ("EURUSD", "EURUSD=X"),
    ("GBPUSD", "GBPUSD=X"),
    ("USDJPY", "JPY=X"),      # USDJPY via JPY=X (USD per JPY). We'll invert internally for display.
    ("AUDUSD", "AUDUSD=X"),
    ("USDCHF", "CHF=X"),      # USDCHF via CHF=X (USD per CHF). We'll invert internally for display.
]

# Options underlyings (liquid, US)
OPTIONS_UNDERLYINGS = [
    "SPY",
    "QQQ",
    "IWM",
    "NVDA",
    "TSLA",
]
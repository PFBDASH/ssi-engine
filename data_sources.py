# data_sources.py
import requests
import pandas as pd
from datetime import datetime, timedelta

# NOTE: This uses free Kraken OHLC for crypto
def fetch_crypto_ohlc(symbol, interval=60):
    pair = symbol.replace("USD", "/USD")
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    r = requests.get(url).json()
    data = list(r["result"].values())[0]
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","vwap","volume","count"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
    return df
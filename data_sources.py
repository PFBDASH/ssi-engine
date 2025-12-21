# data_sources.py
import requests
import pandas as pd

# ---------- CRYPTO (Kraken OHLC) ----------
def fetch_crypto_ohlc(symbol, interval=60):
    pair = symbol.replace("USD", "/USD")
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    r = requests.get(url, timeout=20).json()
    data = list(r["result"].values())[0]
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","vwap","volume","count"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
    return df

# ---------- FOREX (FREE via Stooq daily) ----------
# Stooq uses symbols like:
# EURUSD -> "EURUSD"
# USDJPY -> "USDJPY"
# Endpoint returns daily OHLC.
def fetch_fx_daily(pair):
    sym = pair.upper()
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    # Stooq returns: Date,Open,High,Low,Close,Volume
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))

    # Normalize columns to our engine format
    df.rename(columns={
        "Date": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close"
    }, inplace=True)

    df["time"] = pd.to_datetime(df["time"])
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)

    # Keep last ~300 bars (enough for EMA200 + ATR)
    return df.tail(300).reset_index(drop=True)
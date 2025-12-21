# data_sources.py
import requests
import pandas as pd
from io import StringIO


# ---------- CRYPTO (Kraken OHLC) ----------
def fetch_crypto_ohlc(symbol: str, interval: int = 60) -> pd.DataFrame:
    """
    symbol examples: BTCUSD, ETHUSD, SOLUSD
    Uses Kraken public OHLC endpoint.
    """
    pair = symbol.replace("USD", "/USD")
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    r = requests.get(url, timeout=20).json()

    # Kraken returns {"error":[], "result":{ "<pair>":[...], "last":...}}
    result = r.get("result", {})
    # Find the first list in result values (ignore "last")
    ohlc_key = None
    for k, v in result.items():
        if isinstance(v, list):
            ohlc_key = k
            break
    if ohlc_key is None:
        raise ValueError(f"Kraken OHLC missing for {symbol}: {r}")

    data = result[ohlc_key]
    df = pd.DataFrame(
        data,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df.reset_index(drop=True)


# ---------- FOREX (FREE via Stooq daily) ----------
def fetch_fx_daily(pair: str) -> pd.DataFrame:
    """
    pair examples: EURUSD, USDJPY
    Stooq daily CSV endpoint.
    """
    sym = pair.upper()
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        raise ValueError(f"Empty FX data for {pair}")

    df.rename(
        columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close"},
        inplace=True,
    )
    df["time"] = pd.to_datetime(df["time"])
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df.tail(350).reset_index(drop=True)


# ---------- EQUITIES (FREE via Stooq daily) ----------
def fetch_equity_daily(ticker: str) -> pd.DataFrame:
    """
    ticker examples: SPY, NVDA
    Stooq US equities generally use: <ticker>.us (lowercase).
    """
    sym = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        raise ValueError(f"Empty equity data for {ticker}")

    df.rename(
        columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close"},
        inplace=True,
    )
    df["time"] = pd.to_datetime(df["time"])
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df.tail(350).reset_index(drop=True)
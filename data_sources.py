# data_sources.py
import requests
import pandas as pd
from io import StringIO

# =========================================================
# Shared HTTP config (performance + reliability)
# =========================================================
_SESSION = requests.Session()
_HEADERS = {"User-Agent": "SSI/1.0 (+Render)"}
# Faster failure for flaky upstreams; prevents long multi-ticker stalls
_TIMEOUT = (3.05, 8)  # (connect_timeout, read_timeout)


# ---------- CRYPTO (Kraken OHLC) ----------
def fetch_crypto_ohlc(symbol: str, interval: int = 60) -> pd.DataFrame:
    """
    symbol examples: BTCUSD, ETHUSD, SOLUSD
    Uses Kraken public OHLC endpoint.
    Returns dataframe with: time/open/high/low/close (+ volume/vwap/count).
    """
    pair = symbol.replace("USD", "/USD")
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"

    try:
        resp = _SESSION.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return pd.DataFrame()

    # Kraken returns {"error":[], "result":{ "<pair>":[...], "last":...}}
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    ohlc_key = None
    for k, v in result.items():
        if isinstance(v, list):
            ohlc_key = k
            break
    if ohlc_key is None:
        return pd.DataFrame()

    data = result.get(ohlc_key, [])
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(
        data,
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
    )

    # Robust parsing
    df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df = df.dropna(subset=["time"])
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])

    return df.reset_index(drop=True)


# ---------- FOREX (FREE via Stooq daily) ----------
def fetch_fx_daily(pair: str) -> pd.DataFrame:
    """
    pair examples: EURUSD, USDJPY
    Stooq daily CSV endpoint.
    Returns dataframe with: time/open/high/low/close
    """
    sym = pair.upper()
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"

    try:
        r = _SESSION.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    # Stooq sometimes returns empty/odd payloads; fail soft
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df.rename(
        columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close"},
        inplace=True,
    )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    # Numeric coercion (never raise)
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])

    return df.tail(350).reset_index(drop=True)


# ---------- EQUITIES (FREE via Stooq daily) ----------
def fetch_equity_daily(ticker: str) -> pd.DataFrame:
    """
    ticker examples: SPY, NVDA
    Stooq US equities generally use: <ticker>.us (lowercase).
    Returns dataframe with: time/open/high/low/close
    """
    sym = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"

    try:
        r = _SESSION.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df.rename(
        columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close"},
        inplace=True,
    )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])

    return df.tail(350).reset_index(drop=True)
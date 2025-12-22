import pandas as pd
import yfinance as yf

# -----------------------
# Helpers
# -----------------------

def _fetch_daily(symbol, days=180):
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)

    if df is None or df.empty:
        return None

    # Normalize column names safely
    df.columns = [str(c).title() for c in df.columns]
    df = df.dropna()

    return df


def _get_close_series(df):
    """
    Robustly resolve a usable close price series.
    """
    if "Close" in df.columns:
        return df["Close"]
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    return None


def _score_symbol(label, symbol):
    df = _fetch_daily(symbol)

    if df is None or len(df) < 60:
        return None

    close = _get_close_series(df)
    if close is None:
        return None

    ma50 = close.rolling(50).mean()
    vol20 = close.pct_change().rolling(20).std()

    trend = 1 if close.iloc[-1] > ma50.iloc[-1] else 0
    vol = 1 if vol20.iloc[-1] > 0.02 else 0

    score = round((trend * 2.5) + (vol * 2.5), 2)

    return {
        "symbol": label,
        "trend": trend * 5,
        "vol": vol * 5,
        "score": score
    }


# -----------------------
# Public scans
# -----------------------

def run_crypto_scan():
    symbols = {
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "SOLUSD": "SOL-USD",
        "XRPUSD": "XRP-USD",
        "ADAUSD": "ADA-USD",
    }

    rows = []
    for label, sym in symbols.items():
        r = _score_symbol(label, sym)
        if r:
            rows.append(r)

    return pd.DataFrame(rows).sort_values("score", ascending=False)


def run_fx_scan():
    symbols = {
        "USDJPY": "USDJPY=X",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
    }

    rows = []
    for label, sym in symbols.items():
        r = _score_symbol(label, sym)
        if r:
            rows.append(r)

    return pd.DataFrame(rows).sort_values("score", ascending=False)


def run_options_scan():
    symbols = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "IWM": "IWM",
        "NVDA": "NVDA",
        "TSLA": "TSLA",
    }

    rows = []
    for label, sym in symbols.items():
        r = _score_symbol(label, sym)
        if not r:
            continue

        if r["score"] >= 4.5:
            r["recommendation"] = "Iron Condor (30–45 DTE, ~15 delta)"
        elif r["score"] >= 3.5:
            r["recommendation"] = "Directional Call/Put (5–7 DTE)"
        else:
            r["recommendation"] = "No trade"

        rows.append(r)

    return pd.DataFrame(rows).sort_values("score", ascending=False)
import pandas as pd
import yfinance as yf

# -----------------------
# Helpers
# -----------------------

def _fetch_daily(symbol, days=180):
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
    if df is None or df.empty:
        return None

    # FORCE safe column names
    df.columns = [str(c).title() for c in df.columns]
    df = df.dropna()
    return df


def _score_symbol(label, symbol):
    df = _fetch_daily(symbol)
    if df is None or len(df) < 50:
        return None

    close = df["Close"]

    trend = 1 if close.iloc[-1] > close.rolling(50).mean().iloc[-1] else 0
    vol = 1 if close.pct_change().rolling(20).std().iloc[-1] > 0.02 else 0

    score = round((trend * 2.5) + (vol * 2.5), 2)

    return {
        "symbol": label,
        "trend": trend * 5,
        "vol": vol * 5,
        "score": score
    }


# -----------------------
# Public scan functions
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
    # Underlyings only (options logic is recommendation-based)
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
        if r:
            # add options recommendation
            if r["score"] >= 4.5:
                r["recommendation"] = "Iron Condor (30–45 DTE)"
            elif r["score"] >= 3.5:
                r["recommendation"] = "Directional Call / Put (5–7 DTE)"
            else:
                r["recommendation"] = "No trade"

            rows.append(r)

    return pd.DataFrame(rows).sort_values("score", ascending=False)
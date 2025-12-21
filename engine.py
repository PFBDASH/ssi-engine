# engine.py
from indicators import trend_score, vol_score
from data_sources import fetch_crypto_ohlc, fetch_fx_daily
from universe import CRYPTO_SYMBOLS, FOREX_SYMBOLS

def run_crypto_scan():
    results = []
    for sym in CRYPTO_SYMBOLS:
        try:
            df = fetch_crypto_ohlc(sym)
            t = trend_score(df)
            v = vol_score(df)
            score = round(0.7 * t + 0.3 * v, 1)
            results.append({"symbol": sym, "trend": t, "vol": v, "score": score})
        except Exception:
            continue
    return sorted(results, key=lambda x: x["score"], reverse=True)

def run_fx_scan():
    results = []
    for pair in FOREX_SYMBOLS:
        try:
            df = fetch_fx_daily(pair)
            t = trend_score(df)
            v = vol_score(df)
            score = round(0.7 * t + 0.3 * v, 1)
            results.append({"symbol": pair, "trend": t, "vol": v, "score": score})
        except Exception:
            continue
    return sorted(results, key=lambda x: x["score"], reverse=True)

def compute_ssi(crypto_results, fx_results=None):
    # SSI is primarily crypto risk-on/off for now.
    # Later we can blend FX regime in, but keep it simple for v0.2.
    if not crypto_results:
        return 0
    avg = sum(r["score"] for r in crypto_results) / len(crypto_results)
    return round(avg, 1)
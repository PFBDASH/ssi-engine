# engine.py
from indicators import trend_score, vol_score
from data_sources import fetch_crypto_ohlc
from universe import CRYPTO_SYMBOLS

def run_crypto_scan():
    results = []
    for sym in CRYPTO_SYMBOLS:
        try:
            df = fetch_crypto_ohlc(sym)
            t = trend_score(df)
            v = vol_score(df)
            score = round(0.7 * t + 0.3 * v, 1)
            results.append({
                "symbol": sym,
                "trend": t,
                "vol": v,
                "score": score
            })
        except Exception as e:
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results

def compute_ssi(crypto_results):
    if not crypto_results:
        return 0
    avg = sum(r["score"] for r in crypto_results) / len(crypto_results)
    return round(avg, 1)
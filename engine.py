# engine.py
from indicators import trend_score, vol_score
from data_sources import fetch_crypto_ohlc, fetch_fx_daily, fetch_equity_daily
from universe import CRYPTO_SYMBOLS, FOREX_SYMBOLS, OPTIONS_UNDERLYINGS


def _score_rows(symbols, fetch_fn):
    rows = []
    for sym in symbols:
        try:
            df = fetch_fn(sym)
            t = trend_score(df)
            v = vol_score(df)
            score = round(0.7 * t + 0.3 * v, 1)
            rows.append({"symbol": sym, "trend": t, "vol": v, "score": score})
        except Exception:
            # keep engine resilient; failures just skip
            continue
    return sorted(rows, key=lambda x: x["score"], reverse=True)


def run_crypto_scan():
    # Kraken uses intraday OHLC; scoring still works
    return _score_rows(CRYPTO_SYMBOLS, lambda s: fetch_crypto_ohlc(s, interval=60))


def run_fx_scan():
    return _score_rows(FOREX_SYMBOLS, fetch_fx_daily)


def run_options_scan():
    # “Options lane” is really “best underlyings to consider for options”
    return _score_rows(OPTIONS_UNDERLYINGS, fetch_equity_daily)


def compute_ssi(crypto_rows, fx_rows=None, opt_rows=None):
    """
    SSI = average of top crypto scores (simple + stable).
    Later we can blend FX and options into SSI, but this keeps it consistent.
    """
    if not crypto_rows:
        return 0.0
    top = crypto_rows[:5] if len(crypto_rows) >= 5 else crypto_rows
    return round(sum(r["score"] for r in top) / len(top), 1)
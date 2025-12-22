# engine.py

from indicators import trend_score, vol_score
from data_sources import (
    fetch_crypto_ohlc,
    fetch_fx_daily,
    fetch_equity_daily,
)
from universe import (
    CRYPTO_SYMBOLS,
    FOREX_SYMBOLS,
    OPTIONS_UNDERLYINGS,
)

# ---------------- CORE SCAN ----------------

def _scan(symbols, fetch_fn):
    rows = []
    for sym in symbols:
        try:
            df = fetch_fn(sym)
            t = trend_score(df)
            v = vol_score(df)
            score = round(0.7 * t + 0.3 * v, 1)
            last = float(df["close"].iloc[-1])

            rows.append({
                "symbol": sym,
                "last": round(last, 6),
                "trend": int(t),
                "vol": int(v),
                "score": score,
            })
        except Exception:
            continue

    return sorted(rows, key=lambda x: x["score"], reverse=True)


def run_crypto_scan():
    return _scan(CRYPTO_SYMBOLS, lambda s: fetch_crypto_ohlc(s, interval=60))


def run_fx_scan():
    return _scan(FOREX_SYMBOLS, fetch_fx_daily)


def run_options_scan():
    return _scan(OPTIONS_UNDERLYINGS, fetch_equity_daily)


def compute_ssi(crypto_rows):
    if not crypto_rows:
        return 0.0
    top = crypto_rows[:5]
    return round(sum(r["score"] for r in top) / len(top), 1)

# ---------------- PLAYBOOKS ----------------

def _bias(trend):
    if trend >= 7:
        return "BULLISH"
    if trend <= 3:
        return "BEARISH"
    return "NEUTRAL"


def crypto_playbook(row, ssi, min_score):
    if row["score"] < min_score:
        return "CRYPTO: NO TRADE — score below threshold"

    if ssi <= 3:
        return "CRYPTO: RISK OFF — stand down"
    if ssi >= 7 and row["trend"] >= 7:
        return "CRYPTO: MOMENTUM / LOTTO (small size)"
    return "CRYPTO: CONTROLLED DIRECTIONAL"


def forex_playbook(row, ssi, min_score):
    if row["score"] < min_score:
        return "FOREX: NO TRADE — score below threshold"

    if ssi <= 3:
        return "FOREX: SMALL / STAND DOWN"
    if ssi >= 7:
        return "FOREX: TREND FOLLOW (swing)"
    return "FOREX: RANGE / MEAN REVERSION"


def options_playbook(row, ssi, min_score):
    if row["score"] < min_score:
        return "OPTIONS: NO TRADE — score below threshold"

    if ssi <= 3:
        return "OPTIONS: STAND DOWN"
    if ssi >= 7 and row["trend"] >= 7:
        return "OPTIONS: LOTTO (0–1 DTE, tiny risk)"
    return "OPTIONS: DEFINED RISK (spreads / 3–7 DTE)"
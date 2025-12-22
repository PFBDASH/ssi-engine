# engine.py

from indicators import trend_score, vol_score
from data_sources import fetch_crypto_ohlc, fetch_fx_daily, fetch_equity_daily
from universe import CRYPTO_SYMBOLS, FOREX_SYMBOLS, OPTIONS_UNDERLYINGS
import pandas as pd


# ---------------- HIDDEN GATES ----------------
MIN_CRYPTO_SCORE = 6.5
MIN_FX_SCORE = 6.0
MIN_OPTIONS_SCORE = 6.5


# ---------------- CORE SCAN ----------------

def _scan(symbols, fetch_fn):
    rows = []
    for sym in symbols:
        try:
            df = fetch_fn(sym)
            t = int(trend_score(df))
            v = int(vol_score(df))
            score = round(0.7 * t + 0.3 * v, 1)
            last = float(df["close"].iloc[-1])

            tq = trend_quality(df)
            rr = range_regime(df)

            rows.append({
                "symbol": sym,
                "last": round(last, 6),
                "trend": t,
                "vol": v,
                "score": score,
                "trend_quality": tq,
                "range_regime": rr,
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


# ---------------- METRICS ----------------

def trend_quality(df):
    """
    Measures whether price is expanding away from EMA50 or compressing.
    """
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    dist = abs(df["close"].iloc[-1] - ema50.iloc[-1])
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]

    if atr == 0 or pd.isna(atr):
        return "Unknown"

    ratio = dist / atr
    if ratio > 1.5:
        return "Strong Trend"
    if ratio < 0.7:
        return "Compressed"
    return "Moderate"


def range_regime(df):
    """
    Detects expanding vs contracting ranges.
    """
    recent_range = (df["high"].rolling(10).max() - df["low"].rolling(10).min()).iloc[-1]
    prior_range = (df["high"].rolling(30).max() - df["low"].rolling(30).min()).iloc[-1]

    if prior_range == 0 or pd.isna(prior_range):
        return "Unknown"

    ratio = recent_range / prior_range
    if ratio > 1.2:
        return "Expanding"
    if ratio < 0.8:
        return "Contracting"
    return "Neutral"


# ---------------- HELPERS ----------------

def _direction(trend):
    if trend >= 7:
        return "LONG"
    if trend <= 3:
        return "SHORT"
    return "NO TRADE"


def _confidence(score, min_score, trend_quality, range_regime):
    buffer = score - min_score
    if buffer >= 1.5 and trend_quality == "Strong Trend" and range_regime != "Contracting":
        return "High"
    if buffer >= 0.5:
        return "Medium"
    return "Avoid"


# ---------------- RECOMMENDATIONS ----------------

def recommend_lane(best_row, ssi, lane):
    if not best_row:
        return "No data available."

    min_score = {
        "CRYPTO": MIN_CRYPTO_SCORE,
        "FOREX": MIN_FX_SCORE,
        "OPTIONS": MIN_OPTIONS_SCORE,
    }[lane]

    if best_row["score"] < min_score:
        return f"{lane}: No trade — no setup cleared quality threshold."

    direction = _direction(best_row["trend"])
    if direction == "NO TRADE":
        return f"{lane}: No trade — trend not decisive."

    confidence = _confidence(
        best_row["score"],
        min_score,
        best_row["trend_quality"],
        best_row["range_regime"],
    )

    hold = (
        "24–72h" if lane == "CRYPTO"
        else "1–2 weeks" if lane == "FOREX"
        else "3–7 days"
    )

    why = (
        f"{best_row['trend_quality']} + "
        f"{best_row['range_regime']} range + "
        f"score {best_row['score']}"
    )

    return (
        f"{lane}: {direction} {best_row['symbol']} — "
        f"hold {hold}. "
        f"Confidence: {confidence}. "
        f"Why: {why}."
    )
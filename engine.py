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
    def _round_strike(sym: str, strike: float) -> float:
    sym = sym.upper()
    if sym in ["SPY", "QQQ", "IWM"]:
        step = 1.0
    else:
        step = 5.0
    return round(strike / step) * step


def recommend_options_contract(best_row: dict, ssi: float) -> str:
    """
    Returns explicit options contract-style guidance:
    - IRON CONDOR with approximate strikes + expiry window
    - LOTTO CALL/PUT with approx strike + expiry window
    - DEFINED-RISK CALL/PUT with approx strike + expiry window
    Uses only underlying price + our internal metrics (no option chain).
    """
    if not best_row:
        return "OPTIONS: No data available."

    # pull what we already compute
    sym = best_row.get("symbol")
    px = float(best_row.get("last", 0))
    t = int(best_row.get("trend", 0))
    v = int(best_row.get("vol", 0))
    score = float(best_row.get("score", 0))
    tq = best_row.get("trend_quality", "Unknown")
    rr = best_row.get("range_regime", "Unknown")

    # gate
    if score < MIN_OPTIONS_SCORE:
        return "OPTIONS: No trade — no underlying cleared the threshold."

    # risk-off overlay
    risk_off = ssi <= 3

    # determine directional bias
    if t >= 7:
        bias = "CALL"
        dirn = "LONG"
    elif t <= 3:
        bias = "PUT"
        dirn = "SHORT"
    else:
        bias = "NEUTRAL"
        dirn = "NO TRADE"

    why = f"{tq} + {rr} range + score {score}"

    # 1) Neutral / chop => condor
    if dirn == "NO TRADE" or (4 <= t <= 6) or rr == "Contracting":
        # wings based on vol
        wing = 0.015 if v >= 6 else 0.012      # ~1.2–1.5%
        hedge = wing + 0.01                    # add ~1% for long wings

        short_put = _round_strike(sym, px * (1 - wing))
        long_put  = _round_strike(sym, px * (1 - hedge))
        short_call = _round_strike(sym, px * (1 + wing))
        long_call  = _round_strike(sym, px * (1 + hedge))

        return (
            f"OPTIONS: IRON CONDOR {sym} — expiry {CONDOR_DTE}. "
            f"Sell PUT spread: short ~{short_put}, long ~{long_put}. "
            f"Sell CALL spread: short ~{short_call}, long ~{long_call}. "
            f"Why: {why}."
        )

    # 2) Hot + volatile + directional => lotto (unless risk-off)
    if (v >= 7) and (not risk_off) and dirn in ("LONG", "SHORT"):
        otm = 0.02 if v >= 8 else 0.015
        strike = _round_strike(sym, px * (1 + otm)) if dirn == "LONG" else _round_strike(sym, px * (1 - otm))

        return (
            f"OPTIONS: LOTTO {bias} {sym} — expiry {LOTTO_DTE}. "
            f"Target strike ~{strike} (~{int(otm*100)}% OTM). "
            f"Plan: take profit fast on a spike; cut if momentum dies. "
            f"Why: {why}."
        )

    # 3) Otherwise defined-risk directional
    otm = 0.01
    strike = _round_strike(sym, px * (1 + otm)) if dirn == "LONG" else _round_strike(sym, px * (1 - otm))
    side = "CALL" if dirn == "LONG" else "PUT"

    return (
        f"OPTIONS: DEFINED-RISK {side} {sym} — expiry {DIRECTIONAL_DTE}. "
        f"Target strike ~{strike} (~1% OTM). Prefer spreads if premiums are expensive. "
        f"Why: {why}."
    )
# engine.py

from indicators import trend_score, vol_score
from data_sources import fetch_crypto_ohlc, fetch_fx_daily, fetch_equity_daily
from universe import CRYPTO_SYMBOLS, FOREX_SYMBOLS, OPTIONS_UNDERLYINGS


# ---------------- HIDDEN GATES (edit here only) ----------------
MIN_CRYPTO_SCORE = 6.5
MIN_FX_SCORE = 6.0
MIN_OPTIONS_SCORE = 6.5

LOTTO_DTE = "3–5 trading days"
DIRECTIONAL_DTE = "3–7 trading days"
CONDOR_DTE = "7–14 trading days"


# ---------------- CORE SCAN ----------------

def _scan(symbols, fetch_fn):
    rows = []
    for sym in symbols:
        try:
            df = fetch_fn(sym)
            t = int(trend_score(df))   # 0–10
            v = int(vol_score(df))     # 0–10
            score = round(0.7 * t + 0.3 * v, 1)
            last = float(df["close"].iloc[-1])

            rows.append({
                "symbol": sym,
                "last": round(last, 6),
                "trend": t,
                "vol": v,
                "score": float(score),
            })
        except Exception:
            continue

    return sorted(rows, key=lambda x: x["score"], reverse=True)


def run_crypto_scan():
    # Kraken hourly OHLC (intraday bias)
    return _scan(CRYPTO_SYMBOLS, lambda s: fetch_crypto_ohlc(s, interval=60))


def run_fx_scan():
    # Daily OHLC (swing bias)
    return _scan(FOREX_SYMBOLS, fetch_fx_daily)


def run_options_scan():
    # Scores underlyings only (SPY/QQQ/IWM/NVDA/etc)
    return _scan(OPTIONS_UNDERLYINGS, fetch_equity_daily)


def compute_ssi(crypto_rows):
    if not crypto_rows:
        return 0.0
    top = crypto_rows[:5]
    return round(sum(r["score"] for r in top) / len(top), 1)


# ---------------- RECOMMENDATION PRIMITIVES ----------------

def _direction(trend: int):
    if trend >= 7:
        return "LONG"
    if trend <= 3:
        return "SHORT"
    return "NO TRADE"


def _hold_window(lane: str, vol: int):
    """
    Lane-specific holding windows.
    vol high => shorter holds; vol low => longer holds.
    """
    if lane == "CRYPTO":
        if vol >= 8:
            return "intraday (1–8h)"
        if vol >= 5:
            return "24–72h"
        return "3–7 days"
    if lane == "FX":
        if vol >= 8:
            return "2–5 days"
        if vol >= 5:
            return "1–2 weeks"
        return "2–4 weeks"
    if lane == "OPTIONS":
        if vol >= 8:
            return "0–2 days"
        if vol >= 5:
            return "3–7 days"
        return "7–14 days"
    return "N/A"


def _pct_to_strike(price: float, pct: float):
    return round(price * (1 + pct), 2)


# ---------------- LANE RECOMMENDATIONS ----------------

def recommend_crypto(best_row, ssi: float):
    if not best_row or best_row["score"] < MIN_CRYPTO_SCORE:
        return "CRYPTO: No trade — no symbol cleared the threshold."

    d = _direction(best_row["trend"])
    if d == "NO TRADE":
        return "CRYPTO: No trade — trend not strong enough."

    hold = _hold_window("CRYPTO", best_row["vol"])

    # Global regime overlay via SSI
    if ssi <= 3:
        return "CRYPTO: Risk OFF — stand down."
    if 3 < ssi < 5 and d == "LONG":
        return f"CRYPTO: Caution regime — small size only. {d} {best_row['symbol']} — hold {hold}."

    return f"CRYPTO: {d} {best_row['symbol']} — look to exit within {hold}."


def recommend_fx(best_row, ssi: float):
    if not best_row or best_row["score"] < MIN_FX_SCORE:
        return "FOREX: No trade — no pair cleared the threshold."

    d = _direction(best_row["trend"])
    if d == "NO TRADE":
        return "FOREX: No trade — trend not strong enough."

    hold = _hold_window("FX", best_row["vol"])

    # SSI overlay: if risk-off, bias toward USD strength / defensive stance
    if ssi <= 3:
        return f"FOREX: Risk OFF — if trading at all, keep it SMALL. {d} {best_row['symbol']} — hold {hold}."
    return f"FOREX: {d} {best_row['symbol']} — hold position for at least {hold}."


def recommend_options(best_row, ssi: float):
    """
    Rule-based option recommendation (no option chain).
    Outputs:
      - Iron condor (range / neutral)
      - Lotto (directional + high vol + strong trend)
      - Defined-risk directional (moderate)
    Provides expiry window and strike targets as % offsets.
    """
    if not best_row or best_row["score"] < MIN_OPTIONS_SCORE:
        return "OPTIONS: No trade — no underlying cleared the threshold."

    sym = best_row["symbol"]
    px = float(best_row["last"])
    trend = best_row["trend"]
    vol = best_row["vol"]
    d = _direction(trend)

    # If SSI is risk-off, avoid lottos; prefer defined risk or condors
    risk_off = ssi <= 3

    # Decide strategy
    if d == "NO TRADE" or (4 <= trend <= 6):
        # Neutral-ish: condor candidate
        width_pct = 0.015 if vol >= 6 else 0.012  # 1.2–1.5% wings baseline
        short_put = _pct_to_strike(px, -width_pct)
        short_call = _pct_to_strike(px, +width_pct)
        long_put = _pct_to_strike(px, -(width_pct + 0.01))
        long_call = _pct_to_strike(px, +(width_pct + 0.01))

        return (
            f"OPTIONS: IRON CONDOR on {sym} — expiry {CONDOR_DTE}. "
            f"Sell PUT spread: short ~{short_put}, long ~{long_put}. "
            f"Sell CALL spread: short ~{short_call}, long ~{long_call}."
        )

    # Directional strategies
    if vol >= 7 and not risk_off and d in ("LONG", "SHORT"):
        # Lotto (small size)
        otm_pct = 0.02 if vol >= 8 else 0.015  # ~1.5–2% OTM
        strike = _pct_to_strike(px, +otm_pct) if d == "LONG" else _pct_to_strike(px, -otm_pct)
        side = "CALL" if d == "LONG" else "PUT"
        return (
            f"OPTIONS: LOTTO {side} on {sym} — expiry {LOTTO_DTE}. "
            f"Target strike ~{strike} ({'OTM' if otm_pct > 0 else ''}). "
            f"Plan: quick exit on spike; cut if momentum dies."
        )

    # Default defined-risk directional
    otm_pct = 0.01  # ~1% OTM
    strike = _pct_to_strike(px, +otm_pct) if d == "LONG" else _pct_to_strike(px, -otm_pct)
    side = "CALL" if d == "LONG" else "PUT"
    return (
        f"OPTIONS: DEFINED-RISK {side} on {sym} — expiry {DIRECTIONAL_DTE}. "
        f"Target strike ~{strike} (~1% OTM). "
        f"Prefer spreads if premiums are expensive."
    )
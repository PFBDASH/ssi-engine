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
            last = float(df["close"].iloc[-1])

            rows.append({
                "symbol": sym,
                "last": last,
                "trend": t,
                "vol": v,
                "score": score
            })
        except Exception:
            # keep engine resilient; failures just skip
            continue
    return sorted(rows, key=lambda x: x["score"], reverse=True)


def run_crypto_scan():
    # Kraken hourly OHLC
    return _score_rows(CRYPTO_SYMBOLS, lambda s: fetch_crypto_ohlc(s, interval=60))


def run_fx_scan():
    # Stooq daily OHLC
    return _score_rows(FOREX_SYMBOLS, fetch_fx_daily)


def run_options_scan():
    # Options lane = best UNDERLYINGS to consider for options
    return _score_rows(OPTIONS_UNDERLYINGS, fetch_equity_daily)


def compute_ssi(crypto_rows, fx_rows=None, opt_rows=None):
    """
    SSI = average of top crypto scores (simple + stable).
    """
    if not crypto_rows:
        return 0.0
    top = crypto_rows[:5] if len(crypto_rows) >= 5 else crypto_rows
    return round(sum(r["score"] for r in top) / len(top), 1)


# ---------------- OPTIONS HEURISTICS (no option-chain data required) ----------------

def _round_strike(sym: str, strike: float) -> float:
    """
    Rough rounding rules so strikes look realistic:
    - ETFs (SPY/QQQ/IWM): nearest $1
    - High-priced equities: nearest $5
    """
    sym = sym.upper()
    if sym in ["SPY", "QQQ", "IWM"]:
        step = 1.0
    else:
        step = 5.0
    return round(strike / step) * step


def options_playbook(underlying_row: dict, ssi: float) -> dict:
    """
    Returns a suggestion of:
    - bias (CALL/PUT/NEUTRAL)
    - structure (LOTTO / DEFINED-RISK / STAND DOWN)
    - expiry_days (0 or 3 or 7)
    - target_strike (approx)
    These are heuristics, not guarantees.
    """
    sym = underlying_row["symbol"]
    last = float(underlying_row["last"])
    t = int(underlying_row["trend"])
    v = int(underlying_row["vol"])

    # Bias from trend score
    if t >= 7:
        bias = "CALL bias"
    elif t <= 3:
        bias = "PUT bias"
    else:
        bias = "NEUTRAL bias"

    # Structure decision
    lotto_env = (ssi >= 7 and v >= 7)
    if ssi <= 3:
        structure = "STAND DOWN"
        expiry_days = None
        mny = None
        strike = None
        rationale = "Risk-off regime. Lowest-quality environment for options aggression."
    elif lotto_env and bias != "NEUTRAL bias":
        structure = "LOTTO (small defined risk)"
        expiry_days = 0  # 0DTE idea
        mny = 0.02       # ~2% OTM
        if bias == "CALL bias":
            strike = _round_strike(sym, last * (1 + mny))
        else:
            strike = _round_strike(sym, last * (1 - mny))
        rationale = "Hot/volatile environment + directional trend. Keep risk capped."
    elif 4 <= ssi <= 6:
        structure = "DEFINED-RISK (range / spreads)"
        expiry_days = 7
        mny = 0.01  # tighter, ~1% zone for structure ideas
        if bias == "CALL bias":
            strike = _round_strike(sym, last * (1 + mny))
        elif bias == "PUT bias":
            strike = _round_strike(sym, last * (1 - mny))
        else:
            strike = _round_strike(sym, last)  # near-ATM reference
        rationale = "Chop/neutral regime. Favor defined-risk structures over lottos."
    else:
        structure = "DEFINED-RISK (directional)"
        expiry_days = 3
        mny = 0.015
        if bias == "CALL bias":
            strike = _round_strike(sym, last * (1 + mny))
        elif bias == "PUT bias":
            strike = _round_strike(sym, last * (1 - mny))
        else:
            strike = _round_strike(sym, last)
        rationale = "Moderate regime. Prefer controlled risk; avoid max-gamma gambling unless HOT."

    return {
        "underlying": sym,
        "last": round(last, 2),
        "bias": bias,
        "structure": structure,
        "expiry_days": expiry_days,
        "target_strike": strike,
        "rationale": rationale
    }
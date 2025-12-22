# engine.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import pandas as pd
import yfinance as yf


# -----------------------------
# Config / “Universe”
# -----------------------------
CRYPTO_UNIVERSE = [
    ("BTCUSD", "BTC-USD"),
    ("ETHUSD", "ETH-USD"),
    ("SOLUSD", "SOL-USD"),
    ("XRPUSD", "XRP-USD"),
    ("ADAUSD", "ADA-USD"),
]

FX_UNIVERSE = [
    ("EURUSD", "EURUSD=X"),
    ("GBPUSD", "GBPUSD=X"),
    ("USDJPY", "JPY=X"),       # yfinance uses JPY=X for USDJPY
    ("AUDUSD", "AUDUSD=X"),
    ("USDCHF", "CHF=X"),       # USDCHF proxy
]

OPTIONS_UNDERLYINGS = [
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("IWM", "IWM"),
    ("NVDA", "NVDA"),
    ("TSLA", "TSLA"),
]


# -----------------------------
# Scoring helpers
# -----------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _trend_score(close_now: float, sma_fast: float, sma_slow: float) -> int:
    # simple 0/5/10 bucketed score
    if close_now > sma_fast > sma_slow:
        return 10
    if close_now > sma_slow:
        return 5
    return 0


def _vol_score(atr_pct: float) -> int:
    # ATR% buckets -> 0/5/10-ish
    if atr_pct >= 0.04:
        return 10
    if atr_pct >= 0.02:
        return 5
    return 2


def _atr_pct(df: pd.DataFrame) -> float:
    # ATR(14) approximation from daily OHLC
    if df is None or df.empty or len(df) < 20:
        return 0.0
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    last = c.iloc[-1]
    if last == 0 or pd.isna(last) or pd.isna(atr):
        return 0.0
    return float(atr / last)


def _fetch_daily(symbol: str, days: int = 120) -> pd.DataFrame:
    # Use yfinance for a reliable “engine” baseline
    # (You can swap this to your preferred data source later.)
    period = f"{max(30, days)}d"
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Standardize columns
    df = df.rename(columns={c: c.title() for c in df.columns})
    return df


def _score_symbol(label: str, yf_symbol: str) -> Dict:
    df = _fetch_daily(yf_symbol, days=180)
    if df.empty or len(df) < 60:
        return {
            "symbol": label,
            "trend": 0,
            "vol": 0,
            "score": 0.0,
            "price": None,
        }

    close = df["Close"]
    px = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])

    t = _trend_score(px, sma20, sma50)
    v = _vol_score(_atr_pct(df))

    score = 0.65 * t + 0.35 * v
    return {"symbol": label, "trend": t, "vol": v, "score": float(round(score, 4)), "price": px}


# -----------------------------
# Public scan functions
# -----------------------------
def run_crypto_scan() -> pd.DataFrame:
    rows = [_score_symbol(label, sym) for (label, sym) in CRYPTO_UNIVERSE]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df[["symbol", "price", "trend", "vol", "score"]]


def run_fx_scan() -> pd.DataFrame:
    rows = [_score_symbol(label, sym) for (label, sym) in FX_UNIVERSE]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df[["symbol", "price", "trend", "vol", "score"]]


def run_options_scan() -> pd.DataFrame:
    rows = [_score_symbol(label, sym) for (label, sym) in OPTIONS_UNDERLYINGS]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df[["symbol", "price", "trend", "vol", "score"]]


# -----------------------------
# SSI + Recommendations
# -----------------------------
def compute_ssi(crypto_df: pd.DataFrame, fx_df: pd.DataFrame, options_df: pd.DataFrame) -> float:
    # simple blend: best-of each lane
    def top_score(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 0.0
        return _safe_float(df.iloc[0]["score"], 0.0)

    c = top_score(crypto_df)
    f = top_score(fx_df)
    o = top_score(options_df)

    # SSI scaled 0-10ish (since scores are 0-10)
    ssi = 0.4 * c + 0.3 * f + 0.3 * o
    return float(round(ssi, 3))


def recommend_lane(ssi: float, crypto_df: pd.DataFrame, fx_df: pd.DataFrame, options_df: pd.DataFrame) -> str:
    # Background thresholds (not displayed)
    RISK_ON = 6.5
    RISK_OFF = 4.0

    if ssi < RISK_OFF:
        return "Risk OFF — Stand Down (no trade recommended)."

    # pick best lane by top score
    lanes = [
        ("Crypto", _safe_float(crypto_df.iloc[0]["score"], 0.0) if not crypto_df.empty else 0.0),
        ("Forex", _safe_float(fx_df.iloc[0]["score"], 0.0) if not fx_df.empty else 0.0),
        ("Options", _safe_float(options_df.iloc[0]["score"], 0.0) if not options_df.empty else 0.0),
    ]
    lanes.sort(key=lambda x: x[1], reverse=True)
    lane, top = lanes[0]

    if ssi >= RISK_ON:
        return f"Risk ON — Favor {lane} (top score {round(top,2)})."
    return f"Mixed — Small size only. Best lane: {lane} (top score {round(top,2)})."


def recommend_crypto(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    if _safe_float(row["score"]) < 6.8:
        return None
    sym = row["symbol"]
    return f"Go LONG {sym}. Timeframe: look to exit within 24–72 hours unless momentum stalls."


def recommend_fx(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    if _safe_float(row["score"]) < 6.8:
        return None
    sym = row["symbol"]
    # FX direction inference from trend bucket only (placeholder logic)
    bias = "LONG" if int(row["trend"]) >= 5 else "SHORT"
    hold = "hold 2–7 days" if int(row["vol"]) <= 5 else "hold 1–3 days"
    return f"{bias} {sym}. Suggested holding window: {hold}."


# -----------------------------
# Options contract recommendation (the piece you want back)
# -----------------------------
def recommend_options_contract(options_df: pd.DataFrame) -> Optional[Dict]:
    """
    Returns a contract suggestion for the *top-ranked underlying*.
    This is heuristic (not broker quotes). It outputs:
      - strategy: Iron Condor or Lotto
      - symbol (underlying)
      - bias (CALL/PUT/NEUTRAL)
      - expiry (date string)
      - strike (approx)
      - est_premium (very rough placeholder)
    """
    if options_df is None or options_df.empty:
        return None

    top = options_df.iloc[0]
    score = _safe_float(top["score"])
    if score < 7.2:
        return None

    sym = str(top["symbol"])
    px = _safe_float(top.get("price", 0.0))

    # Rule of thumb:
    # - high vol regime -> iron condor
    # - strong trend + decent vol -> directional "lotto"
    vol = int(top["vol"])
    trend = int(top["trend"])

    today = datetime.utcnow().date()

    if vol >= 8 and trend <= 5:
        # Iron condor ~ 30-45 DTE
        expiry = (today + timedelta(days=35)).isoformat()
        # 10% wings placeholder
        short_put = round(px * 0.95, 0)
        short_call = round(px * 1.05, 0)
        return {
            "strategy": "IRON CONDOR (paper outline)",
            "symbol": sym,
            "bias": "NEUTRAL",
            "expiry": expiry,
            "strike": f"Sell {short_put}P / Sell {short_call}C (define wings)",
            "est_premium": "Varies (check chain)",
        }

    # Directional "lotto" 5-10 DTE
    expiry = (today + timedelta(days=7)).isoformat()
    if trend >= 10:
        bias = "CALL"
        strike = round(px * 1.05, 0)
    elif trend <= 0:
        bias = "PUT"
        strike = round(px * 0.95, 0)
    else:
        # if trend is mixed but score is high, pick mild direction
        bias = "CALL"
        strike = round(px * 1.03, 0)

    return {
        "strategy": "LOTTO (defined-risk)",
        "symbol": sym,
        "bias": bias,
        "expiry": expiry,
        "strike": strike,
        "est_premium": "Aim $30–$100 (check chain)",
    }
# engine.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

import universe


# -----------------------------
# Helpers
# -----------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make yfinance output consistent.
    Handles:
      - empty dfs
      - MultiIndex columns (e.g. ('Close','SPY'))
      - lowercase/uppercase variations
    Returns columns: Open, High, Low, Close, Volume (if available)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # If MultiIndex columns, flatten by taking first level or last level as needed
    if isinstance(df.columns, pd.MultiIndex):
        # Usually it's like ('Close', 'SPY'). We want just 'Close'.
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Standardize names
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "open":
            rename[c] = "Open"
        elif cl == "high":
            rename[c] = "High"
        elif cl == "low":
            rename[c] = "Low"
        elif cl in ("close", "adj close", "adjclose"):
            # prefer Close; yfinance returns Close in history()
            rename[c] = "Close"
        elif cl == "volume":
            rename[c] = "Volume"

    df = df.rename(columns=rename)

    needed = ["Close"]
    for n in needed:
        if n not in df.columns:
            return pd.DataFrame()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()

    # Drop rows with no close
    df = df.dropna(subset=["Close"])
    return df


def _fetch_history(symbol: str, days: int = 200, interval: str = "1d") -> pd.DataFrame:
    """
    More reliable than yf.download in Streamlit Cloud.
    """
    try:
        end = _now_utc()
        start = end - timedelta(days=days)
        t = yf.Ticker(symbol)
        df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
        df = _normalize_ohlc(df)
        return df
    except Exception:
        return pd.DataFrame()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not {"High", "Low", "Close"}.issubset(df.columns):
        # fallback: approximate ATR from Close
        return df["Close"].pct_change().abs().rolling(period).mean() * df["Close"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_trend(df: pd.DataFrame) -> tuple[float, dict]:
    """
    Trend score 0..10 using EMA slope + EMA alignment.
    """
    close = df["Close"]
    if len(close) < 60:
        return 0.0, {"reason": "insufficient_bars"}

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # slope of ema50 over last 10 bars (normalized)
    slope = (ema50.iloc[-1] - ema50.iloc[-11]) / max(1e-9, close.iloc[-1])
    slope_score = _clamp((slope * 1000) + 5, 0, 10)  # heuristic

    align = 0.0
    if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
        align = 10.0
    elif ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
        align = 8.0
    else:
        align = 4.0

    trend = 0.6 * slope_score + 0.4 * align
    return float(_clamp(trend, 0, 10)), {
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "ema200": float(ema200.iloc[-1]),
        "slope": float(slope),
    }


def _score_vol(df: pd.DataFrame) -> tuple[float, dict]:
    """
    Vol score 0..10 using ATR% (higher ATR% = higher vol score).
    """
    close = df["Close"]
    if len(close) < 30:
        return 0.0, {"reason": "insufficient_bars"}

    a = _atr(df, 14)
    atr_last = float(a.iloc[-1]) if not np.isnan(a.iloc[-1]) else 0.0
    price = float(close.iloc[-1])

    atrp = (atr_last / price) if price > 0 else 0.0  # ATR %
    # map 0%..8% to 0..10
    vol = _clamp((atrp / 0.08) * 10, 0, 10)
    return float(vol), {"atr": atr_last, "atrp": atrp}


def _regime_label(trend_score: float, vol_score: float) -> str:
    # simple bucket
    if trend_score >= 7 and vol_score <= 4:
        return "Trend + Calm"
    if trend_score >= 7 and vol_score > 4:
        return "Trend + Volatile"
    if trend_score < 7 and vol_score > 6:
        return "Chop + Volatile"
    return "Chop / Range"


def _lane_reco_crypto(symbol: str, ssi: float) -> str:
    if ssi < 7:
        return ""
    # crypto "lotto-ish" but still actionable
    return f"Go LONG {symbol.replace('-USD','')} (spot). Target 1–3 days; exit on +3% to +6% move or break of last swing low."


def _lane_reco_fx(symbol: str, ssi: float, trend_score: float) -> str:
    if ssi < 7:
        return ""
    # Directional bias from trend score
    direction = "LONG" if trend_score >= 7 else "SHORT"
    clean = symbol.replace("=X", "")
    return f"{direction} {clean}. Hold 3–7 days. Exit if price closes against the direction for 2 consecutive days."


def _next_weekly_expiry() -> str:
    # nearest Friday at least 5 calendar days out
    d = datetime.now().date()
    # move to next Friday
    days_ahead = (4 - d.weekday()) % 7
    if days_ahead < 5:
        days_ahead += 7
    exp = d + timedelta(days=days_ahead)
    return exp.isoformat()


def _next_monthly_expiry() -> str:
    # “~30-45D” approximation: next 3rd Friday of next month
    d = datetime.now().date()
    year = d.year + (1 if d.month == 12 else 0)
    month = 1 if d.month == 12 else d.month + 1
    first = datetime(year, month, 1).date()
    # find 3rd Friday
    fridays = []
    cur = first
    while cur.month == month:
        if cur.weekday() == 4:
            fridays.append(cur)
        cur += timedelta(days=1)
    exp = fridays[2] if len(fridays) >= 3 else fridays[-1]
    return exp.isoformat()


def _options_reco(underlying: str, price: float, trend_score: float, vol_score: float, ssi: float) -> str:
    """
    Produces either:
      - Lotto call/put (nearest weekly)
      - Iron condor (range/vol)
    NOTE: This is a heuristic contract suggestion, not option chain pricing.
    """
    if ssi < 7:
        return ""

    # If trending hard -> directional lotto
    if trend_score >= 7:
        exp = _next_weekly_expiry()
        strike = round(price * 1.03, 0)  # ~3% OTM
        return f"LOTTO: {underlying} CALL {strike:.0f} exp {exp}. Aim 1–5 days. Take profit fast (+50% to +150%)."
    if trend_score <= 3 and vol_score >= 6:
        exp = _next_weekly_expiry()
        strike = round(price * 0.97, 0)  # ~3% OTM
        return f"LOTTO: {underlying} PUT {strike:.0f} exp {exp}. Aim 1–5 days. Take profit fast (+50% to +150%)."

    # Otherwise: iron condor idea when range-y
    exp = _next_monthly_expiry()
    width = max(1.0, price * 0.03)  # ~3% wings
    short_put = round(price - width, 0)
    short_call = round(price + width, 0)
    long_put = round(price - (width * 1.8), 0)
    long_call = round(price + (width * 1.8), 0)
    return (
        f"IRON CONDOR: {underlying} exp {exp} | "
        f"Sell {short_put:.0f}P / Buy {long_put:.0f}P and Sell {short_call:.0f}C / Buy {long_call:.0f}C. "
        f"Target 30–45D, manage at 25–50% max profit."
    )


def _score_symbol(label: str, symbol: str) -> dict:
    df = _fetch_history(symbol, days=240, interval="1d")
    if df.empty or len(df) < 60:
        return {
            "lane": label,
            "symbol": symbol,
            "last": np.nan,
            "trend": 0.0,
            "vol": 0.0,
            "ssi": 0.0,
            "regime": "No Data",
            "reco": "",
            "status": "no_data",
        }

    close = df["Close"]
    last = float(close.iloc[-1])
    trend_score, tmeta = _score_trend(df)
    vol_score, vmeta = _score_vol(df)

    # composite SSI (0..10)
    # trend weighted slightly higher than vol
    ssi = _clamp(0.65 * trend_score + 0.35 * vol_score, 0, 10)

    regime = _regime_label(trend_score, vol_score)

    reco = ""
    if label == "CRYPTO":
        reco = _lane_reco_crypto(symbol, ssi)
    elif label == "FOREX":
        reco = _lane_reco_fx(symbol, ssi, trend_score)
    elif label == "OPTIONS":
        reco = _options_reco(symbol, last, trend_score, vol_score, ssi)

    return {
        "lane": label,
        "symbol": symbol,
        "last": round(last, 4),
        "trend": round(trend_score, 2),
        "vol": round(vol_score, 2),
        "ssi": round(ssi, 2),
        "regime": regime,
        "reco": reco,
        "status": "ok",
    }


def _run_lane(label: str, symbols: list[str]) -> pd.DataFrame:
    rows = [_score_symbol(label, s) for s in symbols]
    df = pd.DataFrame(rows)

    # If df is empty (shouldn't), return empty with expected columns
    if df.empty:
        return pd.DataFrame(columns=["lane", "symbol", "last", "trend", "vol", "ssi", "regime", "reco", "status"])

    # sort high SSI first
    return df.sort_values(by="ssi", ascending=False, kind="mergesort").reset_index(drop=True)


def run_crypto_scan() -> pd.DataFrame:
    return _run_lane("CRYPTO", list(universe.CRYPTO))


def run_fx_scan() -> pd.DataFrame:
    return _run_lane("FOREX", list(universe.FOREX))


def run_options_scan() -> pd.DataFrame:
    return _run_lane("OPTIONS", list(universe.OPTIONS_UNDERLYINGS))


def run_full_scan() -> dict:
    crypto = run_crypto_scan()
    fx = run_fx_scan()
    opts = run_options_scan()

    # SSI headline: average of best-in-lane SSI (if present)
    def best_ssi(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        x = df["ssi"].dropna()
        return float(x.iloc[0]) if len(x) else 0.0

    headline = round((best_ssi(crypto) + best_ssi(fx) + best_ssi(opts)) / 3.0, 2)

    # Risk banner heuristic
    risk_banner = "Risk OFF — Stand Down" if headline < 5 else ("Selective — Small Size" if headline < 7 else "Risk ON — Favor Trend Setups")

    return {
        "headline_ssi": headline,
        "risk_banner": risk_banner,
        "crypto": crypto,
        "fx": fx,
        "options": opts,
    }
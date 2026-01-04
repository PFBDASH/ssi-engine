# engine.py
from __future__ import annotations

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import universe
import data_sources


# =========================================================
# Helpers
# =========================================================
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _map_linear(x: float, x0: float, x1: float, y0: float = 0.0, y1: float = 10.0) -> float:
    """Map x from [x0, x1] -> [y0, y1], clamp at ends."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = _clamp(t, 0.0, 1.0)
    return y0 + t * (y1 - y0)


def _to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming dataframes from data_sources:
      - Expect columns: time, open, high, low, close, volume (volume may be missing)
    Output:
      - DateTimeIndex
      - Columns: Open, High, Low, Close, Volume (Volume optional)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # time -> index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time")

    # Normalize names
    ren = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "open":
            ren[c] = "Open"
        elif cl == "high":
            ren[c] = "High"
        elif cl == "low":
            ren[c] = "Low"
        elif cl == "close":
            ren[c] = "Close"
        elif cl == "volume":
            ren[c] = "Volume"
    df = df.rename(columns=ren)

    # require Close
    if "Close" not in df.columns:
        return pd.DataFrame()

    # numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df.sort_index()
    return df


def _get_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna()


def _get_volume(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Volume" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Volume"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna()


# =========================================================
# Data fetch (uses your data_sources.py)
# =========================================================
def _crypto_symbol_to_kraken_pair(symbol: str) -> str:
    """
    Universe uses BTC-USD, ETH-USD, etc.
    Kraken expects BTCUSD, ETHUSD...
    """
    s = symbol.upper().strip()
    s = s.replace("-USD", "USD")
    s = "".join(ch for ch in s if ch.isalnum())
    return s


def _fx_symbol_to_stooq_pair(symbol: str) -> str:
    """
    Universe uses EURUSD=X. Stooq wants EURUSD.
    """
    s = symbol.upper().strip()
    s = s.replace("=X", "")
    s = "".join(ch for ch in s if ch.isalnum())
    return s


def _fetch_crypto_daily(symbol: str) -> pd.DataFrame:
    pair = _crypto_symbol_to_kraken_pair(symbol)
    # Kraken interval is minutes. 1440 = daily candles.
    df = data_sources.fetch_crypto_ohlc(pair, interval=1440)
    return _to_ohlcv(df)


def _fetch_fx_daily(symbol: str) -> pd.DataFrame:
    pair = _fx_symbol_to_stooq_pair(symbol)
    df = data_sources.fetch_fx_daily(pair)
    return _to_ohlcv(df)


def _fetch_equity_daily(symbol: str) -> pd.DataFrame:
    sym = symbol.upper().strip()
    df = data_sources.fetch_equity_daily(sym)
    return _to_ohlcv(df)


def _fetch_history(label: str, symbol: str) -> pd.DataFrame:
    """
    label: CRYPTO, FOREX, OPTIONS, LC
    """
    if label == "CRYPTO":
        return _fetch_crypto_daily(symbol)
    if label == "FOREX":
        return _fetch_fx_daily(symbol)
    # OPTIONS + LC = US equities universe only
    return _fetch_equity_daily(symbol)


# =========================================================
# Indicators (same concept as before)
# =========================================================
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # If High/Low missing, approximate from abs returns
    if not {"High", "Low", "Close"}.issubset(df.columns):
        c = _get_close(df)
        if c.empty:
            return pd.Series(dtype=float)
        return c.pct_change().abs().rolling(period).mean() * c

    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = _get_close(df)
    if close.empty:
        return pd.Series(dtype=float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _score_trend(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    close = _get_close(df)
    if len(close) < 60:
        return 0.0, {"reason": "insufficient_bars"}

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    slope = (ema50.iloc[-1] - ema50.iloc[-11]) / max(1e-9, close.iloc[-1])
    slope_score = _clamp((slope * 1000) + 5, 0, 10)

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


def _score_vol(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    close = _get_close(df)
    if len(close) < 30:
        return 0.0, {"reason": "insufficient_bars"}

    a = _atr(df, 14)
    if a.empty or np.isnan(a.iloc[-1]):
        return 0.0, {"reason": "atr_unavailable"}

    atr_last = float(a.iloc[-1])
    price = float(close.iloc[-1])
    atrp = (atr_last / price) if price > 0 else 0.0

    vol = _clamp((atrp / 0.08) * 10, 0, 10)  # map 0..8% to 0..10
    return float(vol), {"atr": atr_last, "atrp": atrp}


def _regime_label(trend_score: float, vol_score: float) -> str:
    if trend_score >= 7 and vol_score <= 4:
        return "Trend + Calm"
    if trend_score >= 7 and vol_score > 4:
        return "Trend + Volatile"
    if trend_score < 7 and vol_score > 6:
        return "Chop + Volatile"
    return "Chop / Range"


# =========================================================
# Lane 4: Long Cycle / Phase-4
# =========================================================
def _phase4_label(score: float) -> str:
    if score >= 8.5:
        return "Phase-4: High"
    if score >= 7.0:
        return "Phase-4: Strong"
    if score >= 5.5:
        return "Phase-4: Watch"
    return "Phase-4: Low"


def _score_phase4(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> Tuple[float, Dict[str, Any]]:
    close = _get_close(df)
    vol = _get_volume(df)

    # Need ~1y daily bars for base detection
    if close.empty or len(close) < 260:
        return 0.0, {"reason": "insufficient_bars"}

    w_base = 252
    w_mid = 126
    w_short = 20

    c_base = close.iloc[-w_base:]
    c_mid = close.iloc[-w_mid:]
    price = float(close.iloc[-1])

    # 1) Base compression: 1y range percent (lower is better)
    rng_pct = (float(c_base.max()) - float(c_base.min())) / max(1e-9, float(c_base.mean()))
    base_score = 10.0 - _map_linear(rng_pct, 0.15, 0.60, 0.0, 10.0)

    # 2) Volatility drying: short ATR% vs base ATR%
    atr = _atr(df, 14)
    if atr.empty or np.isnan(atr.iloc[-1]):
        return 0.0, {"reason": "atr_unavailable"}

    atrp_short = float(atr.iloc[-w_short:].mean()) / max(1e-9, price)
    atrp_base = float(atr.iloc[-w_base:].mean()) / max(1e-9, float(c_base.mean()))
    vol_ratio = atrp_short / max(1e-9, atrp_base)
    voldry_score = 10.0 - _map_linear(vol_ratio, 0.60, 1.25, 0.0, 10.0)

    # 3) Volume drying
    if vol.empty or len(vol) < w_base:
        vdry_score = 5.0
        v_ratio = None
    else:
        v_base = float(vol.iloc[-w_base:].mean())
        v_short = float(vol.iloc[-w_short:].mean())
        v_ratio = v_short / max(1e-9, v_base)
        vdry_score = 10.0 - _map_linear(v_ratio, 0.70, 1.10, 0.0, 10.0)

    # 4) Relative strength vs benchmark (SPY)
    rs_score = 5.0
    rs_delta = None
    if benchmark_df is not None and not benchmark_df.empty:
        b_close = _get_close(benchmark_df)
        if len(b_close) >= w_mid and len(close) >= w_mid:
            sym_ret = float(c_mid.iloc[-1] / c_mid.iloc[0] - 1.0)
            b_seg = b_close.iloc[-w_mid:]
            b_ret = float(b_seg.iloc[-1] / b_seg.iloc[0] - 1.0)
            rs_delta = sym_ret - b_ret
            rs_score = _map_linear(rs_delta, -0.10, 0.10, 0.0, 10.0)

    # 5) Not extended: price position inside base
    base_high = float(c_base.max())
    base_low = float(c_base.min())
    pos = (price - base_low) / max(1e-9, (base_high - base_low))
    if pos < 0.60:
        extent_score = _map_linear(pos, 0.20, 0.60, 0.0, 10.0)
    elif pos <= 0.85:
        extent_score = 10.0
    else:
        extent_score = 10.0 - _map_linear(pos, 0.85, 1.00, 0.0, 10.0)

    score = (
        0.26 * base_score +
        0.22 * voldry_score +
        0.18 * vdry_score +
        0.22 * rs_score +
        0.12 * extent_score
    )
    score = float(_clamp(score, 0.0, 10.0))

    details = {
        "range_pct_1y": float(rng_pct),
        "vol_ratio": float(vol_ratio),
        "volume_ratio": float(v_ratio) if v_ratio is not None else None,
        "rs_delta_6m": float(rs_delta) if rs_delta is not None else None,
        "base_position": float(pos),
        "subscores": {
            "base": float(_clamp(base_score, 0, 10)),
            "voldry": float(_clamp(voldry_score, 0, 10)),
            "vdry": float(_clamp(vdry_score, 0, 10)),
            "rs": float(_clamp(rs_score, 0, 10)),
            "extent": float(_clamp(extent_score, 0, 10)),
        }
    }
    return score, details


# =========================================================
# Recommendations
# =========================================================
def _lane_reco_crypto(symbol: str, ssi: float) -> str:
    if ssi < 7:
        return ""
    return f"LONG {symbol.replace('-USD','')} (spot). Target 1–3 days; exit on +3% to +6% move or break of last swing low."


def _lane_reco_fx(symbol: str, ssi: float, trend_score: float) -> str:
    if ssi < 7:
        return ""
    direction = "LONG" if trend_score >= 7 else "SHORT"
    clean = symbol.replace("=X", "")
    return f"{direction} {clean}. Hold 3–7 days. Exit if price closes against the direction for 2 consecutive days."


def _next_weekly_expiry() -> str:
    d = datetime.now().date()
    days_ahead = (4 - d.weekday()) % 7  # Friday
    if days_ahead < 5:
        days_ahead += 7
    return (d + timedelta(days=days_ahead)).isoformat()


def _next_monthly_expiry() -> str:
    d = datetime.now().date()
    year = d.year + (1 if d.month == 12 else 0)
    month = 1 if d.month == 12 else d.month + 1
    first = datetime(year, month, 1).date()

    fridays = []
    cur = first
    while cur.month == month:
        if cur.weekday() == 4:
            fridays.append(cur)
        cur += timedelta(days=1)

    exp = fridays[2] if len(fridays) >= 3 else fridays[-1]
    return exp.isoformat()


def _options_reco(underlying: str, price: float, trend_score: float, vol_score: float, ssi: float) -> str:
    if ssi < 7:
        return ""

    if trend_score >= 7:
        exp = _next_weekly_expiry()
        strike = round(price * 1.03, 0)
        return f"{underlying} CALL {strike:.0f} exp {exp}. Aim 1–5 days. Take profit fast (+50% to +150%)."

    if trend_score <= 3 and vol_score >= 6:
        exp = _next_weekly_expiry()
        strike = round(price * 0.97, 0)
        return f"{underlying} PUT {strike:.0f} exp {exp}. Aim 1–5 days. Take profit fast (+50% to +150%)."

    exp = _next_monthly_expiry()
    width = max(1.0, price * 0.03)
    short_put = round(price - width, 0)
    short_call = round(price + width, 0)
    long_put = round(price - (width * 1.8), 0)
    long_call = round(price + (width * 1.8), 0)

    return (
        f"IRON CONDOR: {underlying} exp {exp} | "
        f"Sell {short_put:.0f}P / Buy {long_put:.0f}P and Sell {short_call:.0f}C / Buy {long_call:.0f}C. "
        "Target 30–45D, manage at 25–50% max profit."
    )


def _lane_reco_lc(symbol: str, phase4: float) -> str:
    if phase4 < 7.0:
        return ""
    return (
        f"Phase-4 watch: {symbol}. "
        "Small starter size is fine. Add only if it breaks out above its base highs on stronger volume. "
        "If it loses the base low, step aside."
    )


# =========================================================
# Scorers (single symbol)
# =========================================================
def _score_symbol(label: str, symbol: str) -> Dict[str, Any]:
    try:
        df = _fetch_history(label, symbol)
        close = _get_close(df)
        if close.empty or len(close) < 60:
            return {"lane": label, "symbol": symbol, "status": "no_data"}

        last = float(close.iloc[-1])
        trend_score, _ = _score_trend(df)
        vol_score, _ = _score_vol(df)

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
    except Exception:
        return {"lane": label, "symbol": symbol, "status": "no_data"}


def _score_symbol_lc(symbol: str, bench_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    try:
        df = _fetch_history("LC", symbol)
        close = _get_close(df)
        if close.empty or len(close) < 260:
            return {"lane": "LC", "symbol": symbol, "status": "no_data"}

        last = float(close.iloc[-1])
        phase4, _details = _score_phase4(df, bench_df)
        regime = _phase4_label(phase4)
        reco = _lane_reco_lc(symbol, phase4)

        return {
            "lane": "LC",
            "symbol": symbol,
            "last": round(last, 4),
            "phase4": round(phase4, 2),
            "regime": regime,
            "reco": reco,
            "status": "ok",
        }
    except Exception:
        return {"lane": "LC", "symbol": symbol, "status": "no_data"}


# =========================================================
# Parallel lane runners
# =========================================================
def _run_parallel(rows_fn, symbols: List[str], max_workers: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not symbols:
        return out

    mw = max(2, min(max_workers, 16))

    with ThreadPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(rows_fn, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                out.append(fut.result())
            except Exception:
                out.append({"symbol": futs[fut], "status": "no_data"})
    return out


def _run_lane(label: str, symbols: List[str]) -> pd.DataFrame:
    rows = _run_parallel(lambda s: _score_symbol(label, s), symbols, max_workers=10)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["lane", "symbol", "last", "trend", "vol", "ssi", "regime", "reco", "status"])
    # keep only ok rows near the top, but preserve others
    if "ssi" in df.columns:
        df["ssi"] = pd.to_numeric(df["ssi"], errors="coerce")
        df = df.sort_values(by="ssi", ascending=False, kind="mergesort")
    return df.reset_index(drop=True)


def _run_lane_lc(symbols: List[str]) -> pd.DataFrame:
    # Fetch SPY once for RS
    try:
        bench_df = _fetch_equity_daily("SPY")
    except Exception:
        bench_df = pd.DataFrame()

    rows = _run_parallel(lambda s: _score_symbol_lc(s, bench_df), symbols, max_workers=10)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["lane", "symbol", "last", "phase4", "regime", "reco", "status"])
    if "phase4" in df.columns:
        df["phase4"] = pd.to_numeric(df["phase4"], errors="coerce")
        df = df.sort_values(by="phase4", ascending=False, kind="mergesort")
    return df.reset_index(drop=True)


# =========================================================
# Public API
# =========================================================
def run_crypto_scan() -> pd.DataFrame:
    return _run_lane("CRYPTO", list(getattr(universe, "CRYPTO", [])))


def run_fx_scan() -> pd.DataFrame:
    return _run_lane("FOREX", list(getattr(universe, "FOREX", [])))


def run_options_scan() -> pd.DataFrame:
    return _run_lane("OPTIONS", list(getattr(universe, "OPTIONS_UNDERLYINGS", [])))


def run_lc_scan() -> pd.DataFrame:
    lc = list(getattr(universe, "LC", []))
    return _run_lane_lc(lc)


def run_full_scan() -> Dict[str, Any]:
    crypto = run_crypto_scan()
    fx = run_fx_scan()
    opts = run_options_scan()
    lc = run_lc_scan()

    def best_score(df: pd.DataFrame, col: str) -> float:
        if df is None or df.empty or col not in df.columns:
            return 0.0
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(x.iloc[0]) if len(x) else 0.0

    headline = round(
        (best_score(crypto, "ssi") + best_score(fx, "ssi") + best_score(opts, "ssi") + best_score(lc, "phase4")) / 4.0,
        2
    )

    if headline < 5:
        risk_banner = "Risk OFF — Stand Down"
    elif headline < 7:
        risk_banner = "Selective — Small Size"
    else:
        risk_banner = "Risk ON — Favor Trend Setups"

    return {
        "headline_ssi": headline,
        "risk_banner": risk_banner,
        "crypto": crypto,
        "fx": fx,
        "options": opts,
        "lc": lc,
    }
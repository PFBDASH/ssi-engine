# engine.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from universe import CRYPTO, FOREX, OPTIONS_UNDERLYINGS


# ---------------------------
# CONFIG (hidden thresholds)
# ---------------------------
CRYPTO_REC_THRESHOLD = 7.5
FOREX_REC_THRESHOLD = 7.0
OPTIONS_REC_THRESHOLD = 7.0

LOOKBACK_DAYS = 220  # enough for ema200 + stability


@dataclass
class ScoreRow:
    symbol: str
    trend: float
    vol: float
    score: float
    direction: str
    hold: str
    rec: str


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return float(x)
    except Exception:
        return default


def _normalize_yf_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return:
    - empty df
    - columns with lowercase
    - MultiIndex columns in some cases
    This normalizes to columns: Open, High, Low, Close, Volume (when available)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # If MultiIndex columns, pick the first level that looks like OHLCV
    if isinstance(df.columns, pd.MultiIndex):
        # Try to find a column level that contains 'Close'
        # If it's like ('Close', 'SPY') we want 'Close' per column.
        # Easiest: take the first ticker's slice if present.
        try:
            # If 2 levels: (field, ticker) or (ticker, field)
            lvl0 = df.columns.get_level_values(0)
            lvl1 = df.columns.get_level_values(1)
            # Determine which level has OHLC names
            ohlc = {"open", "high", "low", "close", "adj close", "volume"}
            if any(str(x).lower() in ohlc for x in set(lvl0)):
                # columns are (field, ticker)
                # choose first ticker
                first_ticker = list(dict.fromkeys(lvl1))[0]
                df = df.xs(first_ticker, axis=1, level=1)
            elif any(str(x).lower() in ohlc for x in set(lvl1)):
                # columns are (ticker, field)
                first_ticker = list(dict.fromkeys(lvl0))[0]
                df = df.xs(first_ticker, axis=1, level=0)
            else:
                # fallback: flatten
                df.columns = ["_".join(map(str, c)) for c in df.columns]
        except Exception:
            df.columns = ["_".join(map(str, c)) for c in df.columns]

    # Standardize column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    colmap = {c: c.title() for c in df.columns}
    df.rename(columns=colmap, inplace=True)

    # Some feeds return "Adj Close" only; keep Close if present
    return df


def _fetch_daily(yf_symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    df = yf.download(
        yf_symbol,
        period=f"{days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    df = _normalize_yf_history(df)
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return 0.0
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    last_close = close.iloc[-1]
    if last_close == 0 or np.isnan(last_close):
        return 0.0
    return float((atr.iloc[-1] / last_close) * 100.0)


def _trend_score(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Returns (trend_score 0-10, direction: LONG/SHORT/NEUTRAL)
    """
    if df.empty or "Close" not in df.columns or len(df) < 60:
        return 0.0, "NEUTRAL"

    close = df["Close"].dropna()
    if len(close) < 60:
        return 0.0, "NEUTRAL"

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)

    px = close.iloc[-1]
    up = (ema20.iloc[-1] > ema50.iloc[-1]) and (px > ema200.iloc[-1])
    down = (ema20.iloc[-1] < ema50.iloc[-1]) and (px < ema200.iloc[-1])

    # slope proxy: recent EMA20 change vs price
    slope = _safe_float((ema20.iloc[-1] - ema20.iloc[-10]) / px * 100.0)

    score = 5.0
    score += 2.0 if up else 0.0
    score -= 2.0 if down else 0.0
    score += np.clip(slope, -2.0, 2.0)

    direction = "LONG" if up else ("SHORT" if down else "NEUTRAL")
    return float(np.clip(score, 0.0, 10.0)), direction


def _vol_score(df: pd.DataFrame) -> float:
    """
    Volatility score 0-10 using ATR% buckets
    """
    atrp = _atr_pct(df)
    # simple bucket mapping
    if atrp <= 0.5:
        return 2.0
    if atrp <= 1.0:
        return 4.0
    if atrp <= 2.0:
        return 6.0
    if atrp <= 3.5:
        return 8.0
    return 10.0


def _composite_score(trend: float, vol: float) -> float:
    # weight trend slightly higher than vol
    return float(np.clip(0.65 * trend + 0.35 * vol, 0.0, 10.0))


def _hold_time(lane: str, direction: str, vol_score: float) -> str:
    if lane == "CRYPTO":
        return "Exit within 6–24h (tight risk), or 2–3 days if trend persists"
    if lane == "FOREX":
        return "Hold 2–7 days (swing), reassess daily"
    if lane == "OPTIONS":
        # higher vol -> shorter
        return "0–5 days (lotto) or 7–21 days (condor), manage risk aggressively"
    return "TBD"


def _lane_recommendation(
    lane: str,
    label: str,
    direction: str,
    score: float,
    threshold: float,
) -> str:
    if score < threshold or direction == "NEUTRAL":
        return "No trade — stand down"
    if lane == "CRYPTO":
        if direction == "LONG":
            return f"Go LONG {label} (spot)."
        return f"Go SHORT {label} only if you have margin/derivatives access."
    if lane == "FOREX":
        if direction == "LONG":
            return f"Go LONG {label}."
        return f"Go SHORT {label}."
    if lane == "OPTIONS":
        # options lane recommendation text is created elsewhere with strikes/expiry
        return "Options setup available (see below)."
    return "No trade."


def _score_symbol(lane: str, label: str, yf_symbol: str, invert_for_display: bool = False) -> ScoreRow:
    df = _fetch_daily(yf_symbol)
    if df.empty:
        return ScoreRow(
            symbol=label, trend=0.0, vol=0.0, score=0.0,
            direction="NEUTRAL", hold=_hold_time(lane, "NEUTRAL", 0.0),
            rec="No data"
        )

    # For pairs like JPY=X, CHF=X, yfinance returns USD per JPY/CHF.
    # If you want "USDJPY" displayed, we still score on the series as-is (direction often flips).
    trend, direction = _trend_score(df)
    vol = _vol_score(df)
    score = _composite_score(trend, vol)
    hold = _hold_time(lane, direction, vol)

    # If inverted pair, flip direction label so it matches USDJPY intuition
    if invert_for_display:
        if direction == "LONG":
            direction = "SHORT"
        elif direction == "SHORT":
            direction = "LONG"

    threshold = CRYPTO_REC_THRESHOLD if lane == "CRYPTO" else (FOREX_REC_THRESHOLD if lane == "FOREX" else OPTIONS_REC_THRESHOLD)
    rec = _lane_recommendation(lane, label, direction, score, threshold)

    return ScoreRow(
        symbol=label,
        trend=float(round(trend, 2)),
        vol=float(round(vol, 2)),
        score=float(round(score, 4)),
        direction=direction,
        hold=hold,
        rec=rec,
    )


def run_crypto_scan() -> pd.DataFrame:
    rows = []
    for label, yf_symbol in CRYPTO:
        r = _score_symbol("CRYPTO", label, yf_symbol)
        rows.append(r.__dict__)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def run_fx_scan() -> pd.DataFrame:
    rows = []
    for label, yf_symbol in FOREX:
        invert = label in ("USDJPY", "USDCHF")
        r = _score_symbol("FOREX", label, yf_symbol, invert_for_display=invert)
        rows.append(r.__dict__)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------
# OPTIONS CONTRACT DETAILS
# ---------------------------

def _pick_near_expiry(expirations: List[str], target_days: int) -> Optional[str]:
    """
    Pick expiry closest to target_days in the future.
    """
    if not expirations:
        return None
    today = datetime.now(timezone.utc).date()
    best = None
    best_dist = 10**9
    for e in expirations:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            dist = abs((d - today).days - target_days)
            if (d - today).days >= 1 and dist < best_dist:
                best = e
                best_dist = dist
        except Exception:
            continue
    return best or expirations[0]


def _mid_price(row: pd.Series) -> float:
    bid = _safe_float(row.get("bid", np.nan), np.nan)
    ask = _safe_float(row.get("ask", np.nan), np.nan)
    last = _safe_float(row.get("lastPrice", np.nan), np.nan)
    if not np.isnan(bid) and not np.isnan(ask) and ask > 0:
        return float(round((bid + ask) / 2.0, 2))
    if not np.isnan(last):
        return float(round(last, 2))
    return 0.0


def _lotto_contract(tkr: yf.Ticker, underlying: str, direction: str) -> Dict[str, str]:
    """
    Lotto: nearest ~5-7 day expiry, ~5% OTM call/put
    """
    expirations = list(getattr(tkr, "options", []) or [])
    exp = _pick_near_expiry(expirations, target_days=5)
    if not exp:
        return {"strategy": "LOTTO", "detail": f"{underlying}: options not available"}

    chain = tkr.option_chain(exp)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    px_df = _fetch_daily(underlying)
    px = float(px_df["Close"].iloc[-1]) if (not px_df.empty and "Close" in px_df.columns) else np.nan
    if np.isnan(px) or px <= 0:
        return {"strategy": "LOTTO", "detail": f"{underlying}: price unavailable"}

    target_otm = 1.05 if direction == "LONG" else 0.95
    target_strike = px * target_otm

    if direction == "LONG":
        # Call
        calls["dist"] = (calls["strike"] - target_strike).abs()
        row = calls.sort_values("dist").iloc[0]
        strike = float(row["strike"])
        price = _mid_price(row)
        return {
            "strategy": "LOTTO CALL",
            "underlying": underlying,
            "expiry": exp,
            "strike": f"{strike:.2f}",
            "est_price": f"{price:.2f}",
            "note": "Target: quick move; cut fast if wrong.",
        }
    else:
        # Put
        puts["dist"] = (puts["strike"] - target_strike).abs()
        row = puts.sort_values("dist").iloc[0]
        strike = float(row["strike"])
        price = _mid_price(row)
        return {
            "strategy": "LOTTO PUT",
            "underlying": underlying,
            "expiry": exp,
            "strike": f"{strike:.2f}",
            "est_price": f"{price:.2f}",
            "note": "Target: quick move; cut fast if wrong.",
        }


def _iron_condor_contract(tkr: yf.Ticker, underlying: str) -> Dict[str, str]:
    """
    Simple iron condor:
    - expiry ~14 days
    - short strikes around +/- 1 stdev move (approx), wings further out
    """
    expirations = list(getattr(tkr, "options", []) or [])
    exp = _pick_near_expiry(expirations, target_days=14)
    if not exp:
        return {"strategy": "IRON CONDOR", "detail": f"{underlying}: options not available"}

    chain = tkr.option_chain(exp)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    px_df = _fetch_daily(underlying)
    if px_df.empty or "Close" not in px_df.columns:
        return {"strategy": "IRON CONDOR", "detail": f"{underlying}: price unavailable"}

    close = px_df["Close"].dropna()
    px = float(close.iloc[-1])

    # crude stdev estimate from daily returns
    rets = close.pct_change().dropna()
    if len(rets) < 30:
        move = 0.03  # fallback 3%
    else:
        move = float(rets.tail(60).std() * np.sqrt(14))  # 14-day stdev
        move = float(np.clip(move, 0.02, 0.12))  # clamp

    put_short_target = px * (1 - move)
    call_short_target = px * (1 + move)
    put_long_target = px * (1 - move * 1.6)
    call_long_target = px * (1 + move * 1.6)

    # Pick nearest strikes
    puts["d_short"] = (puts["strike"] - put_short_target).abs()
    puts["d_long"] = (puts["strike"] - put_long_target).abs()
    calls["d_short"] = (calls["strike"] - call_short_target).abs()
    calls["d_long"] = (calls["strike"] - call_long_target).abs()

    p_short = puts.sort_values("d_short").iloc[0]
    p_long = puts.sort_values("d_long").iloc[0]
    c_short = calls.sort_values("d_short").iloc[0]
    c_long = calls.sort_values("d_long").iloc[0]

    # Ensure ordering (long farther OTM)
    ps = float(p_short["strike"])
    pl = float(p_long["strike"])
    cs = float(c_short["strike"])
    cl = float(c_long["strike"])
    if pl > ps:
        pl, ps = ps, pl
    if cl < cs:
        cl, cs = cs, cl

    # estimate mid prices
    credit_est = (_mid_price(p_short) + _mid_price(c_short)) - (_mid_price(p_long) + _mid_price(c_long))
    credit_est = float(round(max(credit_est, 0.0), 2))

    return {
        "strategy": "IRON CONDOR",
        "underlying": underlying,
        "expiry": exp,
        "put_long": f"{pl:.2f}",
        "put_short": f"{ps:.2f}",
        "call_short": f"{cs:.2f}",
        "call_long": f"{cl:.2f}",
        "est_credit": f"{credit_est:.2f}",
        "note": "Goal: premium capture; manage early if price threatens short strikes.",
    }


def run_options_scan() -> pd.DataFrame:
    """
    Scores underlyings and attaches a suggested options setup for the top candidate (if above threshold).
    """
    rows = []
    for sym in OPTIONS_UNDERLYINGS:
        r = _score_symbol("OPTIONS", sym, sym)
        rows.append(r.__dict__)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Build one detailed recommendation for the best candidate if strong enough
    best = df.iloc[0].to_dict()
    if best.get("score", 0.0) < OPTIONS_REC_THRESHOLD or best.get("direction") == "NEUTRAL":
        df["options_setup"] = ""
        return df

    underlying = best["symbol"]
    direction = best["direction"]
    tkr = yf.Ticker(underlying)

    # Decide condor vs lotto:
    # If vol is high and trend is middling => condor. If trend strong => lotto with direction.
    trend = float(best.get("trend", 0.0))
    vol = float(best.get("vol", 0.0))

    if vol >= 7.5 and trend <= 6.0:
        setup = _iron_condor_contract(tkr, underlying)
    else:
        setup = _lotto_contract(tkr, underlying, direction="LONG" if direction == "LONG" else "SHORT")

    # Put setup only on the top row so UI stays clean
    setups = [""] * len(df)
    setups[0] = setup
    df["options_setup"] = setups

    # Improve options lane rec text
    df.loc[0, "rec"] = f"Use {setup.get('strategy', 'OPTIONS')} on {underlying} (details below)."

    return df


def run_full_scan() -> Dict[str, object]:
    crypto_df = run_crypto_scan()
    fx_df = run_fx_scan()
    opt_df = run_options_scan()

    # Simple SSI: average of top scores across lanes (bounded)
    top_scores = []
    if not crypto_df.empty:
        top_scores.append(float(crypto_df.iloc[0]["score"]))
    if not fx_df.empty:
        top_scores.append(float(fx_df.iloc[0]["score"]))
    if not opt_df.empty:
        top_scores.append(float(opt_df.iloc[0]["score"]))

    ssi = float(np.clip(np.mean(top_scores) if top_scores else 0.0, 0.0, 10.0))

    # Risk banner
    if ssi < 4.0:
        banner = "Risk OFF — Stand Down"
    elif ssi < 6.5:
        banner = "Neutral — Only A+ setups"
    else:
        banner = "Risk ON — Selectively engage"

    return {
        "asof": _now_utc_str(),
        "ssi": round(ssi, 2),
        "banner": banner,
        "crypto": crypto_df,
        "fx": fx_df,
        "options": opt_df,
    }
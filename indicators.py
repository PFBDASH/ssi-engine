# indicators.py
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def trend_score(df: pd.DataFrame) -> int:
    """
    0..10 trend score.
    Uses EMA50/EMA200 alignment + recent slope.
    """
    if df is None or len(df) < 220:
        return 0

    close = df["close"]
    e50 = _ema(close, 50)
    e200 = _ema(close, 200)

    up = e50.iloc[-1] > e200.iloc[-1]
    down = e50.iloc[-1] < e200.iloc[-1]

    # simple slope proxy: last 10 bars change in EMA50
    slope = e50.iloc[-1] - e50.iloc[-11] if len(e50) > 11 else 0.0

    score = 5
    if up:
        score += 3
        if slope > 0:
            score += 2
    elif down:
        score -= 3
        if slope < 0:
            score -= 2

    return int(clamp(score, 0, 10))


def vol_score(df: pd.DataFrame) -> int:
    """
    0..10 volatility score.
    ATR(14) / price mapped to bucket.
    """
    if df is None or len(df) < 60:
        return 0

    atr = _atr(df, 14).iloc[-1]
    px = df["close"].iloc[-1]
    if px <= 0 or pd.isna(atr):
        return 0

    r = atr / px  # relative ATR
    # buckets tuned to be “sane” across FX/Equities/Crypto daily-ish data
    if r < 0.003:
        return 2
    if r < 0.006:
        return 4
    if r < 0.012:
        return 6
    if r < 0.020:
        return 8
    return 10


def composite_score(df: pd.DataFrame) -> float:
    t = trend_score(df)
    v = vol_score(df)
    return round(0.7 * t + 0.3 * v, 1)
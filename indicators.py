# indicators.py
import pandas as pd
import numpy as np

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def trend_score(df):
    ema50 = ema(df["close"], 50)
    ema200 = ema(df["close"], 200)

    slope = ema50.diff()
    score = 0

    if ema50.iloc[-1] > ema200.iloc[-1]:
        score += 5
    if slope.iloc[-1] > 0:
        score += 5

    return min(score, 10)

def vol_score(df):
    a = atr(df, 14).iloc[-1]
    a_avg = atr(df, 14).rolling(50).mean().iloc[-1]
    if a_avg == 0:
        return 0
    r = a / a_avg
    if r < 0.8:
        return 2
    elif r < 1.2:
        return 5
    elif r < 1.6:
        return 7
    else:
        return 9
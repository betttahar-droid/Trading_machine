"""
Trend-following strategy rules (Donchian breakout + ATR chandelier trailing stop).

Pure functions over CLOSED bars, shared by the backtester (backtest_trend.py) and the
live paper trader so both run exactly the same logic.

Rules (all evaluated at a bar's close, orders execute at the next bar's open):
- Entry long:  close > highest high of the previous `entry_n` bars   (and close > SMA(trend_n) if trend filter on)
- Entry short: close < lowest low of the previous `entry_n` bars     (and close < SMA(trend_n) if trend filter on)
- Exit long:   close < lowest low of the previous `exit_n` bars, or a resting stop at
               (highest close since entry - stop_atr * ATR) is touched intrabar
- Exit short:  mirror image
- Size:        risk `risk_pct` of equity between entry and the initial stop (stop_atr * ATR)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np


@dataclass
class TrendParams:
    interval: str = "1d"
    entry_n: int = 20
    exit_n: int = 10
    atr_n: int = 20
    stop_atr: float = 3.0
    trend_n: int = 0            # 0 disables the SMA trend filter
    allow_short: bool = True
    risk_pct: float = 0.01      # equity risked per trade at the initial stop
    max_gross_leverage: float = 3.0
    max_position_leverage: float = 1.0   # cap on one position's notional / equity
    taker_fee: float = 0.0005
    slippage: float = 0.0003


def atr_series(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int) -> np.ndarray:
    """Wilder ATR; value at i uses bars <= i."""
    prev = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum.reduce([highs - lows, np.abs(highs - prev), np.abs(lows - prev)])
    out = np.full(len(tr), np.nan)
    if len(tr) < n:
        return out
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        out[i] = out[i - 1] + (tr[i] - out[i - 1]) / n
    return out


def compute_features(candles: List[dict], p: TrendParams) -> Dict[str, np.ndarray]:
    h = np.array([c["high"] for c in candles], dtype=float)
    l = np.array([c["low"] for c in candles], dtype=float)
    c = np.array([c["close"] for c in candles], dtype=float)
    n = len(c)
    upper = np.full(n, np.nan); lower = np.full(n, np.nan)
    ex_upper = np.full(n, np.nan); ex_lower = np.full(n, np.nan)
    for i in range(n):
        if i >= p.entry_n:
            upper[i] = h[i - p.entry_n:i].max()
            lower[i] = l[i - p.entry_n:i].min()
        if i >= p.exit_n:
            ex_upper[i] = h[i - p.exit_n:i].max()
            ex_lower[i] = l[i - p.exit_n:i].min()
    sma = np.full(n, np.nan)
    if p.trend_n > 0 and n >= p.trend_n:
        cs = np.cumsum(np.insert(c, 0, 0.0))
        sma[p.trend_n - 1:] = (cs[p.trend_n:] - cs[:-p.trend_n]) / p.trend_n
    return {"high": h, "low": l, "close": c, "upper": upper, "lower": lower,
            "ex_upper": ex_upper, "ex_lower": ex_lower, "atr": atr_series(h, l, c, p.atr_n), "sma": sma}


def entry_signal(f: Dict[str, np.ndarray], i: int, p: TrendParams) -> Optional[str]:
    """'LONG' / 'SHORT' / None, known at the close of bar i."""
    if np.isnan(f["upper"][i]) or np.isnan(f["atr"][i]):
        return None
    close = f["close"][i]
    trend_ok_long = trend_ok_short = True
    if p.trend_n > 0:
        if np.isnan(f["sma"][i]):
            return None
        trend_ok_long, trend_ok_short = close > f["sma"][i], close < f["sma"][i]
    if close > f["upper"][i] and trend_ok_long:
        return "LONG"
    if p.allow_short and close < f["lower"][i] and trend_ok_short:
        return "SHORT"
    return None


def exit_on_close(f: Dict[str, np.ndarray], i: int, side: str) -> bool:
    """Channel exit, known at the close of bar i."""
    if side == "LONG":
        return not np.isnan(f["ex_lower"][i]) and f["close"][i] < f["ex_lower"][i]
    return not np.isnan(f["ex_upper"][i]) and f["close"][i] > f["ex_upper"][i]


def trail_stop(side: str, extreme_close: float, atr: float, current_stop: float, p: TrendParams) -> float:
    """Chandelier stop, only ever tightens."""
    if side == "LONG":
        return max(current_stop, extreme_close - p.stop_atr * atr)
    return min(current_stop, extreme_close + p.stop_atr * atr)


def position_size(equity: float, entry: float, atr: float, gross_used: float, p: TrendParams) -> float:
    """Notional in USD, risk-based and capped by per-position and portfolio leverage."""
    stop_dist_pct = p.stop_atr * atr / entry
    if stop_dist_pct <= 0:
        return 0.0
    notional = equity * p.risk_pct / stop_dist_pct
    notional = min(notional, equity * p.max_position_leverage)
    notional = min(notional, max(0.0, equity * p.max_gross_leverage - gross_used))
    return notional

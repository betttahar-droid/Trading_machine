"""
TradFi trend book: gold, silver, S&P 500 and Nasdaq 100 perpetuals on Binance Futures (USDT, "TradFi" tab).

Rule, as tested in cross_asset_lab.py (time-series momentum, Moskowitz-Ooi-Pedersen 2012): at the start of each
month, hold an asset only if its 12-month return is positive, sized so that each asset carries target_vol / sqrt(n)
of yearly volatility (60-day volatility), long or flat. The perps only exist since 2026, so the signal is computed
from the matching ETFs' daily history on Yahoo Finance; positions are marked and filled at the perps' Binance price.

Risk level R (the one number the user picks) sets both books so they carry equal risk, as in the backtest:
crypto trend risk per trade = 0.68% x R, TradFi yearly volatility target = 14.6% x R.
"""

from typing import Dict

import numpy as np
import pandas as pd

ASSETS: Dict[str, str] = {"XAUUSDT": "GLD", "XAGUSDT": "SLV", "SPYUSDT": "SPY", "QQQUSDT": "QQQ"}
NAMES = {"XAUUSDT": "Gold", "XAGUSDT": "Silver", "SPYUSDT": "S&P 500", "QQQUSDT": "Nasdaq 100"}
CRYPTO_RISK_PER_LEVEL = 0.0068      # crypto trend risk per trade at risk level 1 (%)
TRADFI_VOL_PER_LEVEL = 0.146        # TradFi book yearly volatility target at risk level 1
PER_ASSET_CAP = 1.5                 # max notional per asset, x equity
GROSS_CAP = 3.0                     # max TradFi notional in total, x equity
MOM_DAYS, VOL_DAYS = 252, 60


def etf_history(days: int = 420) -> pd.DataFrame:
    """Daily adjusted closes of the signal ETFs (Yahoo Finance)."""
    import yfinance as yf
    tickers = sorted(set(ASSETS.values()))
    px = yf.download(tickers, period=f"{days}d", progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px.dropna(how="all")


def signals(px: pd.DataFrame) -> Dict[str, dict]:
    """Per ETF: 12-month return and 60-day yearly volatility from the last complete daily closes."""
    out = {}
    for etf in px.columns:
        s = px[etf].dropna()
        if len(s) < MOM_DAYS + 1:
            continue
        rets = s.pct_change().dropna()
        out[etf] = {"mom12": float(s.iloc[-1] / s.iloc[-1 - MOM_DAYS] - 1),
                    "vol60": float(rets.iloc[-VOL_DAYS:].std() * np.sqrt(252)),
                    "asof": s.index[-1].strftime("%Y-%m-%d")}
    return out


def target_weights(sig: Dict[str, dict], target_vol: float) -> Dict[str, float]:
    """Notional per perp as a multiple of equity (0 = flat), with per-asset and total caps."""
    have = {perp: sig[etf] for perp, etf in ASSETS.items() if etf in sig and sig[etf]["vol60"] > 0}
    if not have:
        return {}
    per_asset_vol = target_vol / np.sqrt(len(have))
    w = {perp: (min(per_asset_vol / s["vol60"], PER_ASSET_CAP) if s["mom12"] > 0 else 0.0) for perp, s in have.items()}
    gross = sum(w.values())
    if gross > GROSS_CAP:
        w = {k: v * GROSS_CAP / gross for k, v in w.items()}
    return {k: float(v) for k, v in w.items()}

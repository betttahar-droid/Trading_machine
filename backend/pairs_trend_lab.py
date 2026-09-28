"""
Trend following on altcoin / Bitcoin ratios (ETH/BTC, SOL/BTC, ...): a market-neutral trend book (long the altcoin and
short Bitcoin when the ratio breaks out upward, the reverse when it breaks down), which might diversify the plan like
the TradFi book did. Rules copied from the live strategy, fixed in advance:

  signal   4h log(alt / BTC) closes: long above its 120-bar high, short below its 120-bar low; exit on the opposite
           60-bar channel or a 4 x ATR trailing stop (ATR = 20-bar mean absolute change of the ratio)
  size     each pair targets the same volatility (60-bar realised volatility of the ratio); costs 0.16% of notional
           per change (two legs, taker fee + slippage); funding ignored (difference of two perps' funding)
  test     alone, correlation with the crypto trend book, and plan + pairs book (equal risk) vs the current plan;
           in-sample 2020-06 .. 2024-06, out-of-sample 2024-07 .. now

    python -m backend.pairs_trend_lab
"""

import numpy as np
import pandas as pd

from backend.chronos_lab import closes
from backend.cross_asset_lab import BINANCE_TRADFI, FUNDING_2026, crypto_trend_returns, etf_prices, tsmom_returns
from backend.strategy_lab import START, fmt

ENTRY, EXIT, STOP_ATR, COST = 120, 60, 4.0, 0.0016


def pair_positions(lr: pd.Series) -> pd.Series:
    x = lr.to_numpy()
    hi_e = lr.rolling(ENTRY).max().shift(1).to_numpy()
    lo_e = lr.rolling(ENTRY).min().shift(1).to_numpy()
    hi_x = lr.rolling(EXIT).max().shift(1).to_numpy()
    lo_x = lr.rolling(EXIT).min().shift(1).to_numpy()
    atr = lr.diff().abs().rolling(20).mean().to_numpy()
    pos, side, stop, ext = np.zeros(len(x)), 0, np.nan, np.nan
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(hi_e[i]):
            pos[i] = side
            continue
        if side == 1:
            ext = max(ext, x[i])
            stop = max(stop, ext - STOP_ATR * atr[i])
            if x[i] < lo_x[i] or x[i] < stop:
                side = 0
        elif side == -1:
            ext = min(ext, x[i])
            stop = min(stop, ext + STOP_ATR * atr[i])
            if x[i] > hi_x[i] or x[i] > stop:
                side = 0
        if side == 0:
            if x[i] > hi_e[i]:
                side, ext, stop = 1, x[i], x[i] - STOP_ATR * atr[i]
            elif x[i] < lo_e[i]:
                side, ext, stop = -1, x[i], x[i] + STOP_ATR * atr[i]
        pos[i] = side
    return pd.Series(pos, index=lr.index)


def pairs_book() -> pd.Series:
    c = closes()
    alts = [s for s in c.columns if s != "BTCUSDT"]
    rets = {}
    for a in alts:
        lr = np.log(c[a] / c["BTCUSDT"]).dropna()
        pos = pair_positions(lr).shift(1).fillna(0)               # decided at the close, earns from the next bar
        vol = lr.diff().rolling(60, min_periods=30).std().shift(1)
        w = (0.01 / vol).clip(upper=5.0).fillna(0)                 # equal volatility per pair (1% per 4h bar target)
        held = pos * w
        pnl = held * lr.diff().fillna(0) - (held.diff().abs().fillna(0)) * COST
        rets[a] = pnl
    book = pd.DataFrame(rets).fillna(0).mean(axis=1)
    daily = (1 + book).resample("1D").prod() - 1
    return daily


def main():
    pairs = pairs_book()
    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    pb = pairs.reindex(days).fillna(0.0)
    tb = tsmom_returns(etf_prices("2005-01-01")[list(BINANCE_TRADFI)], FUNDING_2026).reindex(days).fillna(0.0)
    pb_scaled = pb * cd[START:"2024-06-30"].std() / pb[START:"2024-06-30"].std()
    print("In-sample 2020-06 .. 2024-06 | out-of-sample 2024-07 .. now")
    print(fmt("pairs trend book alone (scaled)", pb_scaled))
    print(f"correlation with the crypto trend book: in {pb[:'2024-06-30'].corr(cd[:'2024-06-30']):+.2f} | out {pb['2024-07-01':].corr(cd['2024-07-01':]):+.2f}; "
          f"with TradFi {pb.corr(tb):+.2f}")
    for name, cols in (("current plan (crypto + TradFi)", {"c": cd, "t": tb}),
                       ("plan + pairs book", {"c": cd, "t": tb, "p": pb})):
        both = pd.DataFrame(cols)
        vol_is = both[START:"2024-06-30"].std()
        w = (1 / vol_is) / (1 / vol_is).sum()
        combo = (both * w).sum(axis=1)
        combo *= cd[START:"2024-06-30"].std() / combo[START:"2024-06-30"].std()
        print(fmt(name, combo))


if __name__ == "__main__":
    main()

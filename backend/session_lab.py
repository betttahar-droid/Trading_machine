"""
Crypto by trading session: is there a stable time-of-day pattern (like the stock market's overnight effect)?

Six 4-hour blocks in UTC (00-04 and 04-08 Asia, 08-12 Europe, 12-16 Europe / US open, 16-20 US, 20-24 US close /
evening), from Binance 4h perp closes of the trend strategy's 8 coins, 2020-01 .. now. Fixed in advance:
  a block counts only if its mean return is significant (|t| >= 2) in 2020-2023 AND has the same sign in 2024+;
  trading it = holding only in that block (one round trip a day: 0.04% with limit orders, 0.13% taker + slippage).

    python -m backend.session_lab
"""

import numpy as np
import pandas as pd

from backend.chronos_lab import closes

SPLIT = "2024-01-01"
BLOCKS = {0: "00-04 Asia", 4: "04-08 Asia", 8: "08-12 Europe", 12: "12-16 Eur/US open", 16: "16-20 US", 20: "20-24 US close"}


def t(x: pd.Series) -> str:
    x = x.dropna()
    return f"{10000 * x.mean():+6.1f} bp (t {x.mean() / x.std() * np.sqrt(len(x)):+.2f})"


def main():
    c = closes()                                   # index = 4h bar open time (UTC)
    r = c.pct_change(fill_method=None)
    hour = r.index.hour
    print("mean return per 4h block (bar open time, UTC); before 2024 | 2024+")
    for sym in ["BTCUSDT", "ETHUSDT"] + [s for s in c.columns if s not in ("BTCUSDT", "ETHUSDT")]:
        line = []
        for h, name in BLOCKS.items():
            x = r[sym][hour == h]
            line.append(f"{name.split()[0]} {t(x[:SPLIT])} | {t(x[SPLIT:])}")
        print(f"\n  {sym}")
        for ln in line:
            print(f"    {ln}")
    # pooled across the 8 coins (equal weight)
    ew = r.mean(axis=1)
    print("\n  8-coin average")
    passed = []
    for h, name in BLOCKS.items():
        a, b = ew[hour == h][:SPLIT], ew[hour == h][SPLIT:]
        ta = a.mean() / a.std() * np.sqrt(len(a))
        ok = abs(ta) >= 2 and np.sign(a.mean()) == np.sign(b.mean())
        passed += [h] if ok else []
        print(f"    {name:18s} {t(a)} | {t(b)} {'<- passes' if ok else ''}")
    for h in passed:
        x = ew[hour == h]
        sgn = np.sign(x[:SPLIT].mean())
        for lab, cost in (("limit orders", 0.0004), ("taker", 0.0013)):
            net = sgn * x[SPLIT:] - cost
            print(f"    trade block {BLOCKS[h]} ({'long' if sgn > 0 else 'short'}) 2024+, {lab}: "
                  f"{(1 + net).prod() ** (365 / (len(net))) - 1:+.1%}/yr, Sharpe {net.mean() / net.std() * np.sqrt(365):+.2f}")
    # BTC day vs night split like the stock market's overnight effect: US hours (12-20) vs the rest
    btc = r["BTCUSDT"]
    us = btc[(hour == 12) | (hour == 16)]
    rest = btc[~((hour == 12) | (hour == 16))]
    for lab, sl in (("before 2024", slice(None, SPLIT)), ("2024+", slice(SPLIT, None))):
        print(f"\n  BTC {lab}: US hours (12-20 UTC) {t(us[sl])} per 4h bar | other hours {t(rest[sl])}")


if __name__ == "__main__":
    main()

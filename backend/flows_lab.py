"""
Mechanical flows in Bitcoin (1-minute Binance data, 2022-02 .. now): trades someone must do at a known time,
whatever the price. Two ideas, rules fixed in advance:

1  Leveraged-ETF rebalancing at the US close. 2x Bitcoin ETFs (and 2x MSTR/COIN ETFs, whose issuers hedge in related
   instruments) must buy after up days and sell after down days near 16:00 New York time, in proportion to the day's
   move. Their assets grew from ~0 (2022) to billions (2024+), so the effect should appear and grow.
     move      BTC return from 16:00 ET the previous weekday to 15:00 ET (what the funds must hedge)
     close     BTC return 15:00 -> 16:00 ET; after = 16:00 -> 17:00 ET
     test      slope of `close` on `move` per period; the trade = go with `move` at 15:00 when |move| > 1%, exit 16:00
     pass      slope > 0 with t >= 2 in 2024+, larger than in 2022-23, and the trade > 5 bp (maker costs ~4 bp)

2  Stop-loss cascades beyond round numbers (Osler 2003, currency markets): stop orders cluster just past round prices,
   so crossing a round level should run further than crossing an arbitrary level.
     events    a minute close that crosses a multiple of $1,000 (no cross of that level in the prior 60 minutes);
               placebo = the same for levels at $1,000 * k + $370 (arbitrary)
     measure   return over the next 5 / 15 / 60 minutes in the direction of the cross
     pass      round-level continuation above placebo with t >= 2 in both 2022-23 and 2024+

    python -m backend.flows_lab
"""

import numpy as np
import pandas as pd

from backend.news_llm_lab import load_prices


def us_close_test(px: pd.Series):
    et = px.tz_localize("UTC").tz_convert("America/New_York")
    at = lambda hh, mm: et[(et.index.hour == hh) & (et.index.minute == mm)]
    p15, p16, p17 = at(15, 0), at(16, 0), at(17, 0)
    df = pd.DataFrame({"p15": p15.set_axis(p15.index.date), "p16": p16.set_axis(p16.index.date),
                       "p17": p17.set_axis(p17.index.date)})
    df.index = pd.to_datetime(df.index)
    df = df[df.index.dayofweek < 5]
    df["prev16"] = df["p16"].shift(1)
    df["move"] = df["p15"] / df["prev16"] - 1
    df["close"] = df["p16"] / df["p15"] - 1
    df["after"] = df["p17"] / df["p16"] - 1
    df = df.dropna()
    print("1) US-close rebalancing: BTC 15:00-16:00 ET vs the move since the previous 16:00 ET")
    for lab, g in (("2022-23", df[:"2023"]), ("2024+  ", df["2024":])):
        for col in ("close", "after"):
            b = np.polyfit(g["move"], g[col], 1)[0]
            resid = g[col] - b * g["move"]
            se = resid.std() / (g["move"].std() * np.sqrt(len(g)))
            print(f"   {lab} slope of {col:5s} on move {b:+.4f} (t {b / se:+.1f}, n {len(g)})")
        big = g[g["move"].abs() > 0.01]
        trade = np.sign(big["move"]) * big["close"]
        print(f"   {lab} trade (|move| > 1%): {1e4 * trade.mean():+.1f} bp per day (t {trade.mean() / trade.std() * np.sqrt(len(trade)):+.1f}, "
              f"n {len(trade)}, win {(trade > 0).mean():.0%})")
    for y, g in df.groupby(df.index.year):
        b = np.polyfit(g["move"], g["close"], 1)[0]
        print(f"   {y}: slope {b:+.4f}")


def round_level_test(px: pd.Series):
    p = px.to_numpy()
    idx = px.index
    print("\n2) Crossing round $1,000 levels vs arbitrary levels ($1,000k + $370)")
    results = {}
    for name, offset in (("round", 0.0), ("placebo", 370.0)):
        lv = np.floor((p - offset) / 1000.0)                    # level bucket below the price
        cross = np.flatnonzero(np.diff(lv) != 0) + 1              # minute i closes in a new bucket
        last_cross = {}
        rows = []
        for i in cross:
            level = (max(lv[i], lv[i - 1])) * 1000.0 + offset       # the level that was crossed
            if i - last_cross.get(level, -10**9) < 60:
                last_cross[level] = i
                continue
            last_cross[level] = i
            d = 1.0 if lv[i] > lv[i - 1] else -1.0
            if i + 60 >= len(p):
                continue
            rows.append({"t": idx[i], **{f"r{h}": d * (p[i + h] / p[i] - 1) for h in (5, 15, 60)}})
        results[name] = pd.DataFrame(rows).set_index("t")
    for lab, lo, hi in (("2022-23", "2022", "2023"), ("2024+  ", "2024", "2100")):
        cells = []
        for h in (5, 15, 60):
            a, b = results["round"].loc[lo:hi, f"r{h}"], results["placebo"].loc[lo:hi, f"r{h}"]
            d = a.mean() - b.mean()
            se = np.sqrt(a.var() / len(a) + b.var() / len(b))
            cells.append(f"{h}m round {1e4 * a.mean():+.1f} vs placebo {1e4 * b.mean():+.1f} bp (diff t {d / se:+.1f})")
        n = (len(results["round"].loc[lo:hi]), len(results["placebo"].loc[lo:hi]))
        print(f"   {lab} n {n}: " + " | ".join(cells))


def main():
    t0, arr = load_prices()
    px = pd.Series(arr, index=pd.to_datetime((t0 + np.arange(len(arr))) * 60, unit="s"))
    us_close_test(px)
    round_level_test(px)


if __name__ == "__main__":
    main()

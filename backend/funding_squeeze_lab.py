"""
When Binance has to step in: a perp's funding settlement interval is shortened (8h -> 4h, 4h -> 1h, ...) when its
funding rate hits the cap, i.e. when one side is extremely crowded. The smart-money finding says the crowd tends to be
wrong; here the exchange itself flags the crowd. Events come from the funding history files (data.binance.vision
fundingRate, which record the interval of every settlement). Fixed in advance:

  event     the first settlement with a shorter interval than the one before, on a perp that had kept the longer
            interval for 30+ days (new listings that simply start at 4h do not count)
  crowd     the sign of the mean funding rate over the 24 hours before the event (positive = longs pay = longs crowded)
  trade     at the next UTC daily close, take the side opposite the crowd; hold 7 days (1 and 3 also shown);
            minus the equal-weight return of the point-in-time top-100 perps; 0.3% round-trip costs; funding
            received / paid on the position included
  universe  perps in the point-in-time top 150 by volume at the event
  pass      mean > 0 with t >= 2 (events on the same day averaged first) and positive in both halves of the sample

    python -m backend.funding_squeeze_lab
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from backend.growth_study import _bulk_csv, _rows
from backend.universe_data import daily_panel, top_by_volume

COST, HOLDS = 0.003, (1, 3, 7)


def funding_history(sym: str, months: list) -> pd.DataFrame:
    rows = [r for m in months for r in _rows(_bulk_csv("funding", sym, m, "4h"))]
    if not rows:
        return pd.DataFrame(columns=["iv", "rate"])
    df = pd.DataFrame({"t": [int(r[0]) for r in rows], "iv": [float(r[1]) if len(r) > 2 else 8.0 for r in rows],
                       "rate": [float(r[-1]) for r in rows]})
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.drop_duplicates("t").set_index("t").sort_index()


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    close = panel["close"]
    top150 = top_by_volume(panel["qvol"], close, 150)
    top100 = top_by_volume(panel["qvol"], close, 100)
    market = close.pct_change(fill_method=None).where(top100.shift(1).fillna(False)).mean(axis=1)
    syms = sorted(top150.columns[top150.loc["2021-06":].any()])
    months = [str(m) for m in pd.period_range("2021-06", last_month, freq="M")]
    print(f"{len(syms)} perps were in the top 150 since 2021-06; reading funding histories ...", flush=True)
    with ThreadPoolExecutor(16) as ex:
        hist = dict(zip(syms, ex.map(lambda s: funding_history(s, months), syms)))
    events = []
    for s, h in hist.items():
        if len(h) < 10:
            continue
        iv = h["iv"].to_numpy()
        since = h.index[0]
        for k in range(1, len(h)):
            if iv[k] != iv[k - 1]:
                if iv[k] < iv[k - 1] and (h.index[k] - since).days >= 30:
                    t = h.index[k]
                    pre = h["rate"][(h.index > t - pd.Timedelta(hours=24)) & (h.index <= t)]
                    events.append({"sym": s, "t": t, "from": iv[k - 1], "to": iv[k], "crowd": np.sign(pre.mean()),
                                   "funding_24h": pre.sum()})
                since = h.index[k]
    ev = pd.DataFrame(events).sort_values("t")
    print(f"{len(ev)} interval cuts ({(ev['crowd'] > 0).mean():.0%} with longs crowded); "
          f"transitions: {ev.groupby(['from', 'to']).size().to_dict()}")
    rows = []
    for e in ev.itertuples():
        d = e.t.normalize()                                  # the event day; enter at its close
        if d not in close.index or not bool(top150.loc[d].get(e.sym, False)) or e.crowd == 0:
            continue
        i = close.index.get_loc(d)
        px = close[e.sym]
        side = -e.crowd
        rec = {"t": e.t, "sym": e.sym, "crowd": e.crowd}
        f = hist[e.sym]["rate"]
        for hd in HOLDS:
            path = px.iloc[i:i + hd + 1].dropna()
            if len(path) < 2:
                continue
            raw = path.iloc[-1] / path.iloc[0] - 1
            mkt = (1 + market.iloc[i + 1:i + 1 + len(path) - 1]).prod() - 1
            fund = f[(f.index > d + pd.Timedelta(days=1)) & (f.index <= path.index[-1] + pd.Timedelta(days=1))].sum()
            rec[f"r{hd}"] = side * (raw - mkt) - COST - side * fund       # longs pay positive funding
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["r7"])
    print(f"{len(df)} tradable events (in the top 150 at the time)\n")

    def show(g, lab):
        cells = []
        for hd in HOLDS:
            per_day = g.groupby(g["t"].dt.normalize())[f"r{hd}"].mean()
            t = per_day.mean() / per_day.std() * np.sqrt(len(per_day)) if len(per_day) > 2 else np.nan
            cells.append(f"{hd}d {100 * per_day.mean():+.1f}% (t {t:+.1f})")
        print(f"  {lab:34s} n {len(g):4d}: " + " | ".join(cells))
    half = df["t"].iloc[len(df) // 2]
    show(df, "all, against the crowd")
    show(df[df["t"] < half], f"first half (to {half:%Y-%m})")
    show(df[df["t"] >= half], "second half")
    show(df[df["crowd"] > 0], "longs crowded -> short")
    show(df[df["crowd"] < 0], "shorts crowded -> long")


if __name__ == "__main__":
    main()

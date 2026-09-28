"""
Short sellers as the smart money in US stocks. FINRA publishes, for every stock and trading day, how much of the
off-exchange volume was short sales (Reg SHO daily files, free, 2018-08 .. now). Diether, Lee & Werner (2009): rising
short selling predicts lower returns over the next days/weeks. Fixed in advance:

  universe  today's S&P 500 + Nasdaq-100 members (a survivorship bias for the level of returns; the comparisons
            below are within the same set)
  signal    abnormal short share = 20-day mean of ShortVolume / TotalVolume minus its mean over the prior 250 days
  book      every Friday: long the fifth of stocks with the lowest abnormal short share, short the highest fifth,
            equal weight, held one week; 0.05% per side (large caps)
  long-only the lowest fifth vs the equal-weight universe (what a small account could do with a broker)
  periods   2019-2022 vs 2023-now; pass = long/short Sharpe > 0.5 in both, and long-only excess Sharpe > 0.3 in both

    python -m backend.short_volume_lab
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "short_volume"))
UA = {"User-Agent": "Mozilla/5.0 (research script; betttahar@gmail.com)"}
COST = 0.0005


def universe() -> list:
    path = os.path.join(DATA, "universe.txt")
    if os.path.exists(path):
        return open(path).read().split()
    tick = set()
    for url, col in (("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
                     ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker")):
        html = requests.get(url, headers=UA, timeout=60).text
        for t in pd.read_html(io.StringIO(html)):
            if col in t.columns:
                tick |= set(t[col].astype(str).str.replace(".", "-", regex=False))
                break
    open(path, "w").write("\n".join(sorted(tick)))
    return sorted(tick)


def short_panel(tickers: list) -> pd.DataFrame:
    path = os.path.join(DATA, "svr.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    days = pd.bdate_range("2018-08-01", pd.Timestamp.now().normalize() - pd.Timedelta(days=1))
    want = {t.replace("-", ".") for t in tickers} | set(tickers)

    def one(d):
        url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt"
        for k in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=30)
                if r.status_code != 200:
                    return None
                df = pd.read_csv(io.StringIO(r.text), sep="|")
                df = df[df["Symbol"].isin(want)]
                return pd.DataFrame({"d": d, "t": df["Symbol"].str.replace(".", "-", regex=False),
                                     "short": df["ShortVolume"], "total": df["TotalVolume"]})
            except Exception:                              # noqa: BLE001
                time.sleep(2 * (k + 1))
        return None
    with ThreadPoolExecutor(16) as ex:
        frames = [f for f in ex.map(one, days) if f is not None]
    raw = pd.concat(frames)
    short = raw.pivot_table(index="d", columns="t", values="short", aggfunc="sum")
    total = raw.pivot_table(index="d", columns="t", values="total", aggfunc="sum")
    out = {"short": short, "total": total}
    pd.to_pickle(out, path)
    return out


def prices(tickers: list) -> pd.DataFrame:
    import yfinance as yf
    path = os.path.join(DATA, "prices.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    frames = []
    for i in range(0, len(tickers), 50):
        d = yf.download(tickers[i:i + 50], start="2018-01-01", progress=False, auto_adjust=True, threads=4)["Close"]
        frames.append(d)
        time.sleep(3)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    px.to_pickle(path)
    return px


def main():
    os.makedirs(DATA, exist_ok=True)
    tick = universe()
    sp = short_panel(tick)
    px = prices(tick)
    print(f"{len(tick)} tickers; FINRA days {len(sp['short'])} ({sp['short'].index.min():%Y-%m-%d} .. "
          f"{sp['short'].index.max():%Y-%m-%d}); priced {px.notna().any().sum()}")
    svr = (sp["short"].rolling(20, min_periods=15).sum() / sp["total"].rolling(20, min_periods=15).sum())
    base = (sp["short"].rolling(250, min_periods=200).sum() / sp["total"].rolling(250, min_periods=200).sum()).shift(20)
    abn = (svr - base)
    wk = px.resample("W-FRI").last()
    wret = wk.pct_change(fill_method=None)
    sig = abn.resample("W-FRI").last().reindex(wret.index)
    rows = []
    w_prev = pd.Series(0.0, index=wret.columns)
    for d in wret.index[:-1]:
        s = sig.loc[d].dropna()
        s = s[s.index.isin(wret.columns) & wret.loc[d].reindex(s.index).notna()]
        nxt = wret.shift(-1).loc[d]
        if len(s) < 100:
            continue
        q = len(s) // 5
        r = s.sort_values()
        low, high = r.index[:q], r.index[-q:]
        w = pd.Series(0.0, index=wret.columns)
        w[low], w[high] = 0.5 / q, -0.5 / q
        turn = (w - w_prev).abs().sum()
        w_prev = w
        rows.append({"d": d, "ls": (w * nxt.fillna(0)).sum() - turn * COST,
                     "long_ex": nxt[low].mean() - nxt[s.index].mean(), "univ": nxt[s.index].mean()})
    df = pd.DataFrame(rows).set_index("d")
    for col, lab in (("ls", "long low / short high (half each)"), ("long_ex", "long-only lowest fifth minus universe")):
        cells = []
        for per, x in (("2019-22", df[col][:"2022"]), ("2023+", df[col]["2023":])):
            eq = (1 + x).cumprod()
            cells.append(f"{per} {eq.iloc[-1] ** (52 / len(x)) - 1:+6.1%}/yr Sharpe {x.mean() / x.std() * np.sqrt(52):+.2f}")
        print(f"  {lab:40s} " + " | ".join(cells))
    lvl = svr.resample("W-FRI").last().reindex(wret.index)
    print("  (check) level of the short share instead of its change:")
    rows = []
    for d in wret.index[:-1]:
        s = lvl.loc[d].dropna()
        s = s[s.index.isin(wret.columns) & wret.loc[d].reindex(s.index).notna()]
        if len(s) < 100:
            continue
        q = len(s) // 5
        r = s.sort_values()
        nxt = wret.shift(-1).loc[d]
        rows.append({"d": d, "ls": 0.5 * nxt[r.index[:q]].mean() - 0.5 * nxt[r.index[-q:]].mean()})
    x = pd.DataFrame(rows).set_index("d")["ls"]
    print(f"    2019-22 Sharpe {x[:'2022'].mean() / x[:'2022'].std() * np.sqrt(52):+.2f} | "
          f"2023+ Sharpe {x['2023':].mean() / x['2023':].std() * np.sqrt(52):+.2f} (before costs)")


if __name__ == "__main__":
    main()

"""
Crypto-linked stocks (COIN, MSTR, HOOD, MARA, RIOT) vs Bitcoin: does one lead the other over hours or days, and does
their price relative to Bitcoin snap back after stretching? Rules fixed in advance; design 2022-02 .. 2023-12, test
2024-01 .. now.

  daily lead  close-to-close returns at the US close (Bitcoin priced at exactly 16:00 New York from 1-minute data):
              stock today -> Bitcoin over the next 24h, and Bitcoin today -> stock tomorrow (close to close)
  hourly lead during US hours (Yahoo 60-minute bars, last ~2 years): stock hour h -> Bitcoin hour h+1 and vice versa
  snap-back   20-day z-score of log(stock / BTC); |z| > 2 -> bet on reversal for 5 trading days (long the laggard,
              short the leader), 0.2% round-trip costs

    python -m backend.linked_lab
"""

import numpy as np
import pandas as pd

from backend.news_llm_lab import load_prices

STOCKS = ["COIN", "MSTR", "HOOD", "MARA", "RIOT"]
SPLIT = "2024-01-01"


def btc_at_us_close(days) -> pd.Series:
    t0, px = load_prices()
    out = {}
    for d in days:
        t = pd.Timestamp(f"{d:%Y-%m-%d} 16:00", tz="America/New_York").tz_convert("UTC")
        i = int(t.timestamp() // 60) - t0
        if 0 <= i < len(px):
            out[d] = px[i]
    return pd.Series(out)


def t(x) -> str:
    x = pd.Series(x).dropna()
    return f"{10000 * x.mean():+6.1f} bp (t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}, n {len(x)})"


def main():
    import yfinance as yf
    d = yf.download(STOCKS, start="2022-01-15", progress=False, auto_adjust=True)["Close"].dropna(how="all")
    d = d[d.index >= "2022-02-01"]
    btc = btc_at_us_close(d.index)
    d = d.loc[btc.index]
    sr, br = d.pct_change(), btc.pct_change()
    print("DAILY LEAD (correlations; before 2024 | 2024+)")
    for s in STOCKS:
        a = pd.DataFrame({"s": sr[s], "b": br, "b_next": br.shift(-1), "s_next": sr[s].shift(-1)}).dropna()
        out = []
        for lab, part in (("<2024", a[:SPLIT]), ("2024+", a[SPLIT:])):
            out.append(f"{lab}: stock->BTC next {part.s.corr(part.b_next):+.3f}, BTC->stock next {part.b.corr(part.s_next):+.3f}, "
                       f"same day {part.s.corr(part.b):+.2f}")
        print(f"  {s:5s} " + " | ".join(out))

    print("\nHOURLY LEAD during US hours (last ~2 years, Yahoo 60m; correlation of next-hour returns)")
    h = yf.download(STOCKS, period="729d", interval="60m", progress=False, auto_adjust=True)["Close"].dropna(how="all")
    t0, px = load_prices()
    ny = h.index.tz_convert("America/New_York")
    # Yahoo stamps 60-minute bars with their START; each bar closes 60 minutes later (the last one at 16:00)
    ends = pd.DatetimeIndex([min(x + pd.Timedelta(minutes=60), x.normalize() + pd.Timedelta(hours=16)) for x in ny])
    mins = ((ends.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy() - t0
    ok = (mins >= 0) & (mins < len(px))
    h, bh = h[ok], pd.Series(px[mins[ok]], index=h.index[ok])
    hr, bhr = h.pct_change(), bh.pct_change()
    day = pd.Series(ny[ok].date, index=h.index)
    nxt_ok = day.shift(-1) == day                                 # next bar on the same trading day
    for s in STOCKS:
        a = pd.DataFrame({"s": hr[s], "b": bhr, "b_next": bhr.shift(-1), "s_next": hr[s].shift(-1)})[nxt_ok].dropna()
        print(f"  {s:5s} stock->BTC next hour {a.s.corr(a.b_next):+.3f} | BTC->stock next hour {a.b.corr(a.s_next):+.3f} "
              f"| same hour {a.s.corr(a.b):+.2f} (n {len(a)})")

    print("\nSNAP-BACK of the stock / BTC ratio (|20-day z| > 2, hold 5 days, 0.2% costs)")
    for s in STOCKS:
        lr = np.log(d[s] / btc)
        z = (lr - lr.rolling(20).mean()) / lr.rolling(20).std()
        fwd = lr.shift(-5) - lr                                   # stock vs BTC over the next 5 days
        sig = np.where(z > 2, -1, np.where(z < -2, 1, 0))
        pnl = pd.Series(sig * fwd, index=lr.index)[sig != 0] - 0.002
        # one position at a time: keep every 5th signal day at most
        keep, last = [], None
        for day in pnl.index:
            if last is None or (day - last).days >= 7:
                keep.append(day)
                last = day
        pnl = pnl.loc[keep]
        print(f"  {s:5s} before 2024 {t(pnl[:SPLIT])} | 2024+ {t(pnl[SPLIT:])}")


if __name__ == "__main__":
    main()

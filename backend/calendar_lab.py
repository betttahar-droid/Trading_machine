"""
Calendar effects: the oldest documented market patterns, checked on 2022 .. now after being defined on older data.

  US stock index (SPY from 1993, QQQ from 1999; Yahoo daily, dividend-adjusted)
    overnight      close -> next open vs open -> close (Cliff, Cooper & Gulen 2008; Kelly & Clark 2011)
    turn of month  last trading day + first 3 of the month (Lakonishok & Smidt 1988; McConnell & Xu 2008)
    pre-holiday    the trading day before a market holiday (Ariel 1990)
    pre-Fed        the close before an FOMC announcement to the close of that day (Lucca & Moench 2015)
  Bitcoin (Binance 4h / 1m)
    turn of month, weekend vs weekday, Deribit monthly expiry (last Friday 08:00 UTC: 3 days before vs after),
    funding settlements (00/08/16 UTC: 30 minutes before vs after, by the sign of the funding rate)

Pass: the 2022+ effect has the same sign as before 2022 and t >= 2, and the timed strategy beats simply holding.

    python -m backend.calendar_lab
"""

import io
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

SPLIT = "2022-01-01"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def fomc_dates() -> pd.DatetimeIndex:
    """Announcement days of scheduled FOMC meetings, 1994 .. now (federalreserve.gov calendars)."""
    ua = {"User-Agent": "Mozilla/5.0 (research script)"}
    days = []
    cur = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", headers=ua, timeout=30).text
    for y, body in zip(*[iter(re.split(r"(\d{4}) FOMC Meetings", cur)[1:])] * 2):
        for mon, d in re.findall(r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<", body, re.S):
            if "notation" in d.lower() or "unscheduled" in d.lower():
                continue
            last_mon = mon.split("/")[-1].strip()
            last_day = re.findall(r"\d+", d)[-1]
            days.append(pd.Timestamp(f"{last_mon} {last_day} {y}"))
    for y in range(1994, 2021):
        t = requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", headers=ua, timeout=30).text
        for mon, d in re.findall(rf"<h5[^>]*>\s*((?:{MONTHS})(?:/(?:{MONTHS}))?)\s+([\d\-]+)\s+Meeting", t):
            days.append(pd.Timestamp(f"{mon.split('/')[-1]} {d.split('-')[-1]} {y}"))
        time.sleep(0.2)
    return pd.DatetimeIndex(sorted(set(days)))


def tstat(x: pd.Series) -> str:
    x = x.dropna()
    return f"{10000 * x.mean():+6.1f} bp/day (t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}, n {len(x)})"


def diff(a: pd.Series, b: pd.Series) -> str:
    a, b = a.dropna(), b.dropna()
    d = a.mean() - b.mean()
    return f"{10000 * d:+6.1f} bp (t {d / np.sqrt(a.var() / len(a) + b.var() / len(b)):+.2f})"


def sharpe(r: pd.Series, per_year=252) -> str:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return (f"{eq.iloc[-1] ** (per_year / len(r)) - 1:+.1%}/yr, Sharpe {r.mean() / r.std() * np.sqrt(per_year):+.2f}, "
            f"max DD {(eq / eq.cummax() - 1).min():+.0%}")


def equities():
    import yfinance as yf
    fomc = fomc_dates()
    print(f"{len(fomc)} FOMC announcement days {fomc[0]:%Y-%m-%d} .. {fomc[-1]:%Y-%m-%d}")
    for sym in ("SPY", "QQQ"):
        d = yf.download(sym, start="1993-01-01", progress=False, auto_adjust=True)
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        o, c = d["Open"], d["Close"]
        cc = c.pct_change()
        night = o / c.shift(1) - 1
        day = c / o - 1
        idx = d.index
        month = idx.to_period("M")
        pos_in_month = pd.Series(1, index=idx).groupby(month).cumcount()
        from_end = pd.Series(1, index=idx).groupby(month).cumcount(ascending=False)
        tom = (pos_in_month < 3) | (from_end == 0)
        nxt = pd.Series(idx, index=idx).shift(-1)
        pre_hol = (nxt - pd.Series(idx, index=idx)).dt.days > np.where(idx.dayofweek == 4, 3, 1)
        fomc_day = pd.Series(idx.isin(fomc), index=idx)
        print(f"\n=== {sym} {idx[0]:%Y} .. now ===")
        for lab, m in (("before 2022", idx < SPLIT), ("2022+     ", idx >= SPLIT)):
            print(f"  {lab} hold: {sharpe(cc[m])}")
            print(f"    overnight {tstat(night[m])} {sharpe(night[m])} | daytime {tstat(day[m])} {sharpe(day[m])}")
            print(f"    overnight break-even cost per round trip: {10000 * night[m].mean():.1f} bp")
            print(f"    turn of month vs other days: {diff(cc[m & tom], cc[m & ~tom])}; hold only then: {sharpe(cc.where(tom, 0)[m])}")
            print(f"    pre-holiday vs other days: {diff(cc[m & pre_hol], cc[m & ~pre_hol])} (n {int((m & pre_hol).sum())})")
            print(f"    FOMC announcement day vs other days: {diff(cc[m & fomc_day], cc[m & ~fomc_day])} (n {int((m & fomc_day).sum())})")


def bitcoin():
    from backend.chronos_lab import closes
    from backend.news_llm_lab import load_prices
    btc = closes()["BTCUSDT"].dropna()
    daily = btc.resample("1D").last().pct_change().dropna()
    idx = daily.index                                   # return of day d = close(d 24:00) / close(d-1 24:00)
    tom = (idx.day <= 3) | (idx.is_month_end)
    weekend = idx.dayofweek >= 5
    last_fri = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform(
        lambda s: s[s.dt.dayofweek == 4].max())
    rel = (pd.Series(idx, index=idx) - last_fri).dt.days
    print("\n=== Bitcoin (daily, UTC) ===")
    for lab, m in (("before 2022", idx < SPLIT), ("2022+     ", idx >= SPLIT)):
        print(f"  {lab} hold: {sharpe(daily[m], 365)}")
        print(f"    turn of month vs other days: {diff(daily[m & tom], daily[m & ~tom])}")
        print(f"    weekend vs weekday: {diff(daily[m & weekend], daily[m & ~weekend])}")
        print(f"    3 days before monthly expiry vs 3 days after: {diff(daily[m & rel.between(-3, -1).to_numpy()], daily[m & rel.between(1, 3).to_numpy()])}")
    # funding settlements on 1-minute data (2022-02 .. now)
    t0, px = load_prices()
    minutes = t0 + np.arange(len(px))
    ts = pd.to_datetime(minutes * 60, unit="s")
    setl = np.flatnonzero((ts.minute == 0) & (ts.hour % 8 == 0))
    setl = setl[(setl > 30) & (setl + 30 < len(px))]
    before = px[setl] / px[setl - 30] - 1
    after = px[setl + 30] / px[setl] - 1
    f = []
    for p in pd.period_range("2022-01", time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86400)), freq="M"):
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{p}.zip"
        try:
            z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=30).content))
            f.append(pd.read_csv(z.open(z.namelist()[0])))
        except Exception:
            continue
    f = pd.concat(f)
    tcol = [c for c in f.columns if "time" in c.lower()][0]
    rcol = [c for c in f.columns if "rate" in c.lower() and "funding" in c.lower()][0]
    fr = pd.Series(f[rcol].to_numpy(), index=pd.to_datetime(f[tcol], unit="ms").dt.round("min"))
    fr = fr[~fr.index.duplicated()]
    rate = fr.reindex(ts[setl]).to_numpy()
    s = pd.DataFrame({"t": ts[setl], "before": before, "after": after, "rate": rate}).dropna()
    hi = s["rate"] > s["rate"].quantile(0.8)
    lo = s["rate"] < s["rate"].quantile(0.2)
    print("\n  funding settlements (30 min before / after), BTCUSDT 2022-02 .. now")
    for lab, m in (("2022-23", s.t < "2024-01-01"), ("2024+  ", s.t >= "2024-01-01")):
        g = s[m]
        print(f"    {lab} high funding: before {tstat(g.before[hi[m]])}, after {tstat(g.after[hi[m]])} | "
              f"low funding: before {tstat(g.before[lo[m]])}, after {tstat(g.after[lo[m]])}")


def main():
    equities()
    bitcoin()


if __name__ == "__main__":
    main()

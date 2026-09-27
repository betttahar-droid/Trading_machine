"""
Binance stock / index perpetuals trade 24/7, the underlying shares only 09:30-16:00 New York time. Between the close and
the next open (nights and weekends) the perp price is set by crypto traders alone. Do those closed-hours moves
overshoot and reverse once the real market opens?

Data: Binance USDT perp 5-minute klines (data.binance.vision) for the index and single-stock perps listed in 2026;
US trading days from Yahoo's SPY calendar; New York time with daylight saving.

  closed move   perp price at 09:30 NY / perp price at the previous trading day's 16:00 NY - 1
  after open    perp return 09:30 -> 10:30, -> 11:30 and -> 16:00
  real gap      the stock / ETF's own open / previous close - 1 (Yahoo): how well the perp anticipated it

Rule fixed in advance: fade closed moves larger than 1 standard deviation (of that symbol's closed moves up to the
day before), hold until 11:30 NY, 0.1% round-trip costs.

    python -m backend.tradfi_hours_lab
"""

import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

SYMS = ["SPYUSDT", "QQQUSDT", "IWMUSDT", "TSLAUSDT", "NVDAUSDT", "AAPLUSDT", "MSFTUSDT", "AMZNUSDT", "GOOGLUSDT",
        "METAUSDT", "AMDUSDT", "NFLXUSDT", "PLTRUSDT", "COINUSDT", "MSTRUSDT", "HOODUSDT", "CRCLUSDT"]
BASE = "https://data.binance.vision/data/futures/um"
COST = 0.001


def klines_5m(sym: str) -> pd.Series:
    rows = []
    for p in pd.period_range("2026-01", time.strftime("%Y-%m"), freq="M"):
        r = requests.get(f"{BASE}/monthly/klines/{sym}/5m/{sym}-5m-{p}.zip", timeout=60)
        files = []
        if r.status_code == 200:
            files = [r.content]
        else:                                       # current month: daily files
            for d in pd.date_range(p.start_time, min(p.end_time, pd.Timestamp.now().normalize() - pd.Timedelta(days=1))):
                rd = requests.get(f"{BASE}/daily/klines/{sym}/5m/{sym}-5m-{d:%Y-%m-%d}.zip", timeout=60)
                if rd.status_code == 200:
                    files.append(rd.content)
        for content in files:
            z = zipfile.ZipFile(io.BytesIO(content))
            for line in z.read(z.namelist()[0]).decode().splitlines():
                if line[:1].isdigit():
                    x = line.split(",")
                    rows.append((int(x[0]), float(x[1])))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    s.index = pd.to_datetime(s.index, unit="ms", utc=True).tz_convert("America/New_York")
    return s[~s.index.duplicated()]


def main():
    import yfinance as yf
    spy = yf.download("SPY", start="2025-12-01", progress=False, auto_adjust=True)
    tdays = [d.date() for d in spy.index]
    under = {"SPYUSDT": "SPY", "QQQUSDT": "QQQ", "IWMUSDT": "IWM"}
    stocks = yf.download([under.get(s, s[:-4]) for s in SYMS], start="2025-12-01", progress=False, auto_adjust=False)
    rows = []
    for sym in SYMS:
        px = klines_5m(sym)
        if px.empty:
            print(f"  {sym}: no data")
            continue
        tick = under.get(sym, sym[:-4])

        def at(day, hh, mm):
            t = pd.Timestamp(f"{day} {hh:02d}:{mm:02d}", tz="America/New_York")
            return px.get(t, np.nan)
        for d0, d in zip(tdays[:-1], tdays[1:]):
            c0, o = at(d0, 16, 0), at(d, 9, 30)
            if not (np.isfinite(c0) and np.isfinite(o)):
                continue
            try:
                gap = stocks["Open"][tick].loc[str(d)] / stocks["Close"][tick].loc[str(d0)] - 1
            except KeyError:
                gap = np.nan
            rows.append({"sym": sym, "day": pd.Timestamp(d), "weekend": (d - d0).days > 1, "closed": o / c0 - 1,
                         "h1": at(d, 10, 30) / o - 1, "h2": at(d, 11, 30) / o - 1, "day_ret": at(d, 16, 0) / o - 1,
                         "e935_h2": at(d, 11, 30) / at(d, 9, 35) - 1, "e940_h2": at(d, 11, 30) / at(d, 9, 40) - 1,
                         "e940_close": at(d, 15, 55) / at(d, 9, 40) - 1, "real_gap": float(gap)})
        print(f"  {sym}: {px.index[0]:%Y-%m-%d} .. {px.index[-1]:%Y-%m-%d}", flush=True)
    df = pd.DataFrame(rows).sort_values(["sym", "day"])
    df["sd"] = df.groupby("sym")["closed"].transform(lambda x: x.expanding(10).std().shift(1))
    big = df["closed"].abs() > df["sd"]

    def t(x):
        x = x.dropna()
        return f"{100 * x.mean():+.3f}% (t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}, n {len(x)})"
    print(f"\n{len(df)} symbol-days")
    print(f"  perp closed move vs the real opening gap: correlation {df[['closed', 'real_gap']].corr().iloc[0, 1]:+.2f}, "
          f"mean perp-minus-real {100 * (df.closed - df.real_gap).mean():+.3f}% (the perp's error at the open)")
    for name, h in (("first hour", "h1"), ("first two hours", "h2"), ("rest of day", "day_ret")):
        print(f"  {name}: Spearman with the closed move {df['closed'].corr(df[h], method='spearman'):+.3f}")
    fade = -np.sign(df["closed"]) * df["h2"] - COST
    for lab, m in (("all", big), ("weekends", big & df.weekend), ("weekday nights", big & ~df.weekend),
                   ("index perps", big & df.sym.isin(["SPYUSDT", "QQQUSDT", "IWMUSDT"])),
                   ("single stocks", big & ~df.sym.isin(["SPYUSDT", "QQQUSDT", "IWMUSDT"]))):
        print(f"  RULE fade >1 sd closed moves, hold to 11:30, {lab}: {t(fade[m])}")
    first, second = df.day < df.day.quantile(0.5), df.day >= df.day.quantile(0.5)
    print(f"  by half: first {t(fade[big & first])} | second {t(fade[big & second])}")
    # follow-up (found after the fade rule failed): follow big closed moves in single stocks
    single = big & ~df.sym.isin(["SPYUSDT", "QQQUSDT", "IWMUSDT"])
    d = df[single].copy()
    sgn = np.sign(d["closed"])
    for col, lab, cost in (("h2", "enter 09:30, exit 11:30, 0.1%", 0.001), ("e935_h2", "enter 09:35, exit 11:30, 0.2%", 0.002),
                           ("e940_h2", "enter 09:40, exit 11:30, 0.2%", 0.002), ("e940_close", "enter 09:40, exit 15:55, 0.2%", 0.002)):
        net = sgn * d[col] - cost
        per_day = net.groupby(d["day"]).mean()
        print(f"  FOLLOW single stocks, {lab}: per trade {t(net)} | per day (clustered) {t(per_day)}")
    net = sgn * d["e940_h2"] - 0.002
    print("  by month (enter 09:40): " + ", ".join(f"{m}: {100 * x.mean():+.2f}% (n {len(x)})" for m, x in net.groupby(d.day.dt.to_period('M'))))
    print(f"  up-gaps {t(net[sgn > 0])} | down-gaps {t(net[sgn < 0])}")


if __name__ == "__main__":
    main()

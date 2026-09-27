"""
Alternative ideas from the literature, checked on data after they were published or tuned. 0.15% round-trip costs
(taker); the seasonal trades are also shown at 0.05% (limit orders on both sides).

  1 volatility targeting   scale exposure by (target / last 30 days' volatility of the 8-coin market), capped 0.25-2x;
                           target = the 2020-06 .. 2024-06 median. Applied to the live crypto trend book and to the
                           whole plan (crypto + TradFi with funding). Zakamulin and others find most of trend
                           following's benefit comes from this.
  2 Bitcoin clock effects  (a) long 22:00-24:00 UTC every day (Quantpedia; 33%/yr, Sharpe 1.58 on 2015-2021)
                           (b) long Sunday 23:00 UTC .. Monday 23:00 UTC ("Monday Asia open", Concretum 2018-2025)
                           (c) intraday momentum: the 00:00-00:30 UTC return's sign held for 23:30-24:00
  3 crash rebound          after a 4h bar falls k x its recent volatility, buy and hold H bars; k and H picked on
                           2020-2024, checked unchanged on 2025+
  4 funding carry          long spot + short perp on BTC / ETH earns the funding rate with no price exposure

    python -m backend.alt_lab
"""

import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from backend.chronos_lab import closes
from backend.news_llm_lab import load_prices

COST = 0.0015
SPLIT = "2024-07-01"            # the project's usual in-sample / out-of-sample split for the trend book


def stats(r: pd.Series, per_year: float) -> str:
    r = r.dropna()
    if len(r) < 10 or r.std() == 0:
        return "n/a"
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (per_year / len(r)) - 1
    dd = (eq / eq.cummax() - 1).min()
    return f"{cagr:+7.1%}/yr  Sharpe {r.mean() / r.std() * np.sqrt(per_year):+.2f}  max DD {dd:+.0%}"


def vol_targeting():
    from backend.cross_asset_lab import BINANCE_TRADFI, FUNDING_2026, crypto_trend_returns, etf_prices, tsmom_returns
    from backend.strategy_lab import START
    print("\n1  VOLATILITY TARGETING (daily, calendar days)")
    crypto = crypto_trend_returns()
    days = pd.date_range(crypto.index.min(), crypto.index.max(), freq="D")
    cd = crypto.reindex(days).fillna(0.0)
    px = etf_prices("2005-01-01")
    tb = tsmom_returns(px[list(BINANCE_TRADFI)], FUNDING_2026).reindex(days).fillna(0.0)
    both = pd.DataFrame({"c": cd, "t": tb})
    vol_is = both[START:"2024-06-30"].std()
    w = (1 / vol_is) / (1 / vol_is).sum()
    plan = (both * w).sum(axis=1)
    plan *= cd[START:"2024-06-30"].std() / plan[START:"2024-06-30"].std()
    c4 = closes().resample("1D").last()
    mkt = c4.pct_change(fill_method=None).mean(axis=1).reindex(days)
    mvol = mkt.rolling(30, min_periods=20).std().shift(1)          # known at the start of the day
    scale = (mvol[START:"2024-06-30"].median() / mvol).clip(0.25, 2.0).fillna(1.0)
    for name, r in (("crypto trend book", cd), ("whole plan (crypto + TradFi)", plan), ("8-coin basket, buy & hold", mkt.fillna(0))):
        for lab, x in (("as is", r), ("vol-targeted", r * scale)):
            print(f"  {name:30s} {lab:12s} 2020-06..2024-06 {stats(x[START:'2024-06-30'], 365)} | 2024-07..now {stats(x[SPLIT:], 365)}")
    print(f"  average exposure multiplier after 2024-07: {scale[SPLIT:].mean():.2f}")


def clock_effects():
    print("\n2  BITCOIN CLOCK EFFECTS (BTCUSDT perp 1-minute opens, 2022-02 .. now)")
    t0, px = load_prices()
    minutes = t0 + np.arange(len(px))
    ts = pd.to_datetime(minutes * 60, unit="s")
    s = pd.Series(px, index=ts)

    def trades(entry_mask: pd.Series, hold_min: int, direction=None) -> pd.DataFrame:
        idx = np.flatnonzero(entry_mask.to_numpy())
        idx = idx[idx + hold_min < len(px)]
        r = px[idx + hold_min] / px[idx] - 1
        d = np.ones(len(idx)) if direction is None else direction[idx]
        return pd.DataFrame({"t": ts[idx], "g": d * r}).set_index("t")

    hm = ts.hour * 60 + ts.minute
    rules = {
        "(a) long 22:00-24:00 UTC daily": trades(pd.Series(hm == 22 * 60, index=ts), 120),
        "(b) long Sun 23:00 -> Mon 23:00 UTC": trades(pd.Series((ts.dayofweek == 6) & (hm == 23 * 60), index=ts), 1440),
    }
    first = pd.Series(np.nan, index=ts)
    day_open = np.flatnonzero(hm == 0)
    day_open = day_open[day_open + 1440 < len(px)]
    sign = np.full(len(px), 0.0)
    sign[day_open + 1410] = np.sign(px[day_open + 30] / px[day_open] - 1)
    rules["(c) 00:00-00:30 sign held 23:30-24:00"] = trades(pd.Series(sign != 0, index=ts), 30, sign)
    for name, t in rules.items():
        per_year = len(t) / ((t.index[-1] - t.index[0]).days / 365.25)
        print(f"  {name}")
        for lab, part in (("2022-24", t[t.index < "2025-01-01"]), ("2025+  ", t[t.index >= "2025-01-01"])):
            g = part["g"]
            tstat = g.mean() / g.std() * np.sqrt(len(g))
            print(f"    {lab} n={len(g):4d}  gross {100 * g.mean():+.3f}%/trade (t {tstat:+.2f})  "
                  f"net: taker {stats(g - COST, per_year)} | limit orders {stats(g - 0.0005, per_year)}")
        yearly = part = t["g"].groupby(t.index.year).mean() * 100
        print("    gross %/trade by year: " + ", ".join(f"{y}: {v:+.3f}" for y, v in yearly.items()))


def crash_rebound():
    print("\n3  CRASH REBOUND (8 coins, 4h bars; k and H chosen on 2020-2024)")
    c = closes()
    r = c.pct_change(fill_method=None)
    sd = r.rolling(180, min_periods=60).std().shift(1)
    res = {}
    for k in (2.0, 3.0, 4.0):
        for h in (1, 3, 6):
            rows = []
            for s in c.columns:
                hit = np.flatnonzero((r[s] <= -k * sd[s]).to_numpy())
                last = -10 ** 9
                for i in hit:
                    if i <= last or i + h >= len(c):
                        continue
                    last = i + h                                       # no overlapping trades per coin
                    fwd = c[s].iat[i + h] / c[s].iat[i] - 1
                    if np.isfinite(fwd):
                        rows.append((c.index[i], fwd - COST))
            res[(k, h)] = pd.DataFrame(rows, columns=["t", "net"]).set_index("t").sort_index()
    ins = {kh: t[t.index < "2025-01-01"]["net"].mean() for kh, t in res.items()}
    best = max(ins, key=ins.get)
    for (k, h), t in res.items():
        a, b = t[t.index < "2025-01-01"]["net"], t[t.index >= "2025-01-01"]["net"]
        mark = "  <- chosen on 2020-24" if (k, h) == best else ""
        print(f"  drop >= {k:.0f} sd, hold {h * 4:2d}h: 2020-24 n={len(a):4d} {100 * a.mean():+.2f}%/trade (t {a.mean() / a.std() * np.sqrt(len(a)):+.2f})"
              f" | 2025+ n={len(b):3d} {100 * b.mean():+.2f}%/trade (t {b.mean() / b.std() * np.sqrt(len(b)):+.2f}){mark}")


def crash_rebound_pit(k: float = 4.0, h: int = 6, top_n: int = 30):
    """The chosen crash-rebound rule on a point-in-time universe: every USDT perp while it is in the top `top_n` by
    trailing volume (delisted coins included; a coin that stops trading is closed at its last price), trades that
    start on the same bar grouped into one event, and a harsher 0.5% round-trip cost for crash-time slippage."""
    from backend.growth_study import load_market_bulk
    from backend.universe_data import available_months, daily_panel, top_by_volume
    print(f"\n3b CRASH REBOUND, point-in-time top {top_n} (drop >= {k:.0f} sd, hold {h * 4}h)")
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    member = top_by_volume(panel["qvol"], panel["close"], top_n)
    ever = [s for s in member.columns if member[s].any()]
    m = load_market_bulk("4h", last_month, ever, lambda s: available_months(s, "4h"))
    c = pd.DataFrame({s: pd.Series({b["timestamp"]: b["close"] for b in m[s]["bars"]}) for s in m if m[s]["bars"]})
    c.index = pd.to_datetime(c.index, unit="ms")
    c = c.sort_index()
    r = c.pct_change(fill_method=None)
    sd = r.rolling(180, min_periods=60).std().shift(1)
    mem4h = member.reindex(c.index.normalize()).fillna(False)
    mem4h.index = c.index
    rows = []
    for s in c.columns:
        if s not in mem4h.columns:
            continue
        col = c[s].to_numpy()
        valid = np.flatnonzero(~np.isnan(col))
        if len(valid) == 0:
            continue
        last_valid = valid[-1]
        hit = np.flatnonzero(((r[s] <= -k * sd[s]) & mem4h[s]).to_numpy())
        busy = -1
        for i in hit:
            if i <= busy:
                continue
            busy = i + h
            j = min(i + h, last_valid)                     # delisted inside the window: its last traded price
            if j <= i or i + h >= len(c) and last_valid == len(c) - 1:
                continue
            rows.append((c.index[i], s, col[j] / col[i] - 1, j < i + h))
    t = pd.DataFrame(rows, columns=["t", "sym", "gross", "delisted"])
    t["year"] = t["t"].dt.year
    ev = t.groupby("t")["gross"].mean()                     # one number per crash bar
    for lab, part, evp in (("2020-24", t[t.t < "2025-01-01"], ev[ev.index < "2025-01-01"]),
                           ("2025+  ", t[t.t >= "2025-01-01"], ev[ev.index >= "2025-01-01"])):
        for cost in (COST, 0.005):
            e = evp - cost
            print(f"  {lab} cost {cost:.2%}: {len(part):4d} trades on {len(evp):3d} crash bars | per trade {100 * (part.gross - cost).mean():+.2f}%"
                  f" | per crash bar {100 * e.mean():+.2f}% (t {e.mean() / e.std() * np.sqrt(len(e)):+.2f}), "
                  f"{(e > 0).mean():.0%} positive")
    print("  per trade by year (0.15%): " + ", ".join(f"{y}: {100 * (g - COST).mean():+.2f}% (n {len(g)})"
                                                  for y, g in t.groupby("year")["gross"]))
    worst = t.nsmallest(5, "gross")
    print("  worst trades: " + ", ".join(f"{r.sym} {r.t:%Y-%m-%d} {100 * r.gross:+.0f}%{' (delisted)' if r.delisted else ''}"
                                         for r in worst.itertuples()))
    top_ev = ev.sort_values()
    print(f"  share of 2025+ profit from the best 3 crash bars: "
          f"{ev[ev.index >= '2025-01-01'].nlargest(3).sum() / max(ev[ev.index >= '2025-01-01'].sum(), 1e-9):.0%}")


def funding_carry():
    print("\n4  FUNDING CARRY (long spot + short perp; yearly funding received, before ~0.4% entry/exit costs)")
    for sym in ("BTCUSDT", "ETHUSDT"):
        rows = []
        for p in pd.period_range("2022-01", time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86400)), freq="M"):
            url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{p}.zip"
            try:
                z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=30).content))
                df = pd.read_csv(z.open(z.namelist()[0]))
                rows.append(df)
            except Exception:
                continue
        f = pd.concat(rows)
        tcol = [c for c in f.columns if "time" in c.lower()][0]
        rcol = [c for c in f.columns if "rate" in c.lower() and "funding" in c.lower()][0]
        f["t"] = pd.to_datetime(f[tcol], unit="ms")
        by_year = f.groupby(f["t"].dt.year)[rcol].agg(["sum", lambda x: (x < 0).mean()])
        print(f"  {sym}: " + " | ".join(f"{y}: {s:+.1%} ({neg:.0%} of payments negative)" for y, (s, neg) in by_year.iterrows()))


def main():
    vol_targeting()
    clock_effects()
    crash_rebound()
    funding_carry()


if __name__ == "__main__":
    main()

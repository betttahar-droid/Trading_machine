"""
Bitcoin around US macro releases: CPI (08:30 New York), the jobs report (Employment Situation, 08:30) and FOMC decisions
(14:00). Release dates from the BLS archive pages and the Fed calendars; Bitcoin from Binance 1-minute data
(2022-02 .. now). Fixed in advance: a drift counts only if |t| >= 2 in 2022-2023 and the same sign in 2024+.

  pre       24 hours before the release (up to 5 minutes before)
  reaction  5 minutes before to 60 minutes after
  post      1 hour to 24 hours after
  vol       |reaction-window return| vs the same clock window on ordinary weekdays

    python -m backend.macro_lab
"""

import re

import numpy as np
import pandas as pd
import requests

from backend.calendar_lab import fomc_dates
from backend.news_llm_lab import load_prices

UA = {"User-Agent": "Mozilla/5.0 (research script; betttahar@gmail.com)"}
SPLIT = "2024-01-01"


def bls_dates(page: str, prefix: str) -> list:
    t = requests.get(f"https://www.bls.gov/bls/news-release/{page}.htm", headers=UA, timeout=30).text
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)                 # drop scheduled (commented-out) future releases
    days = {pd.Timestamp(f"{m[4:]}-{m[:2]}-{m[2:4]}") for m in re.findall(rf"{prefix}_(\d{{8}})\.htm", t)}
    return sorted(d for d in days if d <= pd.Timestamp.now())


def main():
    t0, px = load_prices()

    def price(ts_ny: pd.Timestamp):
        i = int(ts_ny.tz_convert("UTC").timestamp() // 60) - t0
        return px[i] if 0 <= i < len(px) else np.nan

    events = {"CPI": (bls_dates("cpi", "cpi"), 8, 30), "jobs": (bls_dates("empsit", "empsit"), 8, 30),
              "FOMC": ([d for d in fomc_dates() if d <= pd.Timestamp.now()], 14, 0)}
    ordinary = pd.bdate_range("2022-02-02", pd.Timestamp.now().normalize() - pd.Timedelta(days=2))
    all_event_days = {d for ds, _, _ in events.values() for d in ds}

    def windows(day, hh, mm):
        t = pd.Timestamp(f"{day:%Y-%m-%d} {hh:02d}:{mm:02d}", tz="America/New_York")
        p = {k: price(t + pd.Timedelta(minutes=m)) for k, m in (("pre0", -24 * 60), ("m5", -5), ("p60", 60), ("p24", 24 * 60))}
        return {"pre": p["m5"] / p["pre0"] - 1, "reaction": p["p60"] / p["m5"] - 1, "post": p["p24"] / p["p60"] - 1}

    def t(x):
        x = pd.Series(x).dropna()
        return f"{10000 * x.mean():+6.1f} bp (t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}, n {len(x)})"
    for name, (days, hh, mm) in events.items():
        ev = pd.DataFrame({d: windows(d, hh, mm) for d in days if d >= pd.Timestamp("2022-02-02")}).T
        base = pd.DataFrame({d: windows(d, hh, mm) for d in ordinary if d not in all_event_days}).T
        print(f"\n=== {name} ({len(ev)} releases since 2022-02) ===")
        for lab, e, b in (("2022-23", ev[ev.index < SPLIT], base[base.index < SPLIT]),
                          ("2024+  ", ev[ev.index >= SPLIT], base[base.index >= SPLIT])):
            print(f"  {lab} pre {t(e.pre)} | reaction {t(e.reaction)} | post {t(e.post)}")
            print(f"          size of the reaction hour: {10000 * e.reaction.abs().mean():.0f} bp vs {10000 * b.reaction.abs().mean():.0f} bp "
                  f"on ordinary days ({e.reaction.abs().mean() / b.reaction.abs().mean():.1f}x)")


if __name__ == "__main__":
    main()

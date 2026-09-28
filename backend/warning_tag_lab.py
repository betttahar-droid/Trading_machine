"""
Binance "Monitoring Tag" additions: shorting coins the exchange flags as risky. Binance delistings were the best
event found so far (event_lab.py: short and hold 4h, +3.6% a trade), but they are rare; the Monitoring Tag is the
warning that often comes months earlier (since 2023-10, ~170 coins in 31 announcements, now almost monthly).
Events from @binance_announcements, traded on the coin's USD-M perp. Fixed in advance:

  entry    short at the UTC daily close of the announcement day (the posts come 02:00-11:00 UTC, so a slow, daily
           bot); the move from the previous close to that entry is shown as "reaction" (not tradable in full)
  exits    1, 3, 7 (primary) and 14 days later; perps that stop trading exit at their last close
  costs    0.4% round trip (thin coins), funding received / paid by the short
  adjust   minus the equal-weight return of the point-in-time top-100 perps over the same days
  test     announcements are averaged first (coins in one post share the same day), t-stat across announcements;
           it counts only if the market-adjusted 7-day mean is positive with t >= 2 and positive in both halves

    python -m backend.warning_tag_lab
"""

import re
import time

import numpy as np
import pandas as pd

from backend.telegram_data import load_channel
from backend.universe_data import daily_panel, load_funding, top_by_volume

COST = 0.004
HOLDS = (1, 3, 7, 14)


def events() -> list:
    out = []
    for m in load_channel("binance_announcements"):
        first = m["text"].split("\n")[0]
        hit = re.search(r"Monitoring Tag to (?:Include )?(.+?)(?:,? and Remove|, Remove| on \d{4}-\d{2}-\d{2}|$)", first)
        if not hit or "Extend" not in first:
            continue
        for t in re.split(r",\s*|\s+and\s+|\s*&\s*", hit.group(1)):
            t = t.strip().upper()
            if re.fullmatch(r"[A-Z0-9]{1,12}", t):
                out.append({"ts": m["ts"], "ticker": t})
    return out


def main():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    close = panel["close"]
    top = top_by_volume(panel["qvol"], close, 100)
    ret = close.pct_change(fill_method=None)
    market = ret.where(top.shift(1).fillna(False)).mean(axis=1)
    rows = []
    evs = events()
    print(f"{len(evs)} coins tagged in {len({e['ts'] for e in evs})} announcements")
    for e in evs:
        sym = next((s for s in (e["ticker"] + "USDT", "1000" + e["ticker"] + "USDT") if s in close.columns), None)
        d = pd.Timestamp(e["ts"], unit="s").normalize()
        if sym is None or d not in close.index or np.isnan(close.at[d, sym]):
            continue
        px = close[sym]
        i = close.index.get_loc(d)
        row = {"ts": e["ts"], "day": d, "sym": sym,
               "reaction": -(px.iloc[i] / px.iloc[i - 1] - 1) if i > 0 and not np.isnan(px.iloc[i - 1]) else np.nan,
               "in_top100": bool(top.iloc[i - 1][sym]) if i > 0 else False}
        months = sorted({(d + pd.Timedelta(days=k)).strftime("%Y-%m") for k in range(0, max(HOLDS) + 1)})
        fund = load_funding(sym, months)
        for h in HOLDS:
            path = px.iloc[i:i + h + 1].dropna()
            if len(path) < 2:
                continue
            j_end = path.index[-1]
            raw = -(path.iloc[-1] / path.iloc[0] - 1)
            mkt = (1 + market.loc[d:j_end].iloc[1:]).prod() - 1
            f = fund[(fund.index > d + pd.Timedelta(days=1)) & (fund.index <= j_end + pd.Timedelta(days=1))].sum() if len(fund) else 0.0
            row[f"raw_{h}"] = raw - COST + f
            row[f"adj_{h}"] = raw + mkt - COST + f
            row[f"fund_{h}"] = f
            row[f"worst_{h}"] = path.max() / path.iloc[0] - 1
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("ts")
    print(f"{len(df)} had a Binance perp trading on the announcement day ({df['in_top100'].mean():.0%} in the top 100)\n")

    def line(x: pd.DataFrame, col: str) -> str:
        per = x.groupby("ts")[col].mean().dropna()
        t = per.mean() / per.std() * np.sqrt(len(per)) if len(per) > 2 else np.nan
        return f"{per.mean():+6.1%} (t {t:+.1f}, {len(per)} posts, coins won {(x[col] > 0).mean():3.0%})"

    half = df["ts"].quantile(0.5)
    print(f"reaction (previous close -> announcement-day close, short side): {line(df, 'reaction')}")
    for h in HOLDS:
        print(f"short, hold {h:2d}d  raw {line(df, f'raw_{h}')} | market-adjusted {line(df, f'adj_{h}')}")
        print(f"                   adjusted, first half {line(df[df['ts'] <= half], f'adj_{h}')} | "
              f"second half {line(df[df['ts'] > half], f'adj_{h}')}")
    print(f"\nfunding received by the short over 7 days: mean {df['fund_7'].mean():+.2%}; "
          f"worst rise against the short within 7 days: median {df['worst_7'].median():+.0%}, max {df['worst_7'].max():+.0%}")
    print("\nby year (market-adjusted 7d):")
    for y, g in df.groupby(df["day"].dt.year):
        print(f"  {y}: {line(g, 'adj_7')}")


if __name__ == "__main__":
    main()

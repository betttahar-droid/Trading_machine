"""
What the smart-money book would mean for the user's goal: EUR 500 plus a monthly deposit, reaching EUR 10,000.

Daily returns 2022-01 .. now (the period all books exist, the 2022 crash included):
  plan        crypto trend (live rules) + TradFi trend (2026 funding), equal risk, at the plan's level-2 risk
  plan+smart  the same plus the smart-money book (positioning_lab, 7 slices, funding), equal risk, scaled to the SAME
              volatility as the plan, so the comparison is about quality, not about taking more risk
Monthly paths are drawn as 30-day blocks (5,000 paths, 36 months); the deposit arrives every 30 days.

    python -m backend.goal_lab
"""

import numpy as np
import pandas as pd

from backend import positioning_lab as pl
from backend.cross_asset_lab import BINANCE_TRADFI, FUNDING_2026, crypto_trend_returns, etf_prices, tsmom_returns

LEVEL2 = 1.36        # the plan's level-2 risk is 1.36x the 1%-risk-per-trade scaling used in the labs


def books() -> pd.DataFrame:
    sig, ret, member, _ = pl.signals()
    held = sorted(set(ret.columns[member.any().to_numpy()]))
    fund = pl.daily_funding(held, ret.index).reindex(columns=ret.columns).fillna(0.0)
    smart = pd.concat([pl.backtest(sig["smart"], ret, member, offset=o, funding=fund) for o in range(7)], axis=1).mean(axis=1)
    crypto = crypto_trend_returns()
    days = pd.date_range("2022-01-01", crypto.index.max(), freq="D")
    tb = tsmom_returns(etf_prices("2005-01-01")[list(BINANCE_TRADFI)], FUNDING_2026)
    return pd.DataFrame({"c": crypto.reindex(days), "t": tb.reindex(days), "s": smart.reindex(days)}).fillna(0.0)


def combo(b: pd.DataFrame, cols) -> pd.Series:
    x = b[cols]
    w = (1 / x.std()) / (1 / x.std()).sum()
    return (x * w).sum(axis=1)


def odds(r: pd.Series, deposit: float, n_paths: int = 5000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    a = r.to_numpy()
    hit = np.full(n_paths, np.nan)
    end24 = np.zeros(n_paths)
    for p in range(n_paths):
        eq = 500.0
        starts = rng.integers(0, len(a) - 30, size=36)
        for m, s0 in enumerate(starts, 1):
            eq = max(eq * np.prod(1 + a[s0:s0 + 30]), 0.0) + deposit
            if np.isnan(hit[p]) and eq >= 10_000:
                hit[p] = m
            if m == 24:
                end24[p] = eq
    return {"12m": np.mean(hit <= 12), "24m": np.mean(hit <= 24), "36m": np.mean(hit <= 36),
            "median_months": float(np.nanmedian(hit)) if np.isfinite(hit).any() else np.nan,
            "below_deposits_24m": np.mean(end24 < 500 + 24 * deposit)}


def main():
    b = books()
    plan = combo(b, ["c", "t"])
    both = combo(b, ["c", "t", "s"])
    plan = plan * (b["c"].std() * LEVEL2 / plan.std())     # the plan at level 2 (same scaling as the other labs)
    both = both * (plan.std() / both.std())
    for name, r in (("plan", plan), ("plan + smart money", both)):
        eq = (1 + r).cumprod()
        print(f"{name:20s} 2022-now: {eq.iloc[-1] ** (365 / len(r)) - 1:+.0%}/yr, vol {r.std() * np.sqrt(365):.0%}, "
              f"Sharpe {r.mean() / r.std() * np.sqrt(365):.2f}, max DD {(eq / eq.cummax() - 1).min():+.0%}")
    print("\nEUR 500 start + monthly deposit -> EUR 10,000 (share of 5,000 simulated paths)")
    for dep in (100, 300, 500):
        months_saving = int(np.ceil((10_000 - 500) / dep))
        print(f"  deposit {dep}/month (saving alone: {months_saving} months)")
        for name, r in (("plan", plan), ("plan + smart money", both)):
            o = odds(r, dep)
            print(f"    {name:20s} within 12m {o['12m']:4.0%} | 24m {o['24m']:4.0%} | 36m {o['36m']:4.0%} | "
                  f"median {o['median_months']:.0f} months | below deposits at 24m {o['below_deposits_24m']:4.0%}")


def project(deposit: float = 300.0, n_paths: int = 5000, seed: int = 11):
    """EUR 500 + `deposit` a month with both improvements: gold via PAXGUSDT (~4%/yr funding instead of 12.1%) and
    the smart-money book added at equal risk (scaled to the plan's volatility at each level). Also a cautious case
    with the average daily return halved (volatility unchanged)."""
    b = books()
    paxg = dict(FUNDING_2026, GLD=0.04)
    crypto_days = b.index
    b["t"] = tsmom_returns(etf_prices("2005-01-01")[list(BINANCE_TRADFI)], paxg).reindex(crypto_days).fillna(0.0)
    plan = combo(b, ["c", "t"])
    full = combo(b, ["c", "t", "s"])
    rng = np.random.default_rng(seed)
    for level in (2, 3):
        scale = 0.68 * level
        p = plan * (b["c"].std() * scale / plan.std())
        f = full * (p.std() / full.std())
        for name, r in (("plan (gold via PAXG)", p), ("plan + smart money", f), ("plan + smart money, cautious", f - f.mean() / 2)):
            a = r.to_numpy()
            eq_paths = np.zeros((n_paths, 37))
            for k in range(n_paths):
                eq = 500.0
                eq_paths[k, 0] = eq
                starts = rng.integers(0, len(a) - 30, size=36)
                for m, s0 in enumerate(starts, 1):
                    eq = max(eq * np.prod(1 + a[s0:s0 + 30]), 0.0) + deposit
                    eq_paths[k, m] = eq
            hit = np.array([np.argmax(row >= 10_000) if (row >= 10_000).any() else np.nan for row in eq_paths])
            ann = (1 + r).prod() ** (365 / len(r)) - 1
            cells = []
            for mth in (12, 24, 36):
                q = np.percentile(eq_paths[:, mth], [10, 50, 90])
                dep = 500 + deposit * mth
                cells.append(f"{mth}m: deposited {dep:,.0f} -> median {q[1]:,.0f} (bad 10% {q[0]:,.0f}, good 10% {q[2]:,.0f}), "
                             f"below deposits {np.mean(eq_paths[:, mth] < dep):.0%}")
            print(f"\nlevel {level} | {name}: backtest {ann:+.0%}/yr, vol {r.std() * np.sqrt(365):.0%}, "
                  f"max DD {((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min():+.0%}")
            for c in cells:
                print("   " + c)
            print(f"   10k reached: within 12m {np.mean(hit <= 12):.0%}, 24m {np.mean(hit <= 24):.0%}, 36m {np.mean(hit <= 36):.0%}; "
                  f"median {np.nanmedian(hit):.0f} months")


if __name__ == "__main__":
    import sys
    project() if len(sys.argv) > 1 and sys.argv[1] == "project" else main()

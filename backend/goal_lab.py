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


if __name__ == "__main__":
    main()

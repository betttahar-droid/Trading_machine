"""
Can a pretrained time-series foundation model forecast the 8 coins better than simple rules?

Amazon's Chronos-Bolt (released 2024-11, trained on public time series; run locally on CPU) gets the last 512 4-hour
closes of each coin once a day and forecasts the next 24 hours (6 bars) as quantiles. Tested on 2025-01 .. now only,
after the model was trained:
  direction   median forecast return vs the realised 24h return (Spearman, pooled over coins and days), and a
              long/flat rule (hold a coin for the day when the forecast is up; 0.1% per change) vs holding all 8
  volatility  forecast spread (90% - 10% quantile) vs the realised 24h absolute return, against the obvious baseline:
              the last 30 bars' volatility (what ATR-style sizing already uses)
The same 8 coins and 4h data as the live trend strategy.

    python -m backend.chronos_lab [amazon/chronos-bolt-small ...]
"""

import sys
import time

import numpy as np
import pandas as pd

from backend.backtest_trend import UNIVERSE
from backend.growth_study import load_market_bulk
from backend.universe_data import available_months

TEST_START = "2025-01-01"
CTX, H = 512, 6
COST = 0.001


def closes() -> pd.DataFrame:
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    m = load_market_bulk("4h", last_month, UNIVERSE, lambda s: available_months(s, "4h"))
    df = pd.DataFrame({s: pd.Series({b["timestamp"]: b["close"] for b in m[s]["bars"]}) for s in m})
    df.index = pd.to_datetime(df.index, unit="ms")
    return df.sort_index()


def forecasts(df: pd.DataFrame, model: str) -> pd.DataFrame:
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(model, device_map="cpu", torch_dtype=torch.float32)
    days = [i for i, t in enumerate(df.index) if t >= pd.Timestamp(TEST_START) and t.hour == 0 and i + H < len(df)]
    rows, t0 = [], time.time()
    for n, i in enumerate(days):
        syms, ctx = [], []
        for s in df.columns:
            hist = df[s].iloc[max(0, i + 1 - CTX):i + 1].dropna()
            if len(hist) >= 200:
                syms.append(s)
                ctx.append(torch.tensor(hist.to_numpy(), dtype=torch.float32))
        if not syms:
            continue
        q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
        for k, s in enumerate(syms):
            last = df[s].iat[i]
            past = np.log(df[s].iloc[i - 30:i + 1]).diff().dropna()
            rows.append({"t": df.index[i], "sym": s,
                         "pred_ret": float(q[k, -1, 1]) / last - 1, "pred_ret_4h": float(q[k, 0, 1]) / last - 1,
                         "pred_spread": float(q[k, -1, 2] - q[k, -1, 0]) / last,
                         "ewma_vol": past.std() * np.sqrt(H), "mom_24h": last / df[s].iat[i - H] - 1,
                         "ret_24h": df[s].iat[i + H] / last - 1, "ret_4h": df[s].iat[i + 1] / last - 1})
        if n % 100 == 0:
            print(f"  {model}: day {n}/{len(days)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def report(f: pd.DataFrame, model: str):
    f = f.dropna()
    halves = f["t"] < f["t"].iloc[len(f) // 2]
    print(f"\n=== {model}: {f['t'].nunique()} days x {f['sym'].nunique()} coins, {f['t'].min():%Y-%m-%d} .. {f['t'].max():%Y-%m-%d} ===")
    for label, part in (("all", f), ("first half", f[halves]), ("second half", f[~halves])):
        ic = part["pred_ret"].corr(part["ret_24h"], method="spearman")
        ic4 = part["pred_ret_4h"].corr(part["ret_4h"], method="spearman")
        mom = part["mom_24h"].corr(part["ret_24h"], method="spearman")
        vol_c = part["pred_spread"].corr(part["ret_24h"].abs(), method="spearman")
        vol_b = part["ewma_vol"].corr(part["ret_24h"].abs(), method="spearman")
        print(f"  {label:11s} direction IC 24h {ic:+.3f} | 4h {ic4:+.3f} | naive momentum IC {mom:+.3f} || "
              f"volatility: Chronos {vol_c:+.3f} vs last-30-bar vol {vol_b:+.3f}")
    # long/flat by forecast vs hold all, equal weight per coin-day
    p = f.pivot(index="t", columns="sym", values="ret_24h")
    sig = (f.pivot(index="t", columns="sym", values="pred_ret") > 0).astype(float)
    turn = sig.diff().abs().fillna(sig).sum(axis=1) / sig.shape[1]
    strat = (sig * p).sum(axis=1) / p.notna().sum(axis=1) - turn * COST
    hold = p.mean(axis=1)
    for name, r in (("Chronos long/flat", strat), ("hold all 8", hold)):
        ann = r.mean() * 365
        print(f"  {name:18s} {ann:+.1%}/yr, Sharpe {r.mean() / r.std() * np.sqrt(365):+.2f}, "
              f"time invested {sig.values.mean():.0%}" if name.startswith("Chronos") else
              f"  {name:18s} {ann:+.1%}/yr, Sharpe {r.mean() / r.std() * np.sqrt(365):+.2f}")


def main():
    models = sys.argv[1:] or ["amazon/chronos-bolt-small", "amazon/chronos-bolt-base"]
    df = closes()
    for m in models:
        f = forecasts(df, m)
        f.to_csv(f"data/chronos_{m.split('/')[-1]}.csv", index=False)
        report(f, m)


if __name__ == "__main__":
    main()

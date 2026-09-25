"""
Can a trained model pick better trend entries? (meta-labeling)

For every entry of the trend rules on the point-in-time top-30 universe since 2020, record what was
known at the signal bar (breakout strength, volatility, recent returns, volume, funding, Bitcoin's
trend, market breadth, liquidity rank) and whether the trade made money. Models are trained
walk-forward: each calendar year is predicted by a model trained only on trades that had closed
before that year began.

  GBM    LightGBM on the numeric features (the standard tool for tabular data)
  LOGIT  logistic regression on the same features (simple baseline)
  LAYA   logistic regression on Laya's frozen encoder embeddings of the features written as text.
         This is the cheap version of fine-tuning Laya: if its representation carries no signal,
         a full fine-tune on a few thousand trades will not find one either.

A model helps only if, out of sample, the trades it keeps beat the trades it drops by more than
dropping the same number at random.

    python -m backend.ml_lab
"""

import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.backtest_trend import DATA_DIR
from backend.growth_study import LIVE, load_market_bulk, symbol_trades
from backend.trend_strategy import compute_features
from backend.universe_data import available_months, daily_panel, top_by_volume

TOP_N = 30
START = "2020-06-01"
CACHE = os.path.join(DATA_DIR, "ml_cache")
FEATURES = ["brk_atr", "atr_pct", "r20", "r60", "r120", "r360", "vol_ratio", "sma200_dist", "fund_3d",
            "btc_r120", "btc_sma_dist", "breadth", "vol_rank", "age_bars"]


def build_dataset(last_month: str) -> pd.DataFrame:
    panel = daily_panel(last_month)
    member = top_by_volume(panel["qvol"], panel["close"], TOP_N)
    vol_rank = panel["qvol"].rolling(30, min_periods=30).mean().shift(1).rank(axis=1, ascending=False)
    ever = [s for s in member.columns if member[s].any()]
    market = load_market_bulk("4h", last_month, ever, lambda s: available_months(s, "4h"))

    closes = pd.DataFrame({s: pd.Series({b["timestamp"]: b["close"] for b in m["bars"]}) for s, m in market.items()})
    closes = closes.sort_index()
    sma = closes.rolling(120, min_periods=120).mean()
    breadth = (closes > sma).sum(axis=1) / sma.notna().sum(axis=1).replace(0, np.nan)
    btc = closes["BTCUSDT"]
    btc_r120 = btc / btc.shift(120) - 1
    btc_sma = btc / btc.rolling(120).mean() - 1

    day_ms = 86_400_000
    mem = {d.value // 10**6: row for d, row in zip(member.index, member[ever].to_numpy())}
    col = {s: k for k, s in enumerate(ever)}
    start_ms = int(pd.Timestamp(START).value // 10**6)
    rows = []
    for sym, m in market.items():
        f = compute_features(m["bars"], LIVE)
        c, atr, up = f["close"], f["atr"], f["upper"]
        v = np.array([b["volume"] for b in m["bars"]])

        def can_enter(ts, sym=sym):
            row = mem.get(ts // day_ms * day_ms)
            return row is not None and bool(row[col[sym]])

        for t in symbol_trades(m, f, LIVE, start_ms, can_enter):
            i, ts = t["signal_i"], t["ts"]
            day = pd.Timestamp(ts // day_ms * day_ms, unit="ms")

            def ret(k):
                return c[i] / c[i - k] - 1 if i >= k else np.nan
            rows.append({
                "sym": sym, "ts": ts, "exit_ts": t["exit_ts"], "r": t["r"],
                "brk_atr": (c[i] - up[i]) / atr[i], "atr_pct": atr[i] / c[i],
                "r20": ret(20), "r60": ret(60), "r120": ret(120), "r360": ret(360),
                "vol_ratio": v[i] / max(v[max(0, i - 20):i].mean(), 1e-12) if i >= 20 else np.nan,
                "sma200_dist": c[i] / c[i - 200:i].mean() - 1 if i >= 200 else np.nan,
                "fund_3d": float(m["funding"][max(0, i - 17):i + 1].sum()),
                "btc_r120": btc_r120.get(ts, np.nan), "btc_sma_dist": btc_sma.get(ts, np.nan),
                "breadth": breadth.get(ts, np.nan),
                "vol_rank": vol_rank[sym].get(day, np.nan) if sym in vol_rank else np.nan,
                "age_bars": i,
            })
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def as_text(row) -> str:
    def pct(x):
        return "unknown" if pd.isna(x) else f"{x:+.1%}"
    return (f"Instrument: {row.sym}. Price closed {row.brk_atr:.2f} ATR above its 20-day high. "
            f"Volatility {row.atr_pct:.2%} per 4h bar. Return over 3 days {pct(row.r20)}, 10 days {pct(row.r60)}, "
            f"20 days {pct(row.r120)}, 60 days {pct(row.r360)}. Volume {row.vol_ratio:.1f}x its recent average. "
            f"Distance from 33-day average {pct(row.sma200_dist)}. Funding over 3 days {row.fund_3d:+.3%}. "
            f"Bitcoin 20-day return {pct(row.btc_r120)}, distance from its 20-day average {pct(row.btc_sma_dist)}. "
            f"Share of coins above their 20-day average {row.breadth:.0%}. Liquidity rank {row.vol_rank:.0f}.")


def laya_embeddings(df: pd.DataFrame) -> np.ndarray:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"laya_emb_{len(df)}_{int(df['ts'].iloc[-1])}.npy")
    if os.path.exists(path):
        return np.load(path)
    import laya
    import torch
    agent = laya.load("convaiinnovations/laya", device="cpu")
    embed = laya.embed_fn_from_agent(agent, max_length=256, batch_size=16)
    texts = [as_text(r) for r in df.itertuples()]
    t0, out = time.time(), []
    with torch.no_grad():
        for k in range(0, len(texts), 64):
            out.append(np.asarray(embed(texts[k:k + 64])))
            print(f"  embedded {min(k + 64, len(texts))}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    emb = np.vstack(out)
    np.save(path, emb)
    return emb


def walk_forward(df: pd.DataFrame, X: np.ndarray, make_model) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    y = (df["r"] > 0).to_numpy().astype(int)
    for year in range(2021, df["date"].dt.year.max() + 1):
        t0 = pd.Timestamp(f"{year}-01-01").value // 10**6
        t1 = pd.Timestamp(f"{year + 1}-01-01").value // 10**6
        train = (df["exit_ts"] < t0).to_numpy()
        test = ((df["ts"] >= t0) & (df["ts"] < t1)).to_numpy()
        if train.sum() < 100 or test.sum() == 0:
            continue
        model = make_model()
        model.fit(X[train], y[train])
        p_train = model.predict_proba(X[train])[:, 1]
        # keep a test trade if it scores above the training set's median (no look-ahead threshold)
        pred[test] = model.predict_proba(X[test])[:, 1] - np.median(p_train)
    return pred


def report(name: str, df: pd.DataFrame, pred: np.ndarray, rng):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(pred)
    r, p = df["r"].to_numpy()[ok], pred[ok]
    keep = p > 0
    auc = roc_auc_score(r > 0, p)
    rand = np.array([r[rng.choice(len(r), keep.sum(), replace=False)].mean() for _ in range(5000)])
    pval = np.mean(rand >= r[keep].mean())
    late = (df["date"].to_numpy()[ok] >= np.datetime64("2024-07-01"))
    parts = []
    for lbl, m in (("2021-24H1", ~late), ("2024H2+", late)):
        k = keep & m
        parts.append(f"{lbl}: keep {k.sum()}/{m.sum()} E[R] {r[k].mean():+.3f} vs drop {r[m & ~keep].mean():+.3f}")
    print(f"  {name:<6} AUC {auc:.3f} | kept E[R] {r[keep].mean():+.3f} vs dropped {r[~keep].mean():+.3f} "
          f"(all {r.mean():+.3f}) | random does as well {pval:.0%} | " + " | ".join(parts))


def main():
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    df = build_dataset(last_month)
    print(f"{len(df)} trend trades on the top-{TOP_N} universe {df['date'].min():%Y-%m} .. {df['date'].max():%Y-%m}: "
          f"win {np.mean(df['r'] > 0):.0%}, E[R] {df['r'].mean():+.3f}", flush=True)
    X = df[FEATURES].to_numpy(float)
    rng = np.random.default_rng(0)

    gbm = walk_forward(df, X, lambda: LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=8,
                                                     min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                     colsample_bytree=0.8, verbose=-1))
    logit = walk_forward(df, X, lambda: make_pipeline(SimpleImputer(), StandardScaler(),
                                                      LogisticRegression(C=0.1, max_iter=2000)))
    print("\nWalk-forward results (every prediction is out of sample):")
    report("GBM", df, gbm, rng)
    report("LOGIT", df, logit, rng)

    emb = laya_embeddings(df)
    lay = walk_forward(df, emb, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)))
    report("LAYA", df, lay, rng)

    imp = LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=40,
                         verbose=-1).fit(X, df["r"] > 0).feature_importances_
    print("\nGBM feature usage (all data): " + ", ".join(f"{n} {v}" for n, v in sorted(zip(FEATURES, imp), key=lambda z: -z[1])))


if __name__ == "__main__":
    main()

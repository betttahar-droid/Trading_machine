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

--portfolio runs the live 8-coin strategy with a Laya filter scoring every breakout signal, against
the unfiltered strategy and against skipping the same share of signals at random.

A model helps only if, out of sample, the trades it keeps beat the trades it drops by more than
dropping the same number at random.

    python -m backend.ml_lab
    python -m backend.ml_lab --portfolio
"""

import os
import sys
import time
import zlib
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.backtest_trend import DATA_DIR, UNIVERSE
from backend.growth_study import LIVE, load_market_bulk, symbol_trades
from backend.trend_strategy import compute_features
from backend.universe_data import available_months, daily_panel, top_by_volume

TOP_N = 30
START = "2020-06-01"
CACHE = os.path.join(DATA_DIR, "ml_cache")
FEATURES = ["brk_atr", "atr_pct", "r20", "r60", "r120", "r360", "vol_ratio", "sma200_dist", "fund_3d",
            "btc_r120", "btc_sma_dist", "breadth", "vol_rank", "age_bars"]


def build_dataset(last_month: str) -> pd.DataFrame:
    path = os.path.join(CACHE, f"trades_top{TOP_N}_{last_month}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    df = _build_dataset(last_month)
    os.makedirs(CACHE, exist_ok=True)
    df.to_pickle(path)
    return df


def market_context(last_month: str) -> dict:
    """4h bars for every coin ever in the top N, plus the market-wide series the features use."""
    panel = daily_panel(last_month)
    member = top_by_volume(panel["qvol"], panel["close"], TOP_N)
    ever = [s for s in member.columns if member[s].any()]
    market = load_market_bulk("4h", last_month, ever, lambda s: available_months(s, "4h"))
    closes = pd.DataFrame({s: pd.Series({b["timestamp"]: b["close"] for b in m["bars"]}) for s, m in market.items()})
    closes = closes.sort_index()
    sma = closes.rolling(120, min_periods=120).mean()
    btc = closes["BTCUSDT"]
    return {"market": market, "member": member[ever], "ever": ever,
            "vol_rank": panel["qvol"].rolling(30, min_periods=30).mean().shift(1).rank(axis=1, ascending=False),
            "breadth": (closes > sma).sum(axis=1) / sma.notna().sum(axis=1).replace(0, np.nan),
            "btc_r120": btc / btc.shift(120) - 1, "btc_sma": btc / btc.rolling(120).mean() - 1}


def features_at(ctx: dict, sym: str, m: dict, f: Dict[str, np.ndarray], v: np.ndarray, i: int) -> dict:
    """What is known at the close of signal bar i."""
    c, atr, up = f["close"], f["atr"], f["upper"]
    ts = m["bars"][i]["timestamp"]
    day = pd.Timestamp(ts // 86_400_000 * 86_400_000, unit="ms")

    def ret(k):
        return c[i] / c[i - k] - 1 if i >= k else np.nan
    return {
        "brk_atr": (c[i] - up[i]) / atr[i], "atr_pct": atr[i] / c[i],
        "r20": ret(20), "r60": ret(60), "r120": ret(120), "r360": ret(360),
        "vol_ratio": v[i] / max(v[max(0, i - 20):i].mean(), 1e-12) if i >= 20 else np.nan,
        "sma200_dist": c[i] / c[i - 200:i].mean() - 1 if i >= 200 else np.nan,
        "fund_3d": float(m["funding"][max(0, i - 17):i + 1].sum()),
        "btc_r120": ctx["btc_r120"].get(ts, np.nan), "btc_sma_dist": ctx["btc_sma"].get(ts, np.nan),
        "breadth": ctx["breadth"].get(ts, np.nan),
        "vol_rank": ctx["vol_rank"][sym].get(day, np.nan) if sym in ctx["vol_rank"] else np.nan,
        "age_bars": i,
    }


def _build_dataset(last_month: str) -> pd.DataFrame:
    ctx = market_context(last_month)
    day_ms = 86_400_000
    mem = {d.value // 10**6: row for d, row in zip(ctx["member"].index, ctx["member"].to_numpy())}
    col = {s: k for k, s in enumerate(ctx["ever"])}
    start_ms = int(pd.Timestamp(START).value // 10**6)
    rows = []
    for sym, m in ctx["market"].items():
        f = compute_features(m["bars"], LIVE)
        v = np.array([b["volume"] for b in m["bars"]])

        def can_enter(ts, sym=sym):
            row = mem.get(ts // day_ms * day_ms)
            return row is not None and bool(row[col[sym]])

        for t in symbol_trades(m, f, LIVE, start_ms, can_enter):
            rows.append({"sym": sym, "ts": t["ts"], "exit_ts": t["exit_ts"], "r": t["r"],
                         **features_at(ctx, sym, m, f, v, t["signal_i"])})
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def as_text(row, with_symbol: bool = True) -> str:
    def pct(x):
        return "unknown" if pd.isna(x) else f"{x:+.1%}"
    return ((f"Instrument: {row.sym}. " if with_symbol else "") +
            f"Price closed {row.brk_atr:.2f} ATR above its 20-day high. "
            f"Volatility {row.atr_pct:.2%} per 4h bar. Return over 3 days {pct(row.r20)}, 10 days {pct(row.r60)}, "
            f"20 days {pct(row.r120)}, 60 days {pct(row.r360)}. Volume {row.vol_ratio:.1f}x its recent average. "
            f"Distance from 33-day average {pct(row.sma200_dist)}. Funding over 3 days {row.fund_3d:+.3%}. "
            f"Bitcoin 20-day return {pct(row.btc_r120)}, distance from its 20-day average {pct(row.btc_sma_dist)}. "
            f"Share of coins above their 20-day average {row.breadth:.0%}. Liquidity rank {row.vol_rank:.0f}.")


def laya_embeddings(df: pd.DataFrame, with_symbol: bool = True, tag: str = "") -> np.ndarray:
    os.makedirs(CACHE, exist_ok=True)
    tag = tag + ("" if with_symbol else "_nosym")
    path = os.path.join(CACHE, f"laya_emb{tag}_{len(df)}_{int(df['ts'].iloc[-1])}.npy")
    if os.path.exists(path):
        return np.load(path)
    import laya
    import torch
    agent = laya.load("convaiinnovations/laya", device="cpu")
    embed = laya.embed_fn_from_agent(agent, max_length=256, batch_size=16)
    texts = [as_text(r, with_symbol) for r in df.itertuples()]
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


def report(name: str, df: pd.DataFrame, pred: np.ndarray, rng, subset=None):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(pred) if subset is None else ~np.isnan(pred) & subset
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
    print(f"  {name:<14} AUC {auc:.3f} | kept E[R] {r[keep].mean():+.3f} vs dropped {r[~keep].mean():+.3f} "
          f"(all {r.mean():+.3f}) | random does as well {pval:.0%} | " + " | ".join(parts))


def portfolio_test(last_month: str):
    """The live 8-coin trend strategy with and without a Laya filter that scores every breakout signal.
    Each year's filter is trained only on top-30 trades that closed before that year began."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from backend.growth_study import run_account
    from backend.strategy_lab import odds, stats
    from backend.trend_strategy import entry_signal

    df = build_dataset(last_month)
    emb = laya_embeddings(df, with_symbol=False)
    y = (df["r"] > 0).to_numpy().astype(int)
    ctx = market_context(last_month)
    t_first = int(pd.Timestamp("2021-01-01").value // 10**6)
    rows = []
    for sym in UNIVERSE:
        m = ctx["market"][sym]
        f = compute_features(m["bars"], LIVE)
        v = np.array([b["volume"] for b in m["bars"]])
        for i, b in enumerate(m["bars"]):
            if b["timestamp"] >= t_first and entry_signal(f, i, LIVE):
                rows.append({"sym": sym, "ts": b["timestamp"], **features_at(ctx, sym, m, f, v, i)})
    sig = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    print(f"{len(sig)} breakout signal bars on the 8 coins since 2021", flush=True)
    sig_emb = laya_embeddings(sig, with_symbol=False, tag="_signals8")

    keep: Dict[tuple, bool] = {}
    for year in range(2021, pd.Timestamp(int(sig["ts"].max()), unit="ms").year + 1):
        t0 = pd.Timestamp(f"{year}-01-01").value // 10**6
        t1 = pd.Timestamp(f"{year + 1}-01-01").value // 10**6
        train = (df["exit_ts"] < t0).to_numpy()
        test = ((sig["ts"] >= t0) & (sig["ts"] < t1)).to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(emb[train], y[train])
        thr = np.median(model.predict_proba(emb[train])[:, 1])
        ok = model.predict_proba(sig_emb[test])[:, 1] > thr
        keep.update(zip(zip(sig["sym"][test], sig["ts"][test]), ok))
    skip_rate = 1 - np.mean(list(keep.values()))

    market8 = {s: ctx["market"][s] for s in UNIVERSE}
    feats = {s: compute_features(m["bars"], LIVE) for s, m in market8.items()}
    timeline = sorted({b["timestamp"] for m in market8.values() for b in m["bars"]})

    def daily(can_enter) -> pd.Series:
        curve: list = []
        run_account(market8, feats, timeline, LIVE, t_first, timeline[-1] + 1, liquidation=False,
                    can_enter=can_enter, curve=curve)
        eq = pd.Series(dict(curve))
        eq.index = pd.to_datetime(eq.index, unit="ms")
        return eq.resample("1D").last().dropna().pct_change().dropna()

    def line(name, r):
        a, b = stats(r[:"2024-06-30"]), stats(r["2024-07-01":])
        return (f"  {name:<28} 2021-24H1: CAGR {a['cagr']:+6.1%} Sharpe {a['sharpe']:+.2f} DD {a['maxdd']:+6.1%} | "
                f"2024H2+: CAGR {b['cagr']:+6.1%} Sharpe {b['sharpe']:+.2f} DD {b['maxdd']:+6.1%}")

    base = daily(None)
    filt = daily(lambda s, ts: keep.get((s, ts), True))
    print(f"\nLaya filter skips {skip_rate:.0%} of breakout signals. Live trend strategy, 1% risk:")
    print(line("unfiltered", base))
    print(line("Laya filter", filt))
    for seed in range(5):
        rnd = daily(lambda s, ts, seed=seed: zlib.crc32(f"{s}{ts}{seed}".encode()) % 10_000 / 10_000 >= skip_rate)
        print(line(f"random skips, seed {seed}", rnd))
    print("\nGrowth odds (2021 on), fresh account every 2 weeks, 12-month runs, L x daily returns:")
    for name, r in (("unfiltered", base), ("Laya filter", filt)):
        print(f"  {name}")
        print(odds(r, (1, 2, 3, 5, 7.5, 10)).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def main():
    if "--portfolio" in sys.argv:
        return portfolio_test(time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400)))
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
    laya_model = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
    lay = walk_forward(df, emb, laya_model)
    report("LAYA", df, lay, rng)

    # Is LAYA just recognising the coin? Same model without the symbol in the text, vs. simple rules.
    lay_ns = walk_forward(df, laya_embeddings(df, with_symbol=False), laya_model)
    report("LAYA no symbol", df, lay_ns, rng)
    ok = ~np.isnan(lay)
    rules = {"top-10 liquidity": df["vol_rank"] <= 10, "listed >1 year": df["age_bars"] >= 6 * 365,
             "8 live coins": df["sym"].isin(UNIVERSE)}
    for name, keep in rules.items():
        report(name, df, np.where(keep, 1.0, -1.0) + np.where(ok, 0.0, np.nan), rng)
    print("\nOn the 8 live coins only (models trained on all top-30 trades):")
    live = df["sym"].isin(UNIVERSE).to_numpy()
    for name, pred in (("GBM", gbm), ("LAYA", lay), ("LAYA no symbol", lay_ns)):
        report(name, df, pred, rng, subset=live)

    imp = LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=40,
                         verbose=-1).fit(X, df["r"] > 0).feature_importances_
    print("\nGBM feature usage (all data): " + ", ".join(f"{n} {v}" for n, v in sorted(zip(FEATURES, imp), key=lambda z: -z[1])))


if __name__ == "__main__":
    main()

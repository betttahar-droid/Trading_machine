"""
Follow-up ideas on the news-reading LLM scores (backend/news_llm_lab.py). The zero-shot direction calls failed, so
these ask whether the same information is useful some other way. Every idea is built and tuned on 2022-02 .. 2024-12
only and then checked, unchanged, on 2025-01 .. now (after the models' training data):

  A  reaction   big news the market has not moved on yet -> follow it (under-reaction);
                a big move on news the model calls minor -> fade it (over-reaction)
  B  learned    gradient boosting on all models' answer probabilities + the recent move, volatility and hour,
                trained to predict the next 1h / 4h return sign, instead of trusting the models' own direction
  C  text       headline embeddings from a small local model (bge-small, CPU) + ridge regression on the returns
  D  volatility does the model's "impact" predict the size of the next move beyond recent volatility?

Same entry (1-2 minutes after the post), 0.15% round-trip costs and one position at a time as the main test.

    python -m backend.news_llm_creative
"""

import glob
import os

import numpy as np
import pandas as pd

from backend.news_llm_lab import COST, OUT, POSTS, TEST_START, load_prices, outcomes, trades

LETTERS = "ABCDEFG"


def dataset() -> pd.DataFrame:
    df = outcomes(pd.read_json(POSTS, lines=True))
    t0, px = load_prices()
    m0 = df["entry_min"].to_numpy()
    df["pre60"] = px[m0] / px[m0 - 60] - 1
    lr = np.diff(np.log(px))
    csum, csum2 = np.concatenate([[0], np.cumsum(lr)]), np.concatenate([[0], np.cumsum(lr ** 2)])
    n = 1440
    lo = np.maximum(m0 - n, 0)
    df["vol24h"] = np.sqrt((csum2[m0] - csum2[lo]) / (m0 - lo) * 60)          # typical 1h move over the last day
    df["hour"] = pd.to_datetime(df["ts"], unit="s").dt.hour
    models = []
    for f in sorted(glob.glob(os.path.join(OUT, "scores_*.csv"))):
        name = os.path.basename(f)[7:-4]
        s = pd.read_csv(f).rename(columns={c: f"{name}_{c}" for c in ["score", "impact"] + [f"p_{L}" for L in LETTERS]})
        df = df.merge(s[["id"] + [c for c in s.columns if c.startswith(name)]], on="id", how="inner")
        models.append(name)
    df = df.sort_values("ts").reset_index(drop=True)
    df["score_mean"] = df[[f"{m}_score" for m in models]].mean(axis=1)
    df["impact_mean"] = df[[f"{m}_impact" for m in models]].mean(axis=1)
    return df, models


def net_stats(df: pd.DataFrame, direction: np.ndarray, hk: str, h: int) -> str:
    t = trades(df, direction.astype(int), h, hk)
    if len(t) < 10:
        return f"n={len(t):4d}  (too few)"
    tstat = t["net"].mean() / t["net"].std() * np.sqrt(len(t))
    gross = t["net"] + COST
    return (f"n={len(t):4d}  net {100 * t['net'].mean():+.3f}%/trade (t {tstat:+.2f})  "
            f"gross {100 * gross.mean():+.3f}%  win {100 * (t['net'] > 0).mean():.0f}%")


def split(df):
    tr, te = df[df.ts < TEST_START].reset_index(drop=True), df[df.ts >= TEST_START].reset_index(drop=True)
    return tr, te


def idea_a(df):
    print("\nA  reaction (thresholds from 2022-24)")
    tr, te = split(df)
    quiet = tr["pre15"].abs().quantile(0.5)
    wild = tr["pre15"].abs().quantile(0.9)
    for name, part in (("2022-24", tr), ("2025+ ", te)):
        big = part["score_mean"].abs() >= 1.5
        under = np.where(big & (part["pre15"].abs() < quiet), np.sign(part["score_mean"]), 0)
        over = np.where((part["pre15"].abs() > wild) & (part["impact_mean"] < 0.5), -np.sign(part["pre15"]), 0)
        for hk, h in (("1h", 60), ("4h", 240)):
            print(f"  {name} {hk} under-reaction: {net_stats(part, under, hk, h)}")
            print(f"  {name} {hk} over-reaction:  {net_stats(part, over, hk, h)}")


def idea_b(df, models):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    print("\nB  gradient boosting on the models' answers (trained on 2022-24)")
    feats = [f"{m}_p_{L}" for m in models for L in LETTERS] + ["pre15", "pre60", "vol24h", "hour"]
    tr, te = split(df)
    for hk, h in (("1h", 60), ("4h", 240)):
        y = (tr[f"r_{hk}"] > 0).astype(int)
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                             min_samples_leaf=100, random_state=0).fit(tr[feats], y)
        p_tr, p_te = clf.predict_proba(tr[feats])[:, 1], clf.predict_proba(te[feats])[:, 1]
        hi, lo = np.quantile(p_tr, 0.8), np.quantile(p_tr, 0.2)
        d = np.where(p_te >= hi, 1, np.where(p_te <= lo, -1, 0))
        print(f"  {hk}: AUC train {roc_auc_score(y, p_tr):.3f} | test {roc_auc_score((te[f'r_{hk}'] > 0).astype(int), p_te):.3f}"
              f" | test trades (top/bottom 20%): {net_stats(te, d, hk, h)}")


def idea_c(df):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import RidgeCV
    print("\nC  headline embeddings (bge-small) + ridge regression (trained on 2022-24)")
    path = os.path.join(OUT, "emb_bge_small.npy")
    if os.path.exists(path) and len(np.load(path)) == len(df):
        emb = np.load(path)
    else:
        emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu").encode(
            df["text"].tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.save(path, emb)
    is_tr = (df.ts < TEST_START).to_numpy()
    tr, te = split(df)
    for hk, h in (("1h", 60), ("4h", 240)):
        y = tr[f"r_{hk}"].clip(*tr[f"r_{hk}"].quantile([0.01, 0.99]))
        reg = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(emb[is_tr], y)
        p_tr, p_te = reg.predict(emb[is_tr]), reg.predict(emb[~is_tr])
        hi, lo = np.quantile(p_tr, 0.8), np.quantile(p_tr, 0.2)
        d = np.where(p_te >= hi, 1, np.where(p_te <= lo, -1, 0))
        ic = pd.Series(p_te).corr(te[f"r_{hk}"], method="spearman")
        print(f"  {hk}: alpha {reg.alpha_:.0f} | test IC {ic:+.3f} | test trades: {net_stats(te, d, hk, h)}")


def idea_d(df):
    print("\nD  does the model's 'impact' predict the size of the next hour's move? (Spearman with |r_1h|)")
    for name, part in zip(("2022-24", "2025+ "), split(df)):
        a = part["r_1h"].abs()
        resid = a / part["vol24h"]                          # move size relative to recent volatility
        print(f"  {name}: recent vol {part['vol24h'].corr(a, method='spearman'):+.3f} | impact {part['impact_mean'].corr(a, method='spearman'):+.3f}"
              f" | impact vs move / recent vol {part['impact_mean'].corr(resid, method='spearman'):+.3f}")


def idea_e(df, models):
    """Slow version: the day's average news tone vs the next days' Bitcoin return (no speed needed)."""
    print("\nE  daily news tone (average score of the day's posts) vs Bitcoin over the next 1 / 3 days")
    t0, px = load_prices()
    day = (df["ts"] // 86400).astype(int)
    close = {d: px[min(d * 1440 + 1440 - t0, len(px) - 1)] for d in range(day.min(), day.max() + 4)}   # 00:00 next day
    for col in [f"{m}_score" for m in models] + ["score_mean"]:
        tone = df.groupby(day)[col].mean()
        z = (tone - tone.rolling(30, min_periods=10).mean()) / tone.rolling(30, min_periods=10).std()
        d = pd.DataFrame({"tone": tone, "z": z})
        d["r1"] = [close[k + 1] / close[k] - 1 if k + 1 in close and close[k] else np.nan for k in d.index]
        d["r3"] = [close[k + 3] / close[k] - 1 if k + 3 in close and close[k] else np.nan for k in d.index]
        d = d.dropna()
        test = d.index * 86400 >= TEST_START
        out = []
        for name, part in (("2022-24", d[~test]), ("2025+", d[test])):
            pos = (part["z"] > 0).astype(float)                     # long the day after an above-normal tone day
            strat = pos * part["r1"] - pos.diff().abs().fillna(0) * COST / 2
            out.append(f"{name}: IC 1d {part['z'].corr(part['r1'], method='spearman'):+.3f}, "
                       f"3d {part['z'].corr(part['r3'], method='spearman'):+.3f}, long-when-positive Sharpe "
                       f"{strat.mean() / strat.std() * np.sqrt(365):+.2f} vs hold {part['r1'].mean() / part['r1'].std() * np.sqrt(365):+.2f}")
        print(f"  {col:15s} " + " | ".join(out))


ALTS = {"ETHUSDT": r"ethereum|\$eth\b|\beth\b", "XRPUSDT": r"\bxrp\b|ripple", "SOLUSDT": r"solana|\$sol\b",
        "DOGEUSDT": r"dogecoin|\$doge\b", "ADAUSDT": r"cardano|\$ada\b", "BNBUSDT": r"\$bnb\b|\bbnb\b",
        "1000SHIBUSDT": r"shiba|\$shib\b", "AVAXUSDT": r"avalanche|\$avax\b", "LINKUSDT": r"chainlink|\$link\b",
        "DOTUSDT": r"polkadot|\$dot\b", "1000PEPEUSDT": r"\$pepe\b|\bpepe\b", "TRXUSDT": r"\btron\b|\$trx\b",
        "SUIUSDT": r"\$sui\b|\bsui\b", "LTCUSDT": r"litecoin|\$ltc\b"}


def alt_events(df, models) -> pd.DataFrame:
    """Headlines naming one of ALTS, with that coin's perp returns from 1-2 minutes after the post, and the same
    coin's return at the same clock time a week earlier (a control for time-of-day/week drift)."""
    import re
    from concurrent.futures import ThreadPoolExecutor
    from backend.event_lab import minute_bars
    ev = []
    for r in df.itertuples():
        for sym, pat in ALTS.items():
            if re.search(pat, r.text.lower()):
                ev.append({"ts": r.ts, "sym": sym, **{m: getattr(r, f"{m}_score") for m in models}})
    ev = pd.DataFrame(ev)
    need = {(e.sym, pd.Timestamp(e.ts + k * 86400, unit="s").strftime("%Y-%m-%d")) for e in ev.itertuples() for k in (-7, -6, 0, 1)}
    with ThreadPoolExecutor(8) as ex:
        bars = dict(zip(need, ex.map(lambda sd: minute_bars(*sd), need)))

    def px(sym, m):
        b = bars.get((sym, pd.Timestamp(m * 60, unit="s").strftime("%Y-%m-%d")))
        return float(b["open"].get(m, np.nan)) if b is not None else np.nan
    rows = []
    for e in ev.itertuples():
        m0 = (e.ts + 119) // 60                                  # same entry as the Bitcoin test
        p0, w0 = px(e.sym, m0), px(e.sym, m0 - 7 * 1440)
        rows.append({**e._asdict(), "m0": m0, "first": p0 / px(e.sym, e.ts // 60) - 1,
                     **{f"r_{hk}": px(e.sym, m0 + h) / p0 - 1 for hk, h in (("1h", 60), ("4h", 240))},
                     **{f"c_{hk}": px(e.sym, m0 - 7 * 1440 + h) / w0 - 1 for hk, h in (("1h", 60), ("4h", 240))}})
    x = pd.DataFrame(rows).drop(columns=["Index"]).dropna(subset=["first", "r_1h", "r_4h"])
    x["year"] = pd.to_datetime(x["ts"], unit="s").dt.year
    return x.sort_values("ts").reset_index(drop=True)


def _alt_trades(x, d, hk, h):
    out, free = [], {}
    for r, dd in zip(x.itertuples(), d):
        if dd == 0 or r.m0 < free.get(r.sym, -1):              # one position per coin at a time
            continue
        free[r.sym] = r.m0 + h
        out.append((r.year, dd, dd * getattr(r, f"r_{hk}") - COST, dd * getattr(r, f"c_{hk}") - COST))
    return pd.DataFrame(out, columns=["year", "dir", "net", "ctrl"])


def idea_f(df, models):
    """Altcoin headlines: follow the coin's own first 1-2 minutes, or the model's tone, on that coin."""
    print("\nF  altcoin headlines, traded on the named coin's perp (one position per coin, 0.15% costs)")
    x = alt_events(df, models)

    def s(v):
        return f"{100 * v.mean():+.3f}% (t {v.mean() / v.std() * np.sqrt(len(v)):+.2f}, n {len(v)})" if len(v) > 5 else "n<6"
    rules = [("follow first move", np.sign(x["first"]).astype(int)), ("long every headline", np.ones(len(x), int))]
    rules += [(f"{m} tone", np.where(x[m] >= 1, 1, np.where(x[m] <= -1, -1, 0))) for m in models]
    for hk, h in (("1h", 60), ("4h", 240)):
        for name, d in rules:
            t = _alt_trades(x, d, hk, h)
            for per, tt in (("2022-24", t[t.year < 2025]), ("2025+", t[t.year >= 2025])):
                print(f"  {hk} {name:20s} {per:7s} net {s(tt.net)} | long {s(tt[tt.dir > 0].net)} | "
                      f"short {s(tt[tt.dir < 0].net)} | same trades a week earlier {s(tt.ctrl)}")


def main():
    df, models = dataset()
    print(f"{len(df)} posts with scores from {', '.join(models)}")
    idea_a(df)
    idea_b(df, models)
    idea_c(df)
    idea_d(df)
    idea_e(df, models)
    idea_f(df, models)


if __name__ == "__main__":
    main()

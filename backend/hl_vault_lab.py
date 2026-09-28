"""
Copying skilled traders: Hyperliquid vaults. Anyone can deposit into a vault and share its profits (the leader keeps
10% of profits); every vault's equity and P&L history is public on-chain, closed and blown-up vaults included, so skill
persistence can be tested without survivorship bias. If last quarter's best vaults keep beating the rest, a small
account could hold a basket of them next to the trend plan.

Data: stats-data.hyperliquid.xyz/Mainnet/vaults (every vault ever, open and closed) and each vault's vaultDetails
(allTime equity and cumulative P&L, roughly weekly points). Fixed in advance:

  weekly     return = change in cumulative P&L / equity at the start of the week (deposits and withdrawals excluded);
             history points are ~1/48 of a vault's life apart, so old vaults have gaps and enter only from 2025
             ("--interp" fills the gaps by time interpolation, which biases towards persistence)
  eligible   at formation: equity >= $50k, at least 12 weeks of history, not an HLP sub-vault ("child")
  rule       every 4 weeks rank eligible vaults by their past 12-week return; hold the top 10 equal-weight for 4 weeks;
             the leader's 10% profit share is taken from each vault's positive 4-week return; a vault that closes
             returns its last equity (cash) for the rest of the period
  compare    the average eligible vault, HLP (the exchange's own market-making vault) and Bitcoin
  pass       top 10 beats the average vault in both halves of the history, and the rank correlation between past and
             next returns is positive with t >= 2

    python -m backend.hl_vault_lab collect     # ~1 request per vault, cached in data/hl_vaults/
    python -m backend.hl_vault_lab evaluate [--interp]
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "hl_vaults"))
API = "https://api.hyperliquid.xyz/info"
MIN_EQUITY, LOOKBACK, HOLD, TOP, FEE = 50_000.0, 12, 4, 10, 0.10
INTERP = "--interp" in sys.argv


def vault_list() -> list:
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "_vaults.json")
    if not os.path.exists(path):
        r = requests.get("https://stats-data.hyperliquid.xyz/Mainnet/vaults", timeout=120)
        open(path, "w").write(r.text)
    return json.load(open(path))


def _details(addr: str):
    path = os.path.join(DATA, f"{addr}.json")
    if os.path.exists(path):
        return
    for k in range(6):
        try:
            r = requests.post(API, json={"type": "vaultDetails", "vaultAddress": addr}, timeout=30)
            if r.status_code == 200:
                open(path, "w").write(r.text)
                return
            time.sleep(5 * (k + 1))
        except requests.RequestException:
            time.sleep(5 * (k + 1))


def collect():
    vs = vault_list()
    # skip vaults that never held money (all-time P&L history all zero and no TVL)
    todo = [v["summary"]["vaultAddress"] for v in vs
            if float(v["summary"]["tvl"]) > 0 or any(float(x) != 0 for k, s in v["pnls"] if k == "allTime" for x in s)]
    print(f"{len(vs)} vaults, {len(todo)} ever held money", flush=True)
    done = 0
    with ThreadPoolExecutor(3) as ex:
        for _ in ex.map(_details, todo):
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(todo)}", flush=True)


def weekly_panel():
    rets, equity, child = {}, {}, set()
    for f in os.listdir(DATA):
        if f.startswith("_") or not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(DATA, f)))
        if not isinstance(d, dict) or "portfolio" not in d:
            continue
        if (d.get("relationship") or {}).get("type") == "child":
            child.add(d["vaultAddress"])
            continue
        p = dict(d["portfolio"]).get("allTime")
        if not p or len(p["accountValueHistory"]) < 3:
            continue
        av = pd.Series({t: float(x) for t, x in p["accountValueHistory"]})
        pnl = pd.Series({t: float(x) for t, x in p["pnlHistory"]})
        df = pd.DataFrame({"av": av, "pnl": pnl}).sort_index()
        df.index = pd.to_datetime(df.index, unit="ms")
        if INTERP:
            # history points are ~life/48 apart: interpolate onto week ends. Biased towards persistence (one straight
            # segment can straddle the formation date), so only a check
            grid = pd.date_range(df.index[0].normalize() + pd.offsets.Week(weekday=6), df.index[-1], freq="W-SUN")
            w = df.reindex(df.index.union(grid)).interpolate(method="time").reindex(grid).dropna()
        else:
            w = df.resample("W-SUN").last().dropna()
        if len(w) < 3:
            continue
        start_av = w["av"].shift(1)
        r = (w["pnl"].diff() / start_av).where(start_av > 1_000)
        name = d["vaultAddress"]
        rets[name], equity[name] = r.clip(-1, 5), w["av"]
    R, E = pd.DataFrame(rets).sort_index(), pd.DataFrame(equity).sort_index()
    return R, E


def evaluate():
    R, E = weekly_panel()
    names = {v["summary"]["vaultAddress"]: v["summary"]["name"] for v in vault_list()}
    hlp = next(a for a, n in names.items() if n == "Hyperliquidity Provider (HLP)")
    print(f"{R.shape[1]} vaults with history, {R.index.min():%Y-%m-%d} .. {R.index.max():%Y-%m-%d}")
    weeks = R.index
    age = R.notna().cumsum()
    rows, ics = [], []
    for k in range(LOOKBACK, len(weeks) - 1, HOLD):
        t = weeks[k]
        past = (1 + R.iloc[k - LOOKBACK + 1:k + 1].fillna(0)).prod() - 1
        ok = (E.loc[t] >= MIN_EQUITY) & (age.loc[t] >= LOOKBACK) & R.iloc[k - LOOKBACK + 1:k + 1].notna().sum().ge(LOOKBACK - 2)
        elig = past[ok.reindex(past.index).fillna(False)]
        if len(elig) < 2 * TOP:
            continue
        nxt_w = R.iloc[k + 1:k + 1 + HOLD]
        nxt = (1 + nxt_w.fillna(0)).prod() - 1                   # closed vault: cash after its last week
        nxt = nxt - FEE * nxt.clip(lower=0)
        top = elig.sort_values().index[-TOP:]
        ics.append(elig.rank().corr(nxt[elig.index].rank()))
        rows.append({"t": t, "n": len(elig), "top": nxt[top].mean(), "all": nxt[elig.index].mean(),
                     "bottom": nxt[elig.sort_values().index[:TOP]].mean(),
                     "hlp": (1 + nxt_w[hlp].fillna(0)).prod() - 1 if hlp in nxt_w else np.nan})
    df = pd.DataFrame(rows).set_index("t")
    from backend.universe_data import load_bars
    btc = load_bars("BTCUSDT", "1d", time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400)))["close"]
    df["btc"] = [btc.asof(t + pd.Timedelta(weeks=HOLD)) / btc.asof(t) - 1 for t in df.index]
    ic = pd.Series(ics, index=df.index)
    print(f"{len(df)} four-week periods, {df['n'].median():.0f} eligible vaults on average\n")
    half = df.index[len(df) // 2]
    for lab, x in (("first half", df[:half]), ("second half", df[half:]), ("all", df)):
        cells = []
        for c in ("top", "all", "bottom", "hlp", "btc"):
            s = x[c].dropna()
            eq = (1 + s).prod()
            cells.append(f"{c} {eq ** (13 / len(s)) - 1:+6.0%}/yr (Sharpe {s.mean() / s.std() * np.sqrt(13):+.2f})")
        print(f"{lab:11s} {x.index.min():%Y-%m}..{x.index.max():%Y-%m}: " + " | ".join(cells))
        diff = x["top"] - x["all"]
        print(f"{'':11s} top minus average {diff.mean():+.2%} per 4 weeks (t {diff.mean() / diff.std() * np.sqrt(len(diff)):+.1f}); "
              f"rank corr past/next {ic[x.index].mean():+.3f} (t {ic[x.index].mean() / ic[x.index].std() * np.sqrt(len(x)):+.1f})")
    worst = df["top"].min()
    print(f"\nworst 4 weeks for the top-10 basket {worst:+.1%}; correlation with Bitcoin {df['top'].corr(df['btc']):+.2f}")


if __name__ == "__main__":
    {"collect": collect, "evaluate": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "evaluate"]()

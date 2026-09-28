"""
On-chain fundamentals as signals (DefiLlama, free API):

  chains     does money flowing into a blockchain predict its coin? Signal = 28-day growth of the stablecoin supply on
             the chain (in dollars, so not driven by the coin's own price), e.g. SOL, AVAX, SUI, TRX, ARB ...
  fees       do tokens whose protocols earn growing fees beat the rest (like earnings growth for stocks)? Signal =
             fees of the last 28 days / fees of the 28 days before (protocols and chains with a Binance perp)

Both traded like positioning_lab's book: 7 daily slices, each rebalanced weekly: long the coins with the highest
signal and short the lowest (5 per leg for chains, 8 for fees), equal weight, 0.1% per unit of turnover, real funding;
coins must be Binance USDT perps listed 60+ days and in the point-in-time top 150 by volume. Also shown: the same with
the coins' own past 28-day return removed from the signal (so it is not just momentum), and the long leg alone vs the
average coin. Pass = Sharpe > 0.5 in both 2022-23 and 2024+ with funding, and still positive with momentum removed.
Caveat: DefiLlama backfills history when it adds a protocol, so the set of protocols is today's.

    python -m backend.defi_flow_lab collect
    python -m backend.defi_flow_lab
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

from backend import positioning_lab as pl
from backend.universe_data import daily_panel, top_by_volume

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "defi"))
CHAINS = {  # DefiLlama chain -> Binance perps of its coin, in time order (renames / migrations)
    "Ethereum": ["ETHUSDT"], "Tron": ["TRXUSDT"], "Solana": ["SOLUSDT"], "BSC": ["BNBUSDT"],
    "Hyperliquid L1": ["HYPEUSDT"], "Arbitrum": ["ARBUSDT"], "Polygon": ["MATICUSDT", "POLUSDT"],
    "Avalanche": ["AVAXUSDT"], "XRPL": ["XRPUSDT"], "Aptos": ["APTUSDT"], "Stellar": ["XLMUSDT"], "TON": ["TONUSDT"],
    "OP Mainnet": ["OPUSDT"], "Mantle": ["MNTUSDT"], "Sui": ["SUIUSDT"], "Fantom": ["FTMUSDT"], "Sonic": ["SUSDT"],
    "Sei": ["SEIUSDT"], "Kaia": ["KLAYUSDT", "KAIAUSDT"], "Starknet": ["STRKUSDT"], "Celo": ["CELOUSDT"],
    "Near": ["NEARUSDT"], "Flow": ["FLOWUSDT"], "Polkadot": ["DOTUSDT"], "Cardano": ["ADAUSDT"], "Kava": ["KAVAUSDT"],
    "Algorand": ["ALGOUSDT"], "Injective": ["INJUSDT"], "Hedera": ["HBARUSDT"], "Tezos": ["XTZUSDT"],
    "Harmony": ["ONEUSDT"], "Moonbeam": ["GLMRUSDT"], "CORE": ["COREUSDT"], "ZKsync Era": ["ZKUSDT"],
    "Manta": ["MANTAUSDT"], "Blast": ["BLASTUSDT"], "Scroll": ["SCRUSDT"], "Berachain": ["BERAUSDT"],
    "Monad": ["MONUSDT"], "Plasma": ["XPLUSDT"], "Linea": ["LINEAUSDT"], "Stacks": ["STXUSDT"], "Filecoin": ["FILUSDT"],
    "Conflux": ["CFXUSDT"], "MultiversX": ["EGLDUSDT"], "NEO": ["NEOUSDT"], "Zilliqa": ["ZILUSDT"],
    "IoTeX": ["IOTXUSDT"], "Waves": ["WAVESUSDT"], "Metis": ["METISUSDT"], "Ronin": ["RONINUSDT"],
    "Theta": ["THETAUSDT"], "VeChain": ["VETUSDT"], "Icon": ["ICXUSDT"], "Astar": ["ASTRUSDT"], "Flare": ["FLRUSDT"],
    "Ontology": ["ONTUSDT"], "Kusama": ["KSMUSDT"], "EthereumClassic": ["ETCUSDT"], "Secret": ["SCRTUSDT"],
    "Syscoin": ["SYSUSDT"], "Movement": ["MOVEUSDT"], "Taiko": ["TAIKOUSDT"], "Somnia": ["SOMIUSDT"], "0G": ["0GUSDT"],
    "ApeChain": ["APEUSDT"], "Saga": ["SAGAUSDT"], "Lisk": ["LSKUSDT"], "Immutable zkEVM": ["IMXUSDT"],
    "MANTRA": ["OMUSDT"], "Plume Mainnet": ["PLUMEUSDT"], "Hemi": ["HEMIUSDT"], "Terra Classic": ["1000LUNCUSDT"],
}
MIN_STABLES, MIN_FEES = 20e6, 300e3


def _get(url: str):
    for k in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(10 * (k + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    return None


def _cached(name: str, url: str):
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        return json.load(open(path))
    d = _get(url)
    if d is not None:
        json.dump(d, open(path, "w"))
    time.sleep(0.3)
    return d


def collect():
    os.makedirs(DATA, exist_ok=True)
    for c in CHAINS:
        _cached(f"stables_{c}.json", f"https://stablecoins.llama.fi/stablecoincharts/{requests.utils.quote(c)}")
    ov = _cached("fees_overview.json", "https://api.llama.fi/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true")
    _cached("protocols.json", "https://api.llama.fi/protocols")
    keys = fee_groups(ov)
    print(f"{len(keys)} fee groups with a Binance perp symbol", flush=True)
    for key in keys:
        _cached(f"fees_{key}.json", f"https://api.llama.fi/summary/fees/{key}?dataType=dailyFees")


def fee_groups(ov=None) -> dict:
    """{DefiLlama slug to query: Binance perp symbol} for protocols (grouped by parent) and chains with fees."""
    ov = ov or json.load(open(os.path.join(DATA, "fees_overview.json")))
    prots = json.load(open(os.path.join(DATA, "protocols.json"))) if os.path.exists(os.path.join(DATA, "protocols.json")) \
        else _get("https://api.llama.fi/protocols")
    sym_of = {}
    for p in prots:
        s = (p.get("symbol") or "-").upper()
        if s in ("-", ""):
            continue
        for k in (p.get("slug"), (p.get("parentProtocol") or "").replace("parent#", "")):
            if k:
                sym_of.setdefault(k, s)
    groups = {}
    for p in ov["protocols"]:
        if (p.get("totalAllTime") or 0) < 5e6:
            continue
        if p.get("protocolType") == "chain":
            if p["name"] in CHAINS:
                groups[p["slug"]] = CHAINS[p["name"]][-1]
            continue
        key = (p.get("parentProtocol") or "").replace("parent#", "") or p["slug"]
        s = sym_of.get(key) or sym_of.get(p["slug"])
        if s and s not in ("USDT", "USDC", "ETH", "BTC", "WBTC", "WETH"):
            groups[key] = s + "USDT"
    return groups


def _series(points, value) -> pd.Series:
    s = pd.Series({pd.Timestamp(int(p[0] if isinstance(p, list) else p["date"]), unit="s").normalize(): value(p) for p in points})
    return s[~s.index.duplicated(keep="last")].sort_index()


def panels():
    last_month = time.strftime("%Y-%m", time.gmtime(time.time() - 32 * 86_400))
    panel = daily_panel(last_month)
    close = panel["close"]
    top = top_by_volume(panel["qvol"], close, 150)
    idx = close.loc["2021-10-01":].index

    def returns(syms):                                    # chained returns across renames (MATIC -> POL, ...)
        r = pd.Series(np.nan, index=close.index)
        m = pd.Series(False, index=close.index)
        for s in syms:
            if s in close.columns:
                rs = close[s].pct_change(fill_method=None)
                r = r.fillna(rs)
                m = m | top[s].fillna(False)
        return r, m

    # chains
    sc, cr, cm = {}, {}, {}
    for c, syms in CHAINS.items():
        path = os.path.join(DATA, f"stables_{c}.json")
        if not os.path.exists(path) or not any(s in close.columns for s in syms):
            continue
        d = json.load(open(path)) or []
        s = _series(d, lambda p: sum((p.get("totalCirculatingUSD") or {}).values()))
        sc[c] = s.reindex(pd.date_range(s.index.min(), idx[-1])).ffill().reindex(idx)
        cr[c], cm[c] = returns(syms)
    stables = pd.DataFrame(sc)
    chains = {"ret": pd.DataFrame(cr).reindex(idx), "member": pd.DataFrame(cm).reindex(idx).fillna(False),
              "sig": np.log(stables / stables.shift(28)).where(stables.shift(28) >= MIN_STABLES)}

    # fees
    fr, fm, ff = {}, {}, {}
    for key, sym in fee_groups().items():
        path = os.path.join(DATA, f"fees_{key}.json")
        if not os.path.exists(path) or sym not in close.columns or sym in fr:
            continue
        d = json.load(open(path)) or {}
        pts = d.get("totalDataChart") or []
        if len(pts) < 60:
            continue
        f = _series(pts, lambda p: float(p[1] or 0)).reindex(pd.date_range(idx[0] - pd.Timedelta(days=60), idx[-1])).fillna(0)
        f28 = f.rolling(28).sum()
        ff[sym] = (np.log(f28 / f28.shift(28))).where((f28 >= MIN_FEES) & (f28.shift(28) >= MIN_FEES)).reindex(idx)
        fr[sym] = close[sym].pct_change(fill_method=None).reindex(idx)
        fm[sym] = top[sym].reindex(idx).fillna(False)
    fees = {"ret": pd.DataFrame(fr), "member": pd.DataFrame(fm), "sig": pd.DataFrame(ff)}
    return chains, fees, close


def evaluate():
    chains, fees, close = panels()
    for name, d, legs in (("chain stablecoin inflows", chains, 5), ("protocol fee growth", fees, 8)):
        ret, member, sig = d["ret"], d["member"], d["sig"]
        n = sig.where(member).notna().sum(axis=1)
        print(f"\n=== {name}: {ret.shape[1]} coins, {n['2022':].median():.0f} eligible on a typical day ===")
        cols = [c for c in ret.columns]
        # funding: keyed by the coin's current symbol; chains use their last symbol
        sym = {c: (CHAINS[c][-1] if c in CHAINS else c) for c in cols}
        fund = pl.daily_funding(sorted(set(sym.values())), ret.index)
        fund = pd.DataFrame({c: fund[sym[c]] if sym[c] in fund else 0.0 for c in cols}, index=ret.index)
        px = pd.DataFrame({c: (1 + ret[c].fillna(0)).cumprod() for c in cols})
        past = px / px.shift(28) - 1
        z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
        zs, zp = z(sig.where(member)), z(past.where(member))
        beta = (zs * zp).sum(axis=1) / (zp * zp).sum(axis=1)
        resid = zs - zp.mul(beta, axis=0)
        print(f"  cross-sectional correlation with the past 28-day return: {zs.corrwith(zp, axis=1).mean():+.2f}")
        for lab, s, kw in (("7 slices, price only", sig, {}), ("7 slices, with funding", sig, {"funding": fund}),
                           ("momentum removed, with funding", resid, {"funding": fund}),
                           ("long leg vs average", sig, {"side": "long"}),
                           ("past 28-day return itself", past, {"funding": fund})):
            r = pd.concat([pl.backtest(s.loc["2022":], ret.loc["2022":], member.loc["2022":], legs=legs, offset=o, **kw)
                           for o in range(7)], axis=1).mean(axis=1)
            print(f"  {lab:32s} {pl._fmt(r)}")


if __name__ == "__main__":
    collect() if len(sys.argv) > 1 and sys.argv[1] == "collect" else evaluate()

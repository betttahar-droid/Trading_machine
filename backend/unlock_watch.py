"""
Watch-only token-unlock tracker (research, no real orders).

backend/unlock_lab.py found that coins whose circulating supply jumps >= 5% in a day (unlocks, emissions, airdrops
entering the market) lagged the rest of the market over the next 14 days in the past year, but the evidence is thin
(one year, partly CoinGecko batch updates). This collects fresh, out-of-sample evidence at no cost:

  once a day   one CoinGecko call: price, circulating supply and market cap of the top 250 coins
               (saved in data/unlock_watch/snapshots/YYYY-MM-DD.json)
  event        supply up >= 5% (and < 100%) since the previous snapshot; stablecoins skipped; days with 3+ jumps are
               flagged "batch" (probably a data update, tracked separately)
  paper trade  a hedged short for 14 days: short the coin, long an equal-weight basket of all tracked coins;
               result = basket return - coin return - 0.2% costs
  alerts       Telegram message per event (with a warning if the coin is one the trend strategy trades), one per
               closed paper trade, and a weekly score
  files        data/unlock_watch/events.csv, state.json; GET /api/unlock_watch/status for the plan page
"""

import csv
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger("layaquant.unlock_watch")

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "unlock_watch"))
SNAP_DIR = os.path.join(DATA, "snapshots")
STATE = os.path.join(DATA, "state.json")
EVENTS = os.path.join(DATA, "events.csv")
MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
PERPS = "https://fapi.binance.com/fapi/v1/exchangeInfo"
JUMP, HOLD_DAYS, BATCH, COST = 0.05, 14, 3, 0.002
TREND_COINS = {"BTC", "ETH", "SOL", "DOGE", "XRP", "BNB", "AVAX", "SUI"}
STABLE = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "BUSD", "USD1", "USDD", "FRAX", "RLUSD"}
EVENT_COLS = ["date", "coin", "symbol", "supply_jump", "batch", "trend_coin", "has_perp", "entry_price", "status",
              "exit_date", "excess_return"]


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class UnlockWatch:
    def __init__(self):
        os.makedirs(SNAP_DIR, exist_ok=True)
        self.state = {"last_day": None, "open": [], "closed": [], "basket_level": 1.0, "last_week_report": None}
        if os.path.exists(STATE):
            try:
                self.state.update(json.load(open(STATE)))
            except Exception as e:
                logger.warning(f"unlock_watch state unreadable: {e}")
        self._thread: Optional[threading.Thread] = None

    # ---------- data ----------
    def fetch(self) -> list:
        for k in range(4):
            try:
                r = requests.get(MARKETS, params={"vs_currency": "usd", "per_page": 250, "page": 1}, timeout=30)
                if r.status_code == 200:
                    return [{"id": c["id"], "symbol": c["symbol"].upper(), "price": c["current_price"],
                             "supply": c["circulating_supply"], "mcap": c["market_cap"]}
                            for c in r.json() if c.get("current_price") and c.get("circulating_supply")]
            except requests.RequestException:
                pass
            time.sleep(30 * (k + 1))
        return []

    def perp_bases(self) -> Optional[set]:
        try:
            info = requests.get(PERPS, timeout=20).json()
            return {s["baseAsset"].replace("1000000", "").replace("1000", "") for s in info["symbols"]
                    if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"}
        except Exception:
            return None

    def _previous_snapshot(self, day: str) -> Optional[dict]:
        files = sorted(f for f in os.listdir(SNAP_DIR) if f.endswith(".json") and f[:10] < day)
        return json.load(open(os.path.join(SNAP_DIR, files[-1]))) if files else None

    def _save(self):
        json.dump(self.state, open(STATE, "w"), indent=1)

    def _write_events(self):
        rows = self.state["closed"] + self.state["open"]
        with open(EVENTS, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVENT_COLS, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=lambda r: r["date"]):
                w.writerow(r)

    # ---------- daily step ----------
    def run_day(self, snapshot: Optional[list] = None, day: Optional[str] = None, perps: Optional[set] = "fetch"):
        from backend.notifier import notifier
        day = day or _today()
        snap = snapshot if snapshot is not None else self.fetch()
        if not snap:
            logger.warning("unlock_watch: no CoinGecko data today")
            return
        json.dump({"day": day, "coins": snap}, open(os.path.join(SNAP_DIR, f"{day}.json"), "w"))
        prev = self._previous_snapshot(day)
        self.state["last_day"] = day
        if prev is None:
            self._save()
            return
        now = {c["id"]: c for c in snap}
        before = {c["id"]: c for c in prev["coins"]}
        common = [i for i in now if i in before and before[i]["price"]]
        rets = [now[i]["price"] / before[i]["price"] - 1 for i in common]
        basket_ret = float(np.mean(rets)) if rets else 0.0
        self.state["basket_level"] *= 1 + basket_ret

        # 1 update / close paper trades
        still = []
        for t in self.state["open"]:
            c = now.get(t["coin"])
            if c:
                t["last_price"] = c["price"]
            coin_ret = t.get("last_price", t["entry_price"]) / t["entry_price"] - 1
            basket = self.state["basket_level"] / t["entry_basket"] - 1
            t["excess_return"] = round(basket - coin_ret - COST, 4)
            if (datetime.fromisoformat(day) - datetime.fromisoformat(t["date"])).days >= HOLD_DAYS:
                t["status"], t["exit_date"] = "closed", day
                self.state["closed"].append(t)
                notifier.send(f"🔓 Unlock watch: {t['symbol']} paper hedged short closed after {HOLD_DAYS} days: "
                              f"{100 * t['excess_return']:+.1f}% vs the market{' (batch day)' if t['batch'] else ''}.")
            else:
                still.append(t)
        self.state["open"] = still

        # 2 new supply jumps
        perps = self.perp_bases() if perps == "fetch" else perps
        jumps = []
        for i in common:
            b, n = before[i], now[i]
            if n["symbol"] in STABLE or not b["supply"]:
                continue
            j = n["supply"] / b["supply"] - 1
            if JUMP <= j < 1.0:
                jumps.append((i, n, j))
        batch = len(jumps) >= BATCH
        held = {t["coin"] for t in self.state["open"]}
        for i, n, j in jumps:
            if i in held:
                continue
            t = {"date": day, "coin": i, "symbol": n["symbol"], "supply_jump": round(j, 4), "batch": batch,
                 "trend_coin": n["symbol"] in TREND_COINS, "has_perp": (perps is None) or (n["symbol"] in perps),
                 "entry_price": n["price"], "last_price": n["price"], "entry_basket": self.state["basket_level"],
                 "status": "open", "exit_date": "", "excess_return": 0.0}
            self.state["open"].append(t)
            warn = "\n⚠️ Your trend strategy trades this coin: think twice before a new long." if t["trend_coin"] else ""
            if not batch or t["trend_coin"]:
                notifier.send(f"🔓 Unlock watch: {n['symbol']} circulating supply +{100 * j:.1f}% since the last check "
                              f"(new tokens entering the market). Paper hedged short opened for {HOLD_DAYS} days."
                              f"{'' if t['has_perp'] else ' (no Binance perp)'}{warn}")
        if batch:
            notifier.send(f"🔓 Unlock watch: {len(jumps)} coins' supply jumped on {day}, probably a CoinGecko data "
                          f"update; tracked separately.")

        # 3 weekly score (Sunday)
        week = datetime.fromisoformat(day).strftime("%G-W%V")
        if datetime.fromisoformat(day).weekday() == 6 and self.state.get("last_week_report") != week:
            self.state["last_week_report"] = week
            s = self.score()
            if s["n"]:
                notifier.send(f"🔓 Unlock watch weekly: {s['n']} closed paper trades (excluding batch days), average "
                              f"{100 * s['mean']:+.1f}% vs the market, {s['win']:.0%} winners, t {s['t']:+.2f}. "
                              f"{len(self.state['open'])} open.")
        self._save()
        self._write_events()

    def score(self, include_batch: bool = False) -> dict:
        x = [t["excess_return"] for t in self.state["closed"] if include_batch or not t["batch"]]
        if not x:
            return {"n": 0, "mean": 0.0, "t": 0.0, "win": 0.0}
        a = np.array(x)
        t = float(a.mean() / a.std() * np.sqrt(len(a))) if len(a) > 1 and a.std() > 0 else 0.0
        return {"n": len(a), "mean": float(a.mean()), "t": t, "win": float((a > 0).mean())}

    def status(self) -> dict:
        recent = sorted(self.state["closed"] + self.state["open"], key=lambda r: r["date"], reverse=True)[:30]
        return {"last_check": self.state["last_day"], "open": len(self.state["open"]), "score": self.score(),
                "score_all": self.score(include_batch=True), "events": recent,
                "file": os.path.relpath(EVENTS, os.path.dirname(os.path.dirname(DATA)))}

    # ---------- background loop ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            try:
                if self.state.get("last_day") != _today() and datetime.now(timezone.utc).hour >= 1:
                    self.run_day()
            except Exception as e:
                logger.warning(f"unlock_watch error: {e}")
            time.sleep(1800)


unlock_watch = UnlockWatch()

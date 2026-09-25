"""
News safety brake.

Laya reads every new headline the news engine pulls and answers yes/no questions for events that
can crash a coin or the whole market: hacks/exploits, exchanges or lenders halting withdrawals,
delistings, regulator/prosecutor action, stablecoin depegs and chain outages.

- Exchange failures and depegs block new trend entries on every coin.
- Other events block the coins the headline names (Bitcoin/BTC, Solana/SOL, ...).
- A block lasts BLOCK_HOURS from the headline's publication time. Open positions are not touched.

Every verdict is appended to data/news_guard_log.jsonl with the coins' prices at that moment, so
the brake can be judged later on what actually followed the flagged headlines:

    python -m backend.news_guard --report          # forward returns after flagged headlines
    python -m backend.news_guard --test "FTX halts withdrawals"

Without Laya (not installed / failed to load) it falls back to keyword rules and says so in the log.
"""

import json
import logging
import os
import re
import sys
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger("layaquant.news_guard")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
LOG_FILE = os.path.join(DATA_DIR, "news_guard_log.jsonl")

THRESHOLD = 0.70          # Laya "yes" probability needed to flag
BLOCK_HOURS = 24.0
MAX_AGE_HOURS = 48.0      # ignore headlines older than this (feeds re-serve old items)
POLL_SECONDS = 30.0
SKIP_SOURCES = {"Binance Market Pulse"}   # the news engine's own price-move items, not news

QUESTIONS = {
    "hack": {"type": "noul", "instructions": "Does this headline report that a crypto project, protocol, bridge, wallet or exchange was hacked, exploited or had funds stolen?"},
    "exchange_failure": {"type": "noul", "instructions": "Does this headline report that a crypto exchange, lender or platform halted or paused withdrawals, froze funds, or became insolvent or bankrupt?"},
    "delisting": {"type": "noul", "instructions": "Does this headline report that a token is being delisted or its trading suspended?"},
    "regulatory": {"type": "noul", "instructions": "Does this headline report that a regulator, prosecutor, court or government is suing, charging, fining, banning or sanctioning a crypto company, person or token?"},
    "depeg": {"type": "noul", "instructions": "Does this headline report that a stablecoin lost its dollar peg?"},
    "outage": {"type": "noul", "instructions": "Does this headline report that a blockchain network halted, stopped producing blocks or suffered a major outage?"},
}
MARKET_EVENTS = {"exchange_failure", "depeg"}
# These events also need one of these words: in live tests Laya called "dollar debt crisis" and
# "largest Bitcoin outflow" depegs, and "Binance unveils account overhaul" an outage.
EVENT_GATE = {
    "depeg": r"\b(stablecoins?|usdt|tether|usdc|circle|dai|ust|terrausd|busd|fdusd|usde|ethena|pyusd|peg|depegs?|de-peg)\b",
    "exchange_failure": r"(withdraw|insolven|bankrupt|freez|frozen|\bhalts?\b|\bhalted\b|\bpaus)",
    "outage": r"(outage|\bhalt|\bdown\b|stopped|stall|offline|block production)",
}

KEYWORDS = {   # fallback when Laya is unavailable
    "hack": ["hacked", "hack", "exploit", "exploited", "drained", "stolen"],
    "exchange_failure": ["halts withdrawals", "pauses withdrawals", "suspends withdrawals", "freezes withdrawals",
                         "insolvent", "insolvency", "files for bankruptcy", "bankrupt"],
    "delisting": ["delist", "delisting", "delisted"],
    "regulatory": ["sues", "sued", "charges", "charged", "indicted", "sanctions", "sanctioned", "banned"],
    "depeg": ["depeg", "depegs", "de-peg", "loses peg", "lost its peg", "loses dollar peg"],
    "outage": ["outage", "halted block production", "network halted", "stops producing blocks"],
}

ALIASES = {
    "BTCUSDT": ["bitcoin", "btc"],
    "ETHUSDT": ["ethereum", "ether", "eth"],
    "SOLUSDT": ["solana", "sol"],
    "BNBUSDT": ["bnb", "binance", "bsc"],
    "XRPUSDT": ["xrp", "ripple"],
    "DOGEUSDT": ["dogecoin", "doge"],
    "AVAXUSDT": ["avalanche", "avax"],
    "SUIUSDT": ["sui"],
}
EXCHANGE_ALIASES = {"binance"}   # "Binance delists X" is not news about BNB
DEFAULT_UNIVERSE = list(ALIASES)


def _aliases(sym: str) -> List[str]:
    return ALIASES.get(sym, [sym.replace("USDT", "").lower()])


def mentioned_symbols(text: str, universe: List[str], exclude=()) -> List[str]:
    low = text.lower()
    return [s for s in universe
            if any(re.search(rf"\b{re.escape(a)}\b", low) for a in _aliases(s) if a not in exclude)]


def _norm(headline: str) -> str:
    return re.sub(r"\s+", " ", headline.strip().lower())


class NewsGuard:
    def __init__(self, universe: Optional[List[str]] = None):
        self.universe = list(universe or DEFAULT_UNIVERSE)
        self.agent = None
        self.lock = threading.Lock()
        self.seen: set = set()
        self.blocks: List[dict] = []      # {"scope": "*" or symbol, "until": ts, "reason": str}
        self.recent: List[dict] = []      # last verdicts, newest first (for the API)
        self._thread = None
        self._load_log()

    # ---------- setup ----------
    def attach_agent(self, agent):
        """Use an already-loaded Laya agent (server.py loads one at startup)."""
        self.agent = agent
        logger.info("News guard: using Laya for headline classification.")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _load_log(self):
        """Rebuild seen-set and still-active blocks after a restart."""
        if not os.path.exists(LOG_FILE):
            return
        now = time.time()
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                self.seen.add(_norm(rec["headline"]))
                self.recent.insert(0, rec)
                if rec.get("flagged"):
                    self._add_block(rec, now)
        self.recent = self.recent[:50]

    # ---------- classification ----------
    def classify(self, headline: str) -> dict:
        if self.agent is not None:
            try:
                with self.lock:
                    ans = self.agent.system_one(headline, QUESTIONS)["answers"]
                return {"engine": "laya", "scores": {k: round(float(v["noul"]), 3) for k, v in ans.items()}}
            except Exception as e:
                logger.warning(f"News guard: Laya failed ({e}); using keyword rules for this headline.")
        low = headline.lower()
        scores = {k: float(any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)) for k, words in KEYWORDS.items()}
        return {"engine": "keywords", "scores": scores}

    def evaluate(self, headline: str) -> dict:
        """Classify a headline and decide what it blocks (does not log or apply)."""
        res = self.classify(headline)
        low = headline.lower()
        hits = sorted(((p, e) for e, p in res["scores"].items()
                       if p >= THRESHOLD and (e not in EVENT_GATE or re.search(EVENT_GATE[e], low))), reverse=True)
        market = [(p, e) for p, e in hits if e in MARKET_EVENTS]
        coins: List[str] = []
        for p, e in hits:
            if e not in MARKET_EVENTS:
                exclude = EXCHANGE_ALIASES if e == "delisting" else ()
                coins += [s for s in mentioned_symbols(headline, self.universe, exclude) if s not in coins]
        if market:
            (p, event), scope = market[0], "market"
        elif hits:
            (p, event), scope = hits[0], coins
        else:
            event, p = max(res["scores"].items(), key=lambda kv: kv[1])
            return {**res, "event": None, "prob": round(p, 3), "flagged": False, "scope": []}
        return {**res, "event": event, "prob": round(p, 3), "flagged": bool(scope), "scope": scope}

    # ---------- live loop ----------
    def _loop(self):
        time.sleep(5.0)
        while True:
            try:
                self.scan_once()
            except Exception as e:
                logger.debug(f"News guard scan error: {e}")
            time.sleep(POLL_SECONDS)

    def scan_once(self) -> int:
        from backend.news_engine import news_engine
        with news_engine.lock:
            items = list(news_engine.headlines)
            prices = {s: v.get("price") for s, v in news_engine.ticker_24h_pulses.items()}
        now = time.time()
        n = 0
        for it in items:
            headline = it.get("headline", "")
            key = _norm(headline)
            if not headline or key in self.seen or it.get("source") in SKIP_SOURCES:
                continue
            self.seen.add(key)
            published = int(it.get("timestamp") or now)
            if now - published > MAX_AGE_HOURS * 3600:
                continue
            verdict = self.evaluate(headline)
            rec = {"logged_at": int(now), "published": published, "source": it.get("source", ""),
                   "headline": headline, **verdict, "prices": prices}
            self._append_log(rec)
            self.recent.insert(0, rec)
            self.recent = self.recent[:50]
            if rec["flagged"]:
                self._add_block(rec, now)
                logger.warning(f"[NEWS BRAKE] {verdict['event']} ({verdict['prob']:.2f}) -> blocks "
                               f"{verdict['scope']} for {BLOCK_HOURS:.0f}h: {headline}")
            n += 1
        return n

    def _append_log(self, rec: dict):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _add_block(self, rec: dict, now: float):
        until = rec["published"] + BLOCK_HOURS * 3600
        if until <= now:
            return
        reason = f"{rec['event']} ({rec['prob']:.2f}): {rec['headline'][:90]}"
        scopes = ["*"] if rec["scope"] == "market" else rec["scope"]
        for s in scopes:
            self.blocks.append({"scope": s, "until": until, "reason": reason})

    # ---------- queries ----------
    def block_reason(self, symbol: str) -> Optional[str]:
        now = time.time()
        self.blocks = [b for b in self.blocks if b["until"] > now]
        for b in self.blocks:
            if b["scope"] in ("*", symbol):
                return b["reason"]
        return None

    def status(self) -> dict:
        now = time.time()
        return {
            "engine": "laya" if self.agent is not None else "keywords",
            "threshold": THRESHOLD,
            "block_hours": BLOCK_HOURS,
            "active_blocks": [{**b, "hours_left": round((b["until"] - now) / 3600, 1)}
                              for b in self.blocks if b["until"] > now],
            "recent": self.recent[:20],
        }


news_guard = NewsGuard()


# ---------- offline evaluation ----------
def _hourly_closes(symbol: str, start_s: int, end_s: int) -> Dict[int, float]:
    import requests
    out, cur = {}, start_s * 1000
    while cur < end_s * 1000:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": symbol, "interval": "1h", "startTime": cur, "limit": 1000}, timeout=15)
        rows = r.json()
        if not rows:
            break
        for k in rows:
            out[int(k[0]) // 1000] = float(k[4])
        cur = int(rows[-1][0]) + 3_600_000
    return out


def report():
    """Forward returns of the blocked coins after each flagged headline vs. their average over the log period."""
    if not os.path.exists(LOG_FILE):
        print(f"No log yet at {LOG_FILE}. Run the server for a while first.")
        return
    recs = [json.loads(l) for l in open(LOG_FILE, encoding="utf-8") if l.strip()]
    flagged = [r for r in recs if r.get("flagged")]
    print(f"{len(recs)} headlines logged, {len(flagged)} flagged.")
    if not flagged:
        return
    t0 = min(r["published"] for r in recs) // 3600 * 3600
    t1 = int(time.time())
    closes = {s: _hourly_closes(s, t0, t1) for s in DEFAULT_UNIVERSE}
    horizons = (4, 24, 72)

    def fwd(sym, t, h):
        a, b = closes[sym].get(t), closes[sym].get(t + h * 3600)
        return None if a is None or b is None else b / a - 1

    print(f"\n{'published (UTC)':<17} {'event':<17} {'scope':<22} " + " ".join(f"{f'+{h}h':>7}" for h in horizons) + "  headline")
    sums = {h: [] for h in horizons}
    for r in flagged:
        t = r["published"] // 3600 * 3600 + 3600       # first full hour after the headline
        syms = DEFAULT_UNIVERSE if r["scope"] == "market" else r["scope"]
        cells = []
        for h in horizons:
            vals = [v for v in (fwd(s, t, h) for s in syms) if v is not None]
            m = sum(vals) / len(vals) if vals else None
            if m is not None:
                sums[h].append(m)
            cells.append(f"{m:+7.1%}" if m is not None else "      -")
        scope = "market" if r["scope"] == "market" else ",".join(s.replace("USDT", "") for s in syms)
        when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(r["published"]))
        print(f"{when:<17} {r['event']:<17} {scope:<22} " + " ".join(cells) + f"  {r['headline'][:70]}")

    base = {}
    for h in horizons:
        vals = [fwd(s, t, h) for s in DEFAULT_UNIVERSE for t in closes[s]]
        vals = [v for v in vals if v is not None]
        base[h] = sum(vals) / len(vals) if vals else 0.0
    print("\nAverage after flagged headlines: " + "  ".join(
        f"+{h}h {sum(v) / len(v):+.2%} (n={len(v)})" if v else f"+{h}h -" for h, v in sums.items()))
    print("Average at any hour (same coins, same period): " + "  ".join(f"+{h}h {base[h]:+.2%}" for h in horizons))
    print("The brake helps only if flagged headlines are followed by clearly worse returns than usual.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    elif len(sys.argv) > 2 and sys.argv[1] == "--test":
        try:
            import laya
            news_guard.attach_agent(laya.load("convaiinnovations/laya"))
        except Exception as e:
            print(f"Laya unavailable ({e}); using keyword rules.")
        for h in sys.argv[2:]:
            print(json.dumps({"headline": h, **news_guard.evaluate(h)}, indent=2))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

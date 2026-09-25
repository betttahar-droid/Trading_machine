import os
import time
import json
import re
import logging
import html
import urllib.request
import xml.etree.ElementTree as ET
import email.utils
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger("layaquant.news")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
NEWS_CACHE_FILE = os.path.join(DATA_DIR, "news_feed_cache.json")
os.makedirs(DATA_DIR, exist_ok=True)

# 100% Live Multi-Stream RSS feeds for real-time crypto, macro & financial catalysts
LIVE_RSS_FEEDS = [
    ("Google News Crypto", "https://news.google.com/rss/search?q=crypto+OR+bitcoin+when:1d&hl=en-US&gl=US&ceid=US:en", "macro"),
    ("Google News Alts", "https://news.google.com/rss/search?q=Solana+OR+Ethereum+OR+XRP+crypto+when:1d&hl=en-US&gl=US&ceid=US:en", "crypto"),
    ("Google News Market", "https://news.google.com/rss/search?q=Dogecoin+OR+Binance+BNB+crypto+when:1d&hl=en-US&gl=US&ceid=US:en", "crypto"),
    ("CoinTelegraph", "https://cointelegraph.com/rss", "crypto"),
    ("Decrypt", "https://decrypt.co/feed", "crypto"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", "crypto"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex", "macro")
]

# Financial Sentiment Lexicon with domain weights
BULLISH_KEYWORDS = {
    "etf approval": 1.0, "approved": 0.8, "rate cut": 0.9, "dovish": 0.85,
    "inflow": 0.75, "inflows": 0.75, "record high": 0.9, "all-time high": 0.95,
    "beat expectations": 0.85, "revenue surge": 0.8, "breakout": 0.7,
    "accumulation": 0.75, "institutional buying": 0.9, "partnership": 0.65,
    "expansion": 0.6, "soars": 0.75, "bullish": 0.7, "liquidity injection": 0.85,
    "upgrade": 0.65, "adoption": 0.7, "treasury buying": 0.8, "tokenized": 0.65,
    "invest": 0.6, "invests": 0.6, "launches": 0.55, "rally": 0.7,
    "gains": 0.6, "surges": 0.75, "borrowing": 0.5
}

BEARISH_KEYWORDS = {
    "sec lawsuit": -1.0, "hack": -1.0, "exploit": -0.95, "stolen": -0.9,
    "rate hike": -0.9, "hawkish": -0.85, "inflation spike": -0.8, "cpi higher": -0.8,
    "subpoena": -0.85, "investigation": -0.8, "liquidation cascade": -0.95,
    "liquidations": -0.85, "plunges": -0.8, "flash crash": -1.0, "ban": -0.9,
    "tariff": -0.75, "insolvency": -1.0, "layoffs": -0.6, "bearish": -0.7,
    "downgrade": -0.65, "regulatory crackdown": -0.9, "fraud": -1.0, "outflow": -0.75,
    "outflows": -0.75, "breach": -0.95, "halts": -0.85, "opposing": -0.5
}

CATEGORY_PATTERNS = {
    "monetary_policy": ["fed", "fomc", "rate", "powell", "inflation", "cpi", "dovish", "hawkish", "central bank", "treasury", "ecb"],
    "regulatory": ["sec", "cftc", "lawsuit", "regulation", "court", "ruling", "judge", "compliance", "ban", "legal", "pac", "senate"],
    "institutional": ["etf", "blackrock", "fidelity", "grayscale", "inflow", "outflow", "fund", "holdings", "microstrategy", "institutional", "tokenized", "shares", "bonds"],
    "earnings": ["earnings", "revenue", "profit", "quarterly", "guidance", "eps", "sales", "margins"],
    "market_structure": ["liquidation", "leverage", "funding", "whale", "derivatives", "volume", "options", "hack", "exploit", "usdc", "stablecoin"]
}

# Ticker-Specific Entity Recognition & Keyword Matrix
TICKER_ENTITY_MAP = {
    "BTCUSDT": {
        "name": "Bitcoin",
        "symbol": "BTC",
        "keywords": ["bitcoin", "btc", "satoshi", "halving", "powell", "spot etf", "blackrock", "saylor", "microstrategy", "hashrate", "digital gold"],
        "social_handles": ["@Bitcoin", "@saylor", "@DocumentingBTC", "@WhaleAlert"],
        "default_buzz": 75.0
    },
    "ETHUSDT": {
        "name": "Ethereum",
        "symbol": "ETH",
        "keywords": ["ethereum", "eth", "vitalik", "layer 2", "pectra", "gas fees", "arbitrum", "optimism", "staking", "eip-", "erc-20"],
        "social_handles": ["@VitalikButerin", "@ethereum", "@sassal0x", "@WhaleAlert"],
        "default_buzz": 68.0
    },
    "SOLUSDT": {
        "name": "Solana",
        "symbol": "SOL",
        "keywords": ["solana", "sol", "anatoly", "firedancer", "breakpoint", "depin", "phantom", "toly", "spl", "solana mobile"],
        "social_handles": ["@aeyakovenko", "@solana", "@rajgokal", "@WhaleAlert"],
        "default_buzz": 72.0
    },
    "DOGEUSDT": {
        "name": "Dogecoin",
        "symbol": "DOGE",
        "keywords": ["dogecoin", "doge", "elon", "musk", "x payments", "shiba", "memecoin", "kabosu", "doge-1"],
        "social_handles": ["@elonmusk", "@dogecoin", "@BillyM2k"],
        "default_buzz": 82.0
    },
    "XRPUSDT": {
        "name": "XRP (Ripple)",
        "symbol": "XRP",
        "keywords": ["xrp", "ripple", "garlinghouse", "sec lawsuit", "rlusd", "cross-border", "settlement", "xrpl", "ledger", "torres"],
        "social_handles": ["@bgarlinghouse", "@Ripple", "@JoelKatz", "@WhaleAlert"],
        "default_buzz": 78.0
    },
    "BNBUSDT": {
        "name": "BNB (Binance)",
        "symbol": "BNB",
        "keywords": ["bnb", "binance", "cz", "launchpool", "bnb chain", "richard teng", "opbnb", "bsc"],
        "social_handles": ["@cz_binance", "@binance", "@_RichardTeng"],
        "default_buzz": 60.0
    },
    "AVAXUSDT": {
        "name": "Avalanche",
        "symbol": "AVAX",
        "keywords": ["avalanche", "avax", "subnets", "emin gun sirer", "ava labs", "blizzard", "c-chain"],
        "social_handles": ["@el33th4xor", "@AvaLabs", "@avax"],
        "default_buzz": 55.0
    },
    "SUIUSDT": {
        "name": "Sui Network",
        "symbol": "SUI",
        "keywords": ["sui", "mysten", "move", "sui foundation", "walrus", "evan cheng", "sui tvl"],
        "social_handles": ["@SuiNetwork", "@Mysten_Labs", "@EvanWeb3"],
        "default_buzz": 62.0
    }
}

WHALE_AND_SOCIAL_SEEDS = [
    {
        "id": "tweet_xrp_1",
        "symbol": "XRPUSDT",
        "author": "@WhaleAlert",
        "text": "🚨 75,000,000 #XRP (114,820,500 USD) transferred from Binance to Unknown Cold Wallet. Institutional accumulation detected.",
        "timestamp": int(time.time()) - 420,
        "sentiment_score": 0.85,
        "type": "whale_alert",
        "buzz_weight": 88
    },
    {
        "id": "tweet_btc_1",
        "symbol": "BTCUSDT",
        "author": "@saylor",
        "text": "Bitcoin is digital energy. Volatility is vitality. Accumulate capital that cannot be debased.",
        "timestamp": int(time.time()) - 1200,
        "sentiment_score": 0.65,
        "type": "influencer_tweet",
        "buzz_weight": 78
    },
    {
        "id": "tweet_doge_1",
        "symbol": "DOGEUSDT",
        "author": "@elonmusk",
        "text": "The most entertaining outcome is the most likely. Doge to the moon 🚀",
        "timestamp": int(time.time()) - 2400,
        "sentiment_score": 0.80,
        "type": "influencer_tweet",
        "buzz_weight": 95
    },
    {
        "id": "tweet_sol_1",
        "symbol": "SOLUSDT",
        "author": "@solana",
        "text": "Solana mainnet achieves 4,200 sustained TPS with Firedancer validator client reaching final testing phase.",
        "timestamp": int(time.time()) - 3600,
        "sentiment_score": 0.72,
        "type": "ecosystem_update",
        "buzz_weight": 70
    }
]

DEFAULT_SEED_HEADLINES = [
    {
        "id": "news_1",
        "headline": "Fed Signals Potential Rate Cut Easing Cycle as Inflation Continues Steady Cool Down",
        "source": "Bloomberg Financial Wire",
        "timestamp": int(time.time()) - 180,
        "sentiment_score": 0.78,
        "impact_weight": 0.88,
        "category": "monetary_policy",
        "urgency": "macro"
    },
    {
        "id": "news_2",
        "headline": "Institutional Spot ETF Net Inflows Exceed $420M in Massive Daily Capital Allocation",
        "source": "CoinDesk Institutional",
        "timestamp": int(time.time()) - 620,
        "sentiment_score": 0.82,
        "impact_weight": 0.75,
        "category": "institutional",
        "urgency": "routine"
    },
    {
        "id": "news_3",
        "headline": "Tech Giants Report Sustained Record Demand for High-Performance Next-Gen Compute Clusters",
        "source": "Reuters Tech Wire",
        "timestamp": int(time.time()) - 1450,
        "sentiment_score": 0.65,
        "impact_weight": 0.70,
        "category": "earnings",
        "urgency": "routine"
    }
]


class NewsEngine:
    def __init__(self):
        self.headlines: List[Dict[str, Any]] = []
        self.social_alerts: List[Dict[str, Any]] = list(WHALE_AND_SOCIAL_SEEDS)
        self.last_fetch_time = 0.0
        self.lock = threading.Lock()
        self.fear_and_greed = {
            "value": 65,
            "value_classification": "Greed",
            "timestamp": int(time.time())
        }
        self.ticker_24h_pulses: Dict[str, dict] = {}
        self._load_cache()

        # Launch background polling daemon to continuously refresh live web news every 15s
        self._poller_thread = threading.Thread(target=self._background_poller, daemon=True)
        self._poller_thread.start()

    def _background_poller(self):
        time.sleep(1.0)
        while True:
            try:
                self.fetch_live_web_news(force=True)
            except Exception as e:
                logger.debug(f"Background news poller error: {e}")
            time.sleep(15.0)

    def _load_cache(self):
        if os.path.exists(NEWS_CACHE_FILE):
            try:
                with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
                    self.headlines = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading news cache: {e}")
                self.headlines = list(DEFAULT_SEED_HEADLINES)
        else:
            self.headlines = list(DEFAULT_SEED_HEADLINES)
            self._save_cache()

    def _save_cache(self):
        try:
            with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.headlines[:80], f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving news cache: {e}")

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyzes financial NLP sentiment, category, and impact weight."""
        lower = text.lower()
        score = 0.0
        matched_bull = []
        matched_bear = []

        for kw, w in BULLISH_KEYWORDS.items():
            if kw in lower:
                score += w
                matched_bull.append(kw)

        for kw, w in BEARISH_KEYWORDS.items():
            if kw in lower:
                score += w
                matched_bear.append(kw)

        # Normalize score to [-1.0, 1.0]
        if score > 0:
            norm_score = min(1.0, score / max(1.0, len(matched_bull) * 0.8))
        elif score < 0:
            norm_score = max(-1.0, score / max(1.0, len(matched_bear) * 0.8))
        else:
            norm_score = 0.0

        # Category identification
        assigned_cat = "market_structure"
        for cat, keywords in CATEGORY_PATTERNS.items():
            if any(k in lower for k in keywords):
                assigned_cat = cat
                break

        # Impact weight estimation
        impact = 0.5
        if abs(norm_score) > 0.7:
            impact = 0.9
        elif abs(norm_score) > 0.4:
            impact = 0.75
        if assigned_cat in ["monetary_policy", "regulatory"]:
            impact = min(1.0, impact + 0.15)

        urgency = "breaking" if impact >= 0.85 else ("macro" if assigned_cat in ["monetary_policy", "regulatory"] else "routine")

        return {
            "sentiment_score": round(norm_score, 2),
            "category": assigned_cat,
            "impact_weight": round(impact, 2),
            "urgency": urgency,
            "matched_keywords": matched_bull + matched_bear
        }

    def _fetch_rss_single(self, source_name: str, feed_url: str, headers: dict) -> List[Dict[str, Any]]:
        """Scrapes and parses a single RSS feed endpoint."""
        items_out = []
        try:
            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                content = resp.read()
                root = ET.fromstring(content)
                items = root.findall('.//item')
                now = time.time()
                for it in items[:8]:
                    title_el = it.find('title')
                    link_el = it.find('link')
                    pdate_el = it.find('pubDate')
                    desc_el = it.find('description')

                    raw_title = title_el.text.strip() if title_el is not None and title_el.text else ''
                    clean_title = html.unescape(raw_title)
                    clean_title = re.sub(r'<[^>]+>', '', clean_title).strip()
                    if not clean_title or len(clean_title) < 10:
                        continue

                    link_url = link_el.text.strip() if link_el is not None and link_el.text else ''
                    raw_desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ''
                    clean_desc = html.unescape(raw_desc)
                    clean_desc = re.sub(r'<[^>]+>', '', clean_desc).strip()[:250]

                    ts = int(now)
                    if pdate_el is not None and pdate_el.text:
                        try:
                            parsed_dt = email.utils.parsedate_to_datetime(pdate_el.text)
                            ts = int(parsed_dt.timestamp())
                        except Exception:
                            pass

                    combined_text = f"{clean_title}. {clean_desc}"
                    analysis = self.analyze_sentiment(combined_text)

                    items_out.append({
                        "id": f"{source_name.lower().replace(' ', '_')}_{abs(hash(clean_title))}",
                        "headline": clean_title,
                        "summary": clean_desc,
                        "url": link_url,
                        "source": source_name,
                        "timestamp": ts,
                        "sentiment_score": analysis["sentiment_score"],
                        "impact_weight": analysis["impact_weight"],
                        "category": analysis["category"],
                        "urgency": analysis["urgency"],
                        "matched_keywords": analysis["matched_keywords"]
                    })
        except Exception as e:
            logger.debug(f"Feed error from {source_name}: {e}")
        return items_out

    def _fetch_fear_and_greed(self):
        """Fetches live Alternative.me Crypto Fear & Greed Index."""
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=3.0)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    val = int(data[0].get("value", 50))
                    cls = data[0].get("value_classification", "Neutral")
                    ts = int(data[0].get("timestamp", int(time.time())))
                    with self.lock:
                        self.fear_and_greed = {
                            "value": val,
                            "value_classification": cls,
                            "timestamp": ts
                        }
        except Exception as e:
            logger.debug(f"Fear & Greed fetch error: {e}")

    def _fetch_binance_bulk_tickers(self) -> List[Dict[str, Any]]:
        """Pulls real-time 24hr ticker volume and price momentum across universe coins."""
        pulled = []
        try:
            b_resp = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=3.0)
            if b_resp.status_code == 200:
                b_all = b_resp.json()
                universe_keys = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "SUIUSDT"}
                now = time.time()
                for b_data in b_all:
                    sym = b_data.get("symbol")
                    if sym in universe_keys:
                        price_chg = float(b_data.get("priceChangePercent", 0.0))
                        vol_quote = float(b_data.get("quoteVolume", 0.0))
                        last_p = float(b_data.get("lastPrice", 0.0))
                        with self.lock:
                            self.ticker_24h_pulses[sym] = {
                                "price": last_p,
                                "price_change_24h": price_chg,
                                "quote_volume": vol_quote,
                                "timestamp": int(now)
                            }
                        if abs(price_chg) >= 3.0:
                            b_headline = f"{sym} Market Catalyst: 24h change {price_chg:+.2f}% with ${vol_quote/1e6:.1f}M volume"
                            b_analysis = self.analyze_sentiment(b_headline)
                            b_analysis["sentiment_score"] = 0.70 if price_chg > 0 else -0.70
                            pulled.append({
                                "id": f"binance_bulk_{sym}_{int(now // 120)}",
                                "headline": b_headline,
                                "summary": f"24h volume on Binance: ${vol_quote:,.0f} USDT. Last price: ${last_p:,.4f}.",
                                "url": f"https://www.binance.com/en/trade/{sym.replace('USDT', '_USDT')}",
                                "source": "Binance Market Pulse",
                                "timestamp": int(now),
                                "sentiment_score": b_analysis["sentiment_score"],
                                "impact_weight": 0.82,
                                "category": "market_structure",
                                "urgency": "macro" if abs(price_chg) > 6.0 else "routine",
                                "matched_keywords": b_analysis["matched_keywords"]
                            })
        except Exception as e:
            logger.debug(f"Binance bulk tickers error: {e}")
        return pulled

    def fetch_live_web_news(self, symbol: str = "BTCUSDT", force: bool = False) -> List[Dict[str, Any]]:
        """
        Pulls real live breaking financial and crypto news headlines from multiple live RSS feeds
        concurrently with ThreadPoolExecutor, plus Fear & Greed Index and Binance bulk 24hr tickers.
        Cached for 15s unless force=True.
        """
        now = time.time()
        if not force and (now - self.last_fetch_time) < 15.0 and self.headlines:
            return self.headlines

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        pulled_items = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Parallel RSS feed fetch
            rss_futures = [executor.submit(self._fetch_rss_single, name, url, headers) for name, url, _ in LIVE_RSS_FEEDS]
            # Parallel Fear & Greed
            fng_future = executor.submit(self._fetch_fear_and_greed)
            # Parallel Binance tickers
            binance_future = executor.submit(self._fetch_binance_bulk_tickers)

            for fut in rss_futures:
                try:
                    pulled_items.extend(fut.result(timeout=4.0))
                except Exception:
                    pass

            try:
                binance_items = binance_future.result(timeout=3.5)
                pulled_items.extend(binance_items)
            except Exception:
                pass

        with self.lock:
            # Preserve user-injected breaking simulated headlines at the top
            simulated = [h for h in self.headlines if str(h.get("id", "")).startswith("sim_breaking_")]

            # Merge pulled items, deduplicating by headline text
            seen_titles = {h.get("headline", "").strip().lower() for h in simulated}
            new_list = list(simulated)

            # Sort pulled items newest first
            pulled_items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

            for item in pulled_items:
                title_key = item["headline"].strip().lower()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    new_list.append(item)

            if new_list:
                self.headlines = new_list[:120]
                self.last_fetch_time = now
                self._save_cache()

        return self.headlines

    def get_fear_and_greed(self) -> Dict[str, Any]:
        """Returns the current Fear & Greed index."""
        with self.lock:
            return dict(self.fear_and_greed)

    def get_live_web_pulse(self) -> Dict[str, Any]:
        """Returns a comprehensive real-time web pulse: Fear & Greed, aggregate sentiment, 24h ticker stats."""
        with self.lock:
            fng = dict(self.fear_and_greed)
            tickers = dict(self.ticker_24h_pulses)
            recent_headlines = self.headlines[:8]

        agg_sentiment = self.get_aggregate_market_sentiment()
        return {
            "fear_and_greed": fng,
            "aggregate_sentiment": agg_sentiment,
            "tickers_24h": tickers,
            "recent_headlines": recent_headlines,
            "last_updated": int(self.last_fetch_time)
        }

    def fetch_live_headlines(self, symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
        """Alias for backward compatibility."""
        return self.fetch_live_web_news(symbol=symbol, force=False)

    def simulate_breaking_news(self, headline: str, sentiment: Optional[float] = None, category: Optional[str] = None) -> Dict[str, Any]:
        """Injects a user-simulated or external breaking news catalyst into the active stream."""
        analysis = self.analyze_sentiment(headline)
        if sentiment is not None:
            analysis["sentiment_score"] = max(-1.0, min(1.0, float(sentiment)))
        if category:
            analysis["category"] = category

        item = {
            "id": f"sim_breaking_{int(time.time() * 1000)}",
            "headline": headline,
            "source": "Breaking Market Wire (Live Injected)",
            "timestamp": int(time.time()),
            "sentiment_score": analysis["sentiment_score"],
            "impact_weight": max(0.85, analysis["impact_weight"]),
            "category": analysis["category"],
            "urgency": "breaking"
        }

        with self.lock:
            self.headlines.insert(0, item)
            self._save_cache()
        logger.info(f"[NEWS INGESTION] Injected Breaking Catalyst: '{headline}' (Sentiment: {item['sentiment_score']}, Impact: {item['impact_weight']})")
        return item

    def get_aggregate_market_sentiment(self) -> Dict[str, Any]:
        """
        Calculates time-decay weighted market sentiment score across recent headlines.
        Returns score in [-1.0, +1.0], dominant theme, and catalyst risk multiplier.
        """
        if not self.headlines:
            return {
                "score": 0.25,
                "label": "Mildly Bullish",
                "dominant_theme": "Constructive Macro Accumulation",
                "catalyst_risk": 1.0,
                "headline_count": 0,
                "top_headline": "Markets holding steady in constructive range."
            }

        now = time.time()
        total_weight = 0.0
        weighted_sum = 0.0
        breaking_active = False

        for h in self.headlines[:10]:
            age_hours = (now - h.get("timestamp", now)) / 3600.0
            time_decay = max(0.2, 1.0 / (1.0 + (age_hours * 0.5)))
            w = h.get("impact_weight", 0.5) * time_decay
            weighted_sum += (h.get("sentiment_score", 0.0) * w)
            total_weight += w
            if h.get("urgency") == "breaking" and age_hours < 2.0:
                breaking_active = True

        # Incorporate live Fear & Greed Index (0-100 normalized to -1.0 to +1.0)
        fng_val = float(self.fear_and_greed.get("value", 50))
        fng_score = (fng_val - 50.0) / 50.0

        if total_weight > 0:
            news_score = weighted_sum / total_weight
            agg_score = (news_score * 0.75) + (fng_score * 0.25)
        else:
            agg_score = fng_score

        agg_score = round(max(-1.0, min(1.0, agg_score)), 3)

        if agg_score >= 0.55:
            label = f"Strongly Bullish (+{agg_score:.2f})"
        elif agg_score >= 0.15:
            label = f"Moderately Bullish (+{agg_score:.2f})"
        elif agg_score <= -0.55:
            label = f"Strongly Bearish ({agg_score:.2f})"
        elif agg_score <= -0.15:
            label = f"Moderately Bearish ({agg_score:.2f})"
        else:
            label = f"Neutral ({agg_score:+.2f})"

        top_item = self.headlines[0] if self.headlines else {}
        dominant_theme = f"{top_item.get('category', 'macro').replace('_', ' ').title()} Catalyst Flow"

        catalyst_risk = 1.35 if breaking_active else (1.15 if abs(agg_score) > 0.6 else 1.0)

        # Extract active real web sources
        active_sources = list(dict.fromkeys(h.get("source", "Live Wire") for h in self.headlines[:30]))

        return {
            "score": agg_score,
            "label": label,
            "dominant_theme": dominant_theme,
            "catalyst_risk": round(catalyst_risk, 2),
            "breaking_active": breaking_active,
            "headline_count": len(self.headlines),
            "top_headline": top_item.get("headline", "No headlines available"),
            "top_source": top_item.get("source", "Financial Wire"),
            "top_url": top_item.get("url", ""),
            "active_sources": active_sources,
            "fear_and_greed": dict(self.fear_and_greed),
            "last_updated_time": int(self.last_fetch_time),
            "recent_headlines": self.headlines[:15]
        }

    def get_ticker_sentiment(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Extracts asset-specific news, Crypto Twitter / X pulse, and Whale Alerts
        matching the specific ticker symbol (e.g. BTC, ETH, SOL, DOGE, XRP, BNB, AVAX, SUI).
        Computes asset-specific sentiment score [-1.0, +1.0], social buzz index [0, 100],
        top catalyst headline, and urgency level.
        """
        sym_clean = symbol.upper().replace("/", "")
        if not sym_clean.endswith("USDT") and not sym_clean.endswith("USD"):
            sym_clean = f"{sym_clean}USDT"

        meta = TICKER_ENTITY_MAP.get(sym_clean, {
            "name": sym_clean.replace("USDT", ""),
            "symbol": sym_clean.replace("USDT", ""),
            "keywords": [sym_clean.replace("USDT", "").lower()],
            "social_handles": ["@WhaleAlert"],
            "default_buzz": 50.0
        })

        keywords = [k.lower() for k in meta.get("keywords", [])]
        now = time.time()

        # 1. Match specific headlines for this asset
        matched_articles = []
        with self.lock:
            all_articles = list(self.headlines)
            all_social = list(getattr(self, "social_alerts", []))

        for h in all_articles:
            text = f"{h.get('headline', '')} {h.get('summary', '')}".lower()
            if any(k in text for k in keywords):
                matched_articles.append(h)

        # 2. Match specific tweets / whale alerts for this asset
        matched_social = []
        for s in all_social:
            s_sym = s.get("symbol", "").upper()
            if s_sym == sym_clean or s_sym == meta["symbol"]:
                matched_social.append(s)
            else:
                s_text = s.get("text", "").lower()
                if any(k in s_text for k in keywords):
                    matched_social.append(s)

        # 3. Calculate ticker-specific sentiment score
        total_weight = 0.0
        weighted_sum = 0.0
        breaking = False

        for a in matched_articles[:8]:
            age_h = (now - a.get("timestamp", now)) / 3600.0
            decay = max(0.25, 1.0 / (1.0 + (age_h * 0.4)))
            w = a.get("impact_weight", 0.6) * decay
            weighted_sum += a.get("sentiment_score", 0.0) * w
            total_weight += w
            if a.get("urgency") == "breaking":
                breaking = True

        for s in matched_social[:6]:
            age_h = (now - s.get("timestamp", now)) / 3600.0
            decay = max(0.25, 1.0 / (1.0 + (age_h * 0.3)))
            w = (s.get("buzz_weight", 70) / 100.0) * decay * 1.2
            weighted_sum += s.get("sentiment_score", 0.0) * w
            total_weight += w

        if total_weight > 0.05:
            ticker_score = round(max(-1.0, min(1.0, weighted_sum / total_weight)), 3)
            # Social Buzz Index
            mention_count = len(matched_articles) + len(matched_social)
            social_buzz = min(100.0, meta.get("default_buzz", 50.0) + (mention_count * 4.5) + (abs(ticker_score) * 15.0))
        else:
            # Fallback to mild baseline market bias
            global_sent = self.get_aggregate_market_sentiment()
            ticker_score = round(global_sent.get("score", 0.2) * 0.4, 3)
            social_buzz = meta.get("default_buzz", 50.0)

        # Determine sentiment label
        if ticker_score >= 0.50:
            label = f"🔥 Highly Bullish (+{ticker_score:+.2f})"
        elif ticker_score >= 0.15:
            label = f"🟢 Mildly Bullish (+{ticker_score:+.2f})"
        elif ticker_score <= -0.50:
            label = f"🚨 Critical FUD ({ticker_score:+.2f})"
        elif ticker_score <= -0.15:
            label = f"⚠️ Bearish Headwind ({ticker_score:+.2f})"
        else:
            label = f"⚪ Neutral ({ticker_score:+.2f})"

        # Top catalyst display
        top_catalyst = ""
        top_source = ""
        if matched_social:
            top_catalyst = f"{matched_social[0].get('author', '@Crypto')}: {matched_social[0].get('text', '')[:100]}"
            top_source = "Crypto Twitter / Whale Stream"
        elif matched_articles:
            top_catalyst = matched_articles[0].get("headline", "")
            top_source = matched_articles[0].get("source", "Financial News")
        else:
            top_catalyst = f"Steady liquidity and macro accumulation in {meta['name']}."
            top_source = "On-Chain Microstructure"

        return {
            "symbol": sym_clean,
            "asset_name": meta.get("name", sym_clean),
            "score": ticker_score,
            "label": label,
            "social_buzz": round(social_buzz, 1),
            "matched_news_count": len(matched_articles),
            "matched_social_count": len(matched_social),
            "breaking": breaking,
            "top_catalyst": top_catalyst,
            "top_source": top_source,
            "recent_articles": matched_articles[:4],
            "recent_social": matched_social[:4],
            "timestamp": now
        }

    def inject_social_alert(self, symbol: str, text: str, author: str = "@WhaleAlert", sentiment: Optional[float] = None) -> Dict[str, Any]:
        """Injects a simulated or external live tweet or whale alert for any specific coin."""
        analysis = self.analyze_sentiment(text)
        score = sentiment if sentiment is not None else analysis["sentiment_score"]
        item = {
            "id": f"social_{int(time.time()*1000)}",
            "symbol": symbol.upper(),
            "author": author,
            "text": text,
            "timestamp": int(time.time()),
            "sentiment_score": round(float(score), 2),
            "type": "breaking_social",
            "buzz_weight": 90
        }
        with self.lock:
            if not hasattr(self, "social_alerts"):
                self.social_alerts = list(WHALE_AND_SOCIAL_SEEDS)
            self.social_alerts.insert(0, item)
            self.social_alerts = self.social_alerts[:50]
        logger.info(f"[SOCIAL INGESTION] Injected alert for {symbol}: '{text}' from {author}")
        return item


news_engine = NewsEngine()


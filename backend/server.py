import os
import sys
import time
import math
import random
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import requests

# Set up structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("layaquant")

app = FastAPI(
    title="LayaQuant Decision Daemon",
    description="Ultra-fast (~33ms) System 1 decision microservice powered by convaiinnovations/laya",
    version="1.0.0"
)

# Enable CORS for any frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir), name="app")

@app.get("/")
def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "LayaQuant Daemon Online. Open frontend/index.html"}


# Global model state
MODEL_NAME = "convaiinnovations/laya"
laya_agent = None
engine_type = "calibrated-heuristic-fallback"
device_used = "cpu"
load_start_time = time.time()
load_error_msg = None

import threading

def init_laya_model():
    """Attempt to load convaiinnovations/laya model; gracefully degrade if needed."""
    global laya_agent, engine_type, device_used, load_error_msg
    try:
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device_used = device_str
        logger.info(f"Background loading of Laya Agent on device: {device_str}...")
        
        import laya
        # Load the convaiinnovations/laya checkpoint
        agent = laya.load(MODEL_NAME, device=device_str)
        laya_agent = agent
        engine_type = "convai-laya-neural"
        from backend.news_guard import news_guard
        news_guard.attach_agent(agent)
        logger.info("Successfully loaded convaiinnovations/laya System 1 neural engine!")
    except Exception as e:
        logger.warning(f"Could not load convaiinnovations/laya directly: {e}. Active mode: calibrated-heuristic-fallback.")
        load_error_msg = str(e)
        laya_agent = None
        engine_type = "calibrated-heuristic-fallback"

# Start model loading asynchronously so the microservice starts immediately
threading.Thread(target=init_laya_model, daemon=True).start()

class SmaDistances(BaseModel):
    fast_pct: Optional[float] = 0.0
    slow_pct: Optional[float] = 0.0

class MarketState(BaseModel):
    ticker: str = "BTCUSDT"
    price: float = 60000.0
    rsi: float = 50.0
    sma_fast: Optional[float] = 60000.0
    sma_slow: Optional[float] = 60000.0
    sma_distances: Optional[Dict[str, float]] = None
    volume_spike: Optional[float] = 1.0
    atr_14: Optional[float] = None
    atr_expansion: Optional[float] = 1.0
    funding_rate: Optional[float] = 0.0001
    cvd_delta: Optional[float] = 0.0
    htf_regime: Optional[str] = "neutral"
    is_regime_classifier_call: Optional[bool] = False
    regime: Optional[str] = "neutral"
    criteria: Optional[Dict[str, str]] = None
    choice_options: Optional[List[str]] = ["buy", "sell", "hold"]

class DecisionResponse(BaseModel):
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    reasoning: str
    engine: str
    macro_regime: Optional[str] = "neutral"

def calculate_heuristic_decision(state: MarketState) -> tuple[str, float, Dict[str, float], str, str]:
    """
    Calibrated quantitative decision and regime classification engine:
    1. Multi-Timeframe (MTF) Regime Gating (100% Cash in TOXIC_CHOP)
    2. Derivatives Perpetual Funding Rate Skew Gating
    3. Cumulative Volume Delta (CVD) Order Flow Absorption
    4. Normalized Shannon Entropy Calibration
    """
    rsi = state.rsi
    price = state.price
    sma_fast = state.sma_fast or price
    sma_slow = state.sma_slow or price
    vol_spike = state.volume_spike or 1.0
    atr_exp = state.atr_expansion or 1.0
    funding = state.funding_rate if state.funding_rate is not None else 0.0001
    cvd = state.cvd_delta or 0.0

    sma_spread_pct = ((sma_fast - sma_slow) / max(sma_slow, 1e-6)) * 100.0

    # 1. Macro Regime Detection
    if atr_exp > 1.6 and abs(sma_spread_pct) < 0.45:
        macro_regime = "TOXIC_CHOP"
    elif sma_spread_pct > 0.5 and price >= sma_fast:
        macro_regime = "TREND_BULL"
    elif sma_spread_pct < -0.5 and price <= sma_fast:
        macro_regime = "TREND_BEAR"
    else:
        macro_regime = "RANGE_BOUND"

    # If called strictly to classify regime
    if state.is_regime_classifier_call:
        if macro_regime == "TOXIC_CHOP":
            return "hold", 0.92, {"buy": 0.03, "sell": 0.03, "hold": 0.94}, \
                   f"Regime: TOXIC_CHOP. ATR expansion {atr_exp:.2f}x with flat spread. 100% CASH circuit breaker.", macro_regime
        elif macro_regime == "TREND_BULL":
            return "buy", 0.84, {"buy": 0.85, "sell": 0.05, "hold": 0.10}, \
                   f"Regime: TREND_BULL. SMA spread +{sma_spread_pct:.2f}% with price above fast average.", macro_regime
        elif macro_regime == "TREND_BEAR":
            return "sell", 0.84, {"buy": 0.05, "sell": 0.85, "hold": 0.10}, \
                   f"Regime: TREND_BEAR. Breakdown with spread {sma_spread_pct:.2f}%.", macro_regime
        else:
            return "hold", 0.78, {"buy": 0.20, "sell": 0.20, "hold": 0.60}, \
                   f"Regime: RANGE_BOUND. Volatility compressed ({atr_exp:.2f}x ATR). Range bounce rules active.", macro_regime

    # 2. HTF Circuit Breaker
    effective_htf = state.htf_regime or macro_regime
    if effective_htf == "TOXIC_CHOP":
        return "hold", 0.96, {"buy": 0.02, "sell": 0.02, "hold": 0.96}, \
               "Circuit Breaker Triggered: HTF Macro Regime is TOXIC CHOP (liquidation risk). Mandating 100% Cash.", macro_regime

    # 3. Base Indicator Scoring
    buy_score = 0.0
    sell_score = 0.0
    hold_score = 1.0

    # Macro regime alignment
    if effective_htf == "TREND_BULL":
        buy_score += 2.0
        sell_score -= 2.0  # do not fight macro trend
    elif effective_htf == "TREND_BEAR":
        sell_score += 2.0
        buy_score -= 2.0

    # RSI Trigger
    if rsi < 32.0:
        intensity = (32.0 - rsi) / 15.0
        buy_score += 2.8 * intensity
    elif rsi > 68.0:
        intensity = (rsi - 68.0) / 15.0
        sell_score += 2.8 * intensity

    # SMA trend edge
    if sma_spread_pct > 0.4:
        buy_score += 1.6 * min(sma_spread_pct, 2.5)
    elif sma_spread_pct < -0.4:
        sell_score += 1.6 * min(abs(sma_spread_pct), 2.5)

    # CVD Order Flow Alpha (Aggressive buyer/seller delta)
    if cvd > 0.5:
        buy_score += 1.4  # institutional absorption buying
    elif cvd < -0.5:
        sell_score += 1.4

    # Derivatives Funding Rate Crowded Positioning Skew
    funding_note = ""
    if funding > 0.0003:
        # Over-leveraged longs -> Long squeeze crash risk
        buy_score -= 2.5
        hold_score += 1.2
        funding_note = f" [Funding Skew: Crowded Long {funding*100:.3f}% - gating buy]"
    elif funding < -0.0003:
        # Over-leveraged shorts -> Short squeeze rip potential
        sell_score -= 2.5
        buy_score += 1.0
        funding_note = f" [Funding Skew: Crowded Short {funding*100:.3f}% - short squeeze alert]"

    # Volume expansion
    if vol_spike > 1.35:
        if buy_score > sell_score:
            buy_score += 1.2 * min(vol_spike - 1.0, 2.0)
        elif sell_score > buy_score:
            sell_score += 1.2 * min(vol_spike - 1.0, 2.0)
    else:
        hold_score += 0.8

    # Prevent negative logits
    buy_score = max(0.0, buy_score)
    sell_score = max(0.0, sell_score)
    hold_score = max(0.2, hold_score)

    # Softmax probabilities
    scores = np.array([buy_score, sell_score, hold_score], dtype=np.float64)
    temperature = 1.15
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    probs = exp_scores / np.sum(exp_scores)
    p_buy, p_sell, p_hold = float(probs[0]), float(probs[1]), float(probs[2])

    # Normalized Shannon Entropy Confidence
    k = 3
    h_p = - (p_buy * math.log(max(p_buy, 1e-9)) + 
             p_sell * math.log(max(p_sell, 1e-9)) + 
             p_hold * math.log(max(p_hold, 1e-9)))
    max_entropy = math.log(k)
    confidence = float(np.clip(1.0 - (h_p / max_entropy), 0.05, 0.99))

    choices = ["buy", "sell", "hold"]
    best_idx = int(np.argmax(probs))
    choice = choices[best_idx]

    if choice == "buy":
        reasoning = (
            f"Bullish setup: RSI {rsi:.1f} in {effective_htf} regime. "
            f"SMA spread {'+' if sma_spread_pct >= 0 else ''}{sma_spread_pct:.2f}%, Vol {vol_spike:.2f}x.{funding_note}"
        )
    elif choice == "sell":
        reasoning = (
            f"Bearish exit/short: RSI {rsi:.1f} in {effective_htf} regime. "
            f"SMA spread {sma_spread_pct:.2f}%, Vol {vol_spike:.2f}x.{funding_note}"
        )
    else:
        reasoning = (
            f"Hold mode: {effective_htf} regime with balanced RSI ({rsi:.1f}).{funding_note}"
        )

    prob_dict = {
        "buy": round(p_buy, 4),
        "sell": round(p_sell, 4),
        "hold": round(p_hold, 4)
    }

    return choice, round(confidence, 4), prob_dict, reasoning, macro_regime

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "LayaQuant Studio Daemon",
        "model": MODEL_NAME,
        "engine": engine_type,
        "device": device_used,
        "uptime_sec": round(time.time() - load_start_time, 2),
        "fallback_active": (laya_agent is None),
        "last_error": load_error_msg
    }

@app.post("/v1/systemone", response_model=DecisionResponse)
def system_one_decision(state: MarketState):
    t0 = time.perf_counter()
    global laya_agent, engine_type

    criteria = state.criteria or {
        "buy": "RSI < 35 or fast SMA crossing above slow SMA with high volume expansion.",
        "sell": "RSI > 65 or fast SMA crossing below slow SMA indicating bearish momentum.",
        "hold": "Market in sideways consolidation with neutral indicators."
    }

    used_engine = engine_type

    if laya_agent is not None:
        try:
            # Format state prompt string for Laya
            state_text = (
                f"Instrument: {state.ticker}\n"
                f"Current Price: {state.price:.2f}\n"
                f"RSI (14): {state.rsi:.1f}\n"
                f"Fast SMA: {state.sma_fast or state.price:.2f}\n"
                f"Slow SMA: {state.sma_slow or state.price:.2f}\n"
                f"Volume Spike: {state.volume_spike or 1.0:.2f}x\n"
                f"Market Regime: {state.regime or 'ranging'}"
            )
            
            questions = {
                "decision": {
                    "type": "choice",
                    "instructions": "Evaluate the quantitative market metrics and determine the highest-probability trading action.",
                    "criteria": {
                        "buy": criteria.get("buy", "Buy on bullish divergence or oversold mean reversion."),
                        "sell": criteria.get("sell", "Sell on bearish breakdown or overbought momentum exhaustion."),
                        "hold": criteria.get("hold", "Hold during indeterminate consolidation.")
                    }
                }
            }

            res = laya_agent.system_one(state_text, questions)
            ans = res["answers"]["decision"]
            choice = ans["choice"]
            confidence = float(ans["confidence"])
            probabilities = {k: float(v) for k, v in ans["probabilities"].items()}
            reasoning = f"Laya System 1: Selected '{choice}' with {confidence*100:.1f}% confidence. " + \
                        f"Indicators: RSI={state.rsi:.1f}, VolSpike={state.volume_spike:.1f}x."
            used_engine = "convai-laya-neural"
            macro_regime = state.htf_regime or "neutral"
        except Exception as e:
            logger.warning(f"Laya neural inference error, falling back to calibrated heuristic: {e}")
            choice, confidence, probabilities, reasoning, macro_regime = calculate_heuristic_decision(state)
            used_engine = "calibrated-heuristic-fallback"
    else:
        choice, confidence, probabilities, reasoning, macro_regime = calculate_heuristic_decision(state)
        used_engine = "calibrated-heuristic-fallback"

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return DecisionResponse(
        choice=choice,
        confidence=confidence,
        probabilities=probabilities,
        latency_ms=latency_ms,
        reasoning=reasoning,
        engine=used_engine,
        macro_regime=macro_regime
    )

@app.get("/api/market/derivatives")
def get_derivatives_metrics(symbol: str = Query("BTCUSDT")):
    """
    Query open unauthenticated Binance perpetual premium index & funding rate endpoint.
    Reveals over-leveraged positioning and crowded long/short liquidation traps.
    """
    clean_sym = symbol.upper().replace("/", "").replace("-", "")
    url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={clean_sym}"
    try:
        resp = requests.get(url, timeout=3.5)
        if resp.status_code == 200:
            d = resp.json()
            fr = float(d.get("lastFundingRate", 0.0001))
            mark = float(d.get("markPrice", 0.0))
            idx = float(d.get("indexPrice", 0.0))
            basis_bps = ((mark - idx) / max(idx, 1e-6)) * 10000.0

            skew = "neutral"
            if fr > 0.0003:
                skew = "crowded_long"
            elif fr < -0.0003:
                skew = "crowded_short"

            return {
                "symbol": clean_sym,
                "funding_rate": fr,
                "funding_rate_pct": round(fr * 100.0, 4),
                "mark_price": mark,
                "index_price": idx,
                "basis_bps": round(basis_bps, 2),
                "skew": skew,
                "time": d.get("time", int(time.time() * 1000))
            }
    except Exception as e:
        logger.warning(f"Derivatives funding rate fetch error ({e}). Returning fallback.")

    return {
        "symbol": clean_sym,
        "funding_rate": 0.0001,
        "funding_rate_pct": 0.01,
        "mark_price": get_default_price(clean_sym),
        "index_price": get_default_price(clean_sym),
        "basis_bps": 0.0,
        "skew": "neutral",
        "time": int(time.time() * 1000)
    }

@app.get("/api/market/binance")
def get_binance_klines(
    symbol: str = Query("BTCUSDT", description="Crypto symbol"),
    interval: str = Query("1h", description="Candle interval (1m, 5m, 15m, 1h, 4h, 1d)"),
    limit: int = Query(150, ge=10, le=500)
):
    """
    Proxy to free, open public Binance Klines REST endpoint with CVD (Cumulative Volume Delta)
    derived directly from open order flow taker buy volume.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=4.0)
        if resp.status_code == 200:
            raw_klines = resp.json()
            candles = []
            for k in raw_klines:
                # k[5]: total volume, k[9]: taker buy volume
                tot_vol = float(k[5])
                taker_buy = float(k[9]) if len(k) > 9 else tot_vol * 0.5
                taker_sell = max(0.0, tot_vol - taker_buy)
                cvd_delta = taker_buy - taker_sell

                candles.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": tot_vol,
                    "taker_buy": round(taker_buy, 4),
                    "taker_sell": round(taker_sell, 4),
                    "cvd_delta": round(cvd_delta, 4)
                })
            return {"symbol": symbol.upper(), "source": "binance_public", "candles": candles}
        else:
            logger.warning(f"Binance public API returned status {resp.status_code}. Generating synthetic fallback.")
    except Exception as e:
        logger.warning(f"Binance API fetch error ({e}). Using synthetic fallback.")

    return generate_stochastic_series(symbol.upper(), limit, base_price=get_default_price(symbol))

@app.get("/api/market/yfinance")
def get_yfinance_history(
    symbol: str = Query("SPY", description="Equity ticker (e.g. SPY, NVDA, AAPL, QQQ)"),
    period: str = Query("3mo", description="Period"),
    interval: str = Query("1d", description="Interval")
):
    """
    Fetch historical daily/hourly equity data using open-source yfinance.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=period, interval=interval)
        if not df.empty:
            candles = []
            for idx, row in df.iterrows():
                ts = int(idx.timestamp() * 1000) if hasattr(idx, "timestamp") else int(time.time() * 1000)
                candles.append({
                    "timestamp": ts,
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": round(float(row["Volume"]), 2),
                })
            return {"symbol": symbol.upper(), "source": "yfinance", "candles": candles}
    except Exception as e:
        logger.warning(f"yfinance error for {symbol}: {e}. Returning synthetic.")

    return generate_stochastic_series(symbol.upper(), 120, base_price=get_default_price(symbol))

@app.get("/api/market/synthetic")
def get_synthetic_market(
    symbol: str = Query("SYNTH_BTC", description="Symbol name"),
    regime: str = Query("trending_bull", description="Regime: trending_bull, trending_bear, mean_reverting, high_volatility"),
    points: int = Query(150, ge=30, le=500)
):
    """
    Direct endpoint for generating realistic synthetic markets with jump-diffusion.
    """
    base_price = get_default_price(symbol)
    return generate_stochastic_series(symbol, points, base_price=base_price, regime=regime)

def get_default_price(symbol: str) -> float:
    s = symbol.upper()
    if "BTC" in s:
        return 65000.0
    elif "ETH" in s:
        return 3400.0
    elif "SOL" in s:
        return 160.0
    elif "NVDA" in s:
        return 120.0
    elif "SPY" in s:
        return 540.0
    return 100.0

def generate_stochastic_series(symbol: str, count: int, base_price: float = 100.0, regime: str = "trending_bull") -> dict:
    """
    Geometric Brownian Motion with Jump Diffusion and Volatility Regimes.
    """
    np.random.seed(int(time.time()) % 100000)
    dt = 1.0 / count
    
    if regime == "trending_bull":
        mu, sigma, jump_prob, jump_size = 0.45, 0.28, 0.05, 0.03
    elif regime == "trending_bear":
        mu, sigma, jump_prob, jump_size = -0.35, 0.35, 0.08, -0.04
    elif regime == "mean_reverting":
        mu, sigma, jump_prob, jump_size = 0.02, 0.18, 0.03, 0.015
    else:  # high_volatility / jump
        mu, sigma, jump_prob, jump_size = 0.05, 0.55, 0.12, 0.06

    prices = [base_price]
    cur_p = base_price
    
    # Generate candles
    candles = []
    now_ms = int(time.time() * 1000) - (count * 3600 * 1000)

    for i in range(count):
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal()
        jump = jump_size * np.random.normal() if np.random.random() < jump_prob else 0.0
        
        ret = drift + shock + jump
        cur_p = max(0.5, cur_p * np.exp(ret))
        
        # Intra-candle high / low / open
        intraday_vol = cur_p * (sigma * 0.4)
        c_open = prices[-1] if i > 0 else cur_p * (1.0 - 0.003 * np.random.normal())
        c_close = cur_p
        c_high = max(c_open, c_close) + abs(np.random.normal() * intraday_vol)
        c_low = min(c_open, c_close) - abs(np.random.normal() * intraday_vol)
        c_low = max(0.1, c_low)
        
        volume = float(np.random.gamma(shape=2.5, scale=12000.0) * (1.0 + abs(ret) * 15.0))

        candles.append({
            "timestamp": now_ms + (i * 3600 * 1000),
            "open": round(c_open, 2),
            "high": round(c_high, 2),
            "low": round(c_low, 2),
            "close": round(c_close, 2),
            "volume": round(volume, 2)
        })
        prices.append(cur_p)

    return {
        "symbol": symbol,
        "source": "synthetic_stochastic_gbm",
        "regime": regime,
        "candles": candles
    }

# ==========================================
# 24-HOUR PAPER TRADING SYSTEM INTEGRATION
# ==========================================
from backend.paper_trader import paper_trader, TRADES_CSV

def paper_decision_bridge(market_state: dict):
    state_obj = MarketState(**market_state)
    res = system_one_decision(state_obj)
    return {
        "choice": res.choice,
        "confidence": res.confidence,
        "probabilities": res.probabilities,
        "reasoning": res.reasoning,
        "engine": res.engine
    }

paper_trader.decision_callback = paper_decision_bridge

@app.on_event("startup")
def startup_news_guard():
    from backend.news_guard import news_guard
    news_guard.start()

@app.get("/api/news_guard/status")
def get_news_guard_status():
    from backend.news_guard import news_guard
    return news_guard.status()

@app.post("/api/news_guard/enforce")
def set_news_guard_enforce(on: bool = Query(...)):
    from backend.news_guard import news_guard
    news_guard.enforce = on
    return {"enforce": news_guard.enforce}

@app.on_event("startup")
def startup_paper_trader():
    # Guarantee worker thread runs if positions are active or is_running was persisted
    if paper_trader.is_running or (hasattr(paper_trader, "positions") and len(paper_trader.positions) > 0):
        paper_trader.is_running = True
        paper_trader.stop_event.clear()
        if paper_trader.worker_thread is None or not paper_trader.worker_thread.is_alive():
            import threading
            paper_trader.worker_thread = threading.Thread(target=paper_trader._run_loop, daemon=True)
            paper_trader.worker_thread.start()
            logger.info(f"Server Startup: Resumed paper trading loop ({len(paper_trader.positions)} active positions).")

class PaperStartRequest(BaseModel):
    symbol: Optional[str] = "BTCUSDT"
    interval: Optional[str] = "5m"
    capital: Optional[float] = 10000.0
    gate: Optional[float] = 0.72
    duration_hours: Optional[float] = 720.0

@app.get("/api/paper/status")
def get_paper_status():
    return paper_trader.get_status()

@app.get("/api/paper/sessions")
def get_paper_sessions():
    return paper_trader.get_live_sessions()

@app.post("/api/paper/start")
def start_paper_trading(req: PaperStartRequest):
    return paper_trader.start(
        symbol=req.symbol or "BTCUSDT",
        interval=req.interval or "5m",
        initial_capital=req.capital or 10000.0,
        confidence_gate=req.gate or 0.72,
        target_duration_hours=req.duration_hours or 720.0
    )

@app.post("/api/paper/stop")
def stop_paper_trading():
    return paper_trader.stop()

@app.post("/api/paper/reset")
def reset_paper_trading(capital: float = Query(10000.0)):
    return paper_trader.reset(initial_capital=capital)

@app.post("/api/paper/step")
def step_paper_trading(bars: int = Query(1, ge=1, le=50)):
    results = []
    for _ in range(bars):
        res = paper_trader.evaluate_step()
        results.append(res)
    return {"status": "step_executed", "bars_stepped": bars, "result": results[-1], "all_steps": results}

class ShockRequest(BaseModel):
    symbol: Optional[str] = "BTCUSDT"
    shock_type: str = "crash"  # "crash", "surge", "chop"

@app.post("/api/market/inject_shock")
def inject_market_shock(req: ShockRequest):
    current_status = paper_trader.get_status()
    base_px = current_status.get("current_price", 80000.0)
    
    if req.shock_type == "crash":
        new_px = base_px * 0.95
        note = "Flash crash injected: -5.0% price shock"
    elif req.shock_type == "surge":
        new_px = base_px * 1.05
        note = "Bull surge injected: +5.0% breakout momentum"
    else:
        new_px = base_px * (1.0 + (random.random() - 0.5) * 0.04)
        note = "Toxic chop injected: erratic whipsaw volatility"
        
    paper_trader.current_price = round(new_px, 2)
    res = paper_trader.evaluate_step()
    return {"status": "shock_injected", "shock_type": req.shock_type, "new_price": round(new_px, 2), "step_result": res, "note": note}

class DepositRequest(BaseModel):
    amount: float = 1000.0

@app.post("/api/paper/deposit")
def deposit_paper_money(req: DepositRequest):
    return paper_trader.deposit_cash(req.amount)

class RecurringDepositRequest(BaseModel):
    amount: float = 100.0            # added every 30 days; 0 turns it off
    target: Optional[float] = 10000.0

@app.post("/api/paper/recurring_deposit")
def set_recurring_deposit(req: RecurringDepositRequest):
    return paper_trader.set_recurring_deposit(req.amount, req.target)

# ==========================================
# PLAN: crypto trend + TradFi trend at one risk level (frontend/plan.html)
# ==========================================
class PlanStartRequest(BaseModel):
    budget: float = 500.0
    risk_level: float = 2.0          # 1 = calm ... 3 = aggressive (see tradfi_book.py)
    monthly_deposit: float = 100.0
    target: float = 10000.0
    tradfi: bool = True              # include gold / silver / S&P 500 / Nasdaq 100 perps

@app.get("/plan")
def serve_plan():
    return FileResponse(os.path.join(frontend_dir, "plan.html"))

@app.post("/api/plan/start")
def start_plan(req: PlanStartRequest):
    return paper_trader.start_plan(req.budget, req.risk_level, req.monthly_deposit, req.target, req.tradfi)

@app.post("/api/plan/pause")
def pause_plan():
    return paper_trader.pause_plan()

@app.post("/api/plan/resume")
def resume_plan():
    return paper_trader.resume_plan()

@app.get("/api/plan/status")
def get_plan_status():
    return paper_trader.get_plan_status()

@app.get("/api/plan/journal")
def get_plan_journal(all_plans: bool = Query(False)):
    """Hourly equity and events from data/journal/*.csv (current plan only unless all_plans)."""
    from backend.plan_reporter import plan_reporter
    since = 0 if all_plans else paper_trader.plan.get("started_at", 0)
    return plan_reporter.journal(since_ts=since)

class TelegramConnectRequest(BaseModel):
    token: str

class TelegramSettingsRequest(BaseModel):
    enabled: Optional[bool] = None
    weekly: Optional[bool] = None
    daily: Optional[bool] = None

@app.get("/api/telegram/status")
def telegram_status():
    from backend.notifier import notifier
    return notifier.status()

@app.post("/api/telegram/connect")
def telegram_connect(req: TelegramConnectRequest):
    """Checks the token, then waits up to 90 s for the user to message the bot."""
    from backend.notifier import notifier
    return notifier.connect(req.token)

@app.post("/api/telegram/settings")
def telegram_settings(req: TelegramSettingsRequest):
    from backend.notifier import notifier
    return notifier.settings(req.enabled, req.weekly, req.daily)

@app.post("/api/telegram/test")
def telegram_test():
    from backend.notifier import notifier
    if not notifier.status()["configured"]:
        return {"status": "error", "message": "Not connected yet. Send your bot any message in Telegram, then press "
                                              "Connect and wait for \"Connected\"."}
    if not notifier.ready:
        return {"status": "error", "message": "Notifications are switched off. Tick \"Notifications on\"."}
    s = paper_trader.get_plan_status()
    notifier.send(f"🔔 Test message. Equity {s['equity']:,.2f}, deposited {s['deposited']:,.2f}, "
                  f"{'running' if s['is_running'] else 'paused'}.")
    return {"status": "sent"}

class SetCashRequest(BaseModel):
    cash: float = 10000.0

@app.post("/api/paper/set_cash")
def set_paper_cash(req: SetCashRequest):
    return paper_trader.set_cash(req.cash)

class BudgetRequest(BaseModel):
    allocation_pct: float = 0.5

@app.post("/api/paper/budget")
def update_paper_budget(req: BudgetRequest):
    return paper_trader.set_budget(req.allocation_pct)

@app.post("/api/paper/toggle_maker")
def toggle_paper_maker(maker: bool = Query(True)):
    with paper_trader.lock:
        paper_trader.use_maker_execution = maker
        paper_trader.save_state()
    return {"status": "success", "use_maker_execution": paper_trader.use_maker_execution}

class SetLeverageRequest(BaseModel):
    leverage: float = 3.0

@app.post("/api/paper/set_leverage")
def set_paper_leverage(req: SetLeverageRequest):
    return paper_trader.set_leverage(req.leverage)

class SetCompoundingRequest(BaseModel):
    enabled: bool = True

@app.post("/api/paper/set_compounding")
def set_paper_compounding(req: SetCompoundingRequest):
    return paper_trader.set_compounding(req.enabled)

class SetMilestoneModeRequest(BaseModel):
    enabled: bool = True

@app.post("/api/paper/set_milestone_mode")
def set_paper_milestone_mode(req: SetMilestoneModeRequest):
    return paper_trader.set_milestone_mode(req.enabled)

class ManualOrderRequest(BaseModel):
    side: str = "LONG"
    margin_amount: Optional[float] = None

@app.post("/api/paper/manual_order")
def execute_manual_order(req: ManualOrderRequest):
    return paper_trader.open_manual_position(req.side, margin_amount=req.margin_amount)

class ClosePositionRequest(BaseModel):
    symbol: Optional[str] = None

@app.post("/api/paper/close_position")
def close_paper_position(req: Optional[ClosePositionRequest] = None, symbol: Optional[str] = None):
    target_sym = req.symbol if (req and req.symbol) else symbol
    return paper_trader.close_manual_position(symbol=target_sym)

class SetMaxPositionsRequest(BaseModel):
    max_positions: int = 3

@app.post("/api/paper/set_max_positions")
def set_paper_max_positions(req: SetMaxPositionsRequest):
    return paper_trader.set_max_concurrent_positions(req.max_positions)

@app.get("/api/paper/scanner_radar")
def get_paper_scanner_radar():
    return paper_trader.get_scanner_radar()

class SetStrategyModeRequest(BaseModel):
    mode: str = "TREND"

@app.post("/api/paper/set_strategy_mode")
def set_paper_strategy_mode(req: SetStrategyModeRequest):
    return paper_trader.set_strategy_mode(req.mode)

class SetScannerModeRequest(BaseModel):
    enabled: bool = True

@app.post("/api/paper/set_scanner_mode")
def set_paper_scanner_mode(req: SetScannerModeRequest):
    return paper_trader.set_scanner_mode(req.enabled)

class SetScannerUniverseRequest(BaseModel):
    symbols: List[str]

@app.post("/api/paper/set_scanner_universe")
def set_paper_scanner_universe(req: SetScannerUniverseRequest):
    return paper_trader.set_scanner_universe(req.symbols)

# Dual-Model Collaborative Intelligence Endpoints
@app.get("/api/dual_model/status")
def get_dual_model_status():
    return paper_trader.get_dual_model_status()

class ConfigureDualModelRequest(BaseModel):
    enforce_consensus: bool = True

@app.post("/api/dual_model/configure")
def configure_dual_model(req: ConfigureDualModelRequest):
    return paper_trader.configure_dual_model(req.enforce_consensus)

@app.get("/api/paper/export")
def export_paper_trades():
    if os.path.exists(TRADES_CSV):
        return FileResponse(TRADES_CSV, media_type="text/csv", filename=f"paper_trades_{int(time.time())}.csv")
    raise HTTPException(status_code=404, detail="No paper trades logged yet.")

# ==========================================
# DREAM-RSI RECURSIVE SELF-IMPROVEMENT ENGINE
# ==========================================
from backend.dream_rsi import DreamRSISelfImprover

dream_improver = DreamRSISelfImprover()

class DreamOptimizeRequest(BaseModel):
    symbol: Optional[str] = "BTCUSDT"
    candles: Optional[List[Dict[str, Any]]] = None

@app.get("/api/dream/history")
def get_dream_history():
    return {"history": dream_improver.get_dream_history()}

@app.get("/api/dream/replays")
def get_dream_replays():
    return {"replays": dream_improver.get_all_replays()}

class RecordReplayRequest(BaseModel):
    session: Dict[str, Any]

@app.post("/api/dream/record_replay")
def record_dream_replay(req: RecordReplayRequest):
    dream_improver.record_replay_session(req.session)
    return {"status": "success"}

@app.post("/api/dream/optimize_multi")
def run_cross_asset_dream_optimization():
    current_status = paper_trader.get_status()
    current_config = {
        "confidence_gate": current_status.get("confidence_gate", 0.65),
        "sl_atr_mult": current_status.get("sl_atr_mult", 1.5),
        "tp_atr_mult": current_status.get("tp_atr_mult", 3.75)
    }
    return dream_improver.run_cross_asset_dream_optimization(current_config)

class DeployDreamRequest(BaseModel):
    policy: Optional[Dict[str, Any]] = None
    formula: Optional[str] = None

@app.post("/api/paper/deploy_dream")
def deploy_dream_policy(req: Optional[DeployDreamRequest] = None):
    pol = req.policy if req else None
    form = (req.formula or "") if req else ""
    if not pol:
        pol = dream_improver.get_active_dream_policy()
        if not form:
            form = "If RSI < 36.0 or (RSI > 52.0 with Bullish Trend) and Confidence >= 78% --> BUY LONG (SL: 2.2x ATR, TP: 2.8x ATR)"
    return paper_trader.apply_dream_policy(pol, formula=form)

# ==========================================
# LIVE NEWS INGESTION & NEURAL DREAM TRAINER
# ==========================================
from backend.news_engine import news_engine
from backend.neural_dream import intelligent_dream_trainer

@app.get("/api/news/latest")
def get_latest_news(symbol: Optional[str] = "BTCUSDT", refresh: Optional[bool] = False):
    news_engine.fetch_live_web_news(symbol or "BTCUSDT", force=bool(refresh))
    return news_engine.get_aggregate_market_sentiment()

@app.post("/api/news/refresh")
def refresh_live_news_endpoint(symbol: Optional[str] = "BTCUSDT"):
    news_engine.fetch_live_web_news(symbol or "BTCUSDT", force=True)
    return news_engine.get_aggregate_market_sentiment()

class SimulateNewsRequest(BaseModel):
    headline: str
    sentiment: Optional[float] = None
    category: Optional[str] = None

@app.post("/api/news/simulate")
def simulate_news_endpoint(req: SimulateNewsRequest):
    item = news_engine.simulate_breaking_news(req.headline, sentiment=req.sentiment, category=req.category)
    agg = news_engine.get_aggregate_market_sentiment()
    return {"status": "injected", "headline": item, "aggregate_sentiment": agg}

@app.get("/api/news/ticker_sentiment")
def get_ticker_sentiment_endpoint(symbol: Optional[str] = "BTCUSDT"):
    return news_engine.get_ticker_sentiment(symbol or "BTCUSDT")

@app.get("/api/news/pulse")
def get_live_news_pulse():
    return news_engine.get_live_web_pulse()

@app.get("/api/news/fear_and_greed")
def get_fear_and_greed_endpoint():
    return news_engine.get_fear_and_greed()

class InjectSocialAlertRequest(BaseModel):
    symbol: str = "XRPUSDT"
    text: str
    author: Optional[str] = "@WhaleAlert"
    sentiment: Optional[float] = None

@app.post("/api/news/inject_social_alert")
def inject_social_alert_endpoint(req: InjectSocialAlertRequest):
    item = news_engine.inject_social_alert(req.symbol, req.text, author=req.author or "@WhaleAlert", sentiment=req.sentiment)
    ticker_sent = news_engine.get_ticker_sentiment(req.symbol)
    return {"status": "injected", "alert": item, "ticker_sentiment": ticker_sent}

class TrainNeuralRequest(BaseModel):
    symbol: Optional[str] = "BTCUSDT"
    interval: Optional[str] = "1d"
    epochs: Optional[int] = 15

@app.post("/api/dream/train_neural_model")
def train_neural_model_endpoint(req: TrainNeuralRequest):
    sym = (req.symbol or "BTCUSDT").upper()
    interval = (req.interval or "1d").lower()
    candles = fetch_historical_dataset(sym, interval=interval)
    epochs = req.epochs or 15
    res = intelligent_dream_trainer.train_model(candles, epochs=epochs)
    paper_trader.deploy_neural_model()
    return res

@app.post("/api/paper/deploy_neural_model")
def deploy_neural_model_endpoint():
    return paper_trader.deploy_neural_model()

# ==========================================
# BEHAVIORAL ECONOMICS & PHD RESEARCH ENGINE
# ==========================================
from backend.research_engine import research_engine

@app.get("/api/research/library")
def get_research_library():
    """Returns curated repository of foundational PhD economic and behavioral research papers."""
    return {"papers": research_engine.papers, "count": len(research_engine.papers)}

@app.get("/api/research/metrics")
def get_research_metrics(symbol: Optional[str] = "BTCUSDT"):
    """Computes real-time behavioral economics & market microstructure metrics for the current market state."""
    sym = (symbol or "BTCUSDT").upper()
    status = paper_trader.get_status()
    m_state = {
        "ticker": sym,
        "price": status.get("current_price", 60000.0),
        "rsi": 50.0,
        "sma_fast": status.get("current_price", 60000.0),
        "sma_slow": status.get("current_price", 60000.0),
        "volume_spike": 1.0,
        "atr_14": status.get("current_atr", 1000.0),
        "cvd_delta": 0.0,
        "funding_rate": status.get("funding_rate", 0.0001),
        "htf_regime": status.get("htf_regime", "neutral")
    }
    n_state = news_engine.get_aggregate_market_sentiment()
    metrics = research_engine.compute_research_features(m_state, n_state)
    synthesis = research_engine.generate_academic_synthesis(metrics, m_state)
    return {
        "symbol": sym,
        "metrics": metrics,
        "academic_synthesis": synthesis,
        "papers_count": len(research_engine.papers),
        "factor_weights": intelligent_dream_trainer.learned_factor_weights,
        "timestamp": time.time()
    }

class InjectPaperRequest(BaseModel):
    title: str
    authors: str
    institution: Optional[str] = "Academic / Quantitative Research"
    journal: Optional[str] = "Independent Quantitative Whitepaper"
    year: Optional[int] = 2026
    citation_count: Optional[int] = 100
    category: Optional[str] = "behavioral_psychology"
    core_thesis: str
    mathematical_formulation: Optional[str] = "y = f(x)"
    algorithmic_trading_rule: Optional[str] = ""

@app.post("/api/research/inject")
def inject_research_paper_endpoint(req: InjectPaperRequest):
    paper = research_engine.inject_research_paper(req.dict())
    return {"status": "injected", "paper": paper, "total_papers": len(research_engine.papers)}


def fetch_historical_dataset(sym: str, interval: str = "1d") -> List[Dict[str, Any]]:
    """
    Unified multi-timeframe candle fetching for 15m, 1h, and 1d.
    Supports both Binance (crypto) and yfinance (US equities) with synthetic fallback.
    """
    sym = (sym or "BTCUSDT").upper()
    interval = (interval or "1d").lower()
    if interval not in ("15m", "1h", "1d"):
        interval = "1d"

    if interval == "15m":
        binance_limit = 500
        yf_period = "60d"
    elif interval == "1h":
        binance_limit = 500
        yf_period = "6mo"
    else:  # 1d
        binance_limit = 180
        yf_period = "6mo"

    candles = []
    if "USDT" in sym:
        try:
            m_res = get_binance_klines(symbol=sym, interval=interval, limit=binance_limit)
            candles = m_res.get("candles", [])
        except Exception as e:
            logger.warning(f"Binance fetch err for {sym} ({interval}): {e}")
    else:
        try:
            m_res = get_yfinance_history(symbol=sym, period=yf_period, interval=interval)
            candles = m_res.get("candles", [])
        except Exception as e:
            logger.warning(f"yfinance fetch err for {sym} ({interval}): {e}")

    if not candles or len(candles) < 30:
        candles = generate_stochastic_series(sym, binance_limit, base_price=get_default_price(sym))

    return candles

class Dream6MoRequest(BaseModel):
    symbol: Optional[str] = "BTCUSDT"
    interval: Optional[str] = "1d"
    target_sharpe: Optional[float] = 2.2
    min_winrate: Optional[float] = 65.0

@app.post("/api/dream/learn_6mo")
def learn_6mo_dream(req: Dream6MoRequest):
    sym = (req.symbol or "BTCUSDT").upper()
    interval = (req.interval or "1d").lower()
    candles = fetch_historical_dataset(sym, interval=interval)

    res = dream_improver.dream_6mo_until_converged(
        symbol=sym,
        candles=candles,
        target_sharpe=req.target_sharpe or 2.2,
        min_winrate=req.min_winrate or 65.0,
        max_generations=8,
        interval=interval
    )
    if "converged_policy" in res and "config" in res["converged_policy"]:
        paper_trader.apply_dream_policy(res["converged_policy"]["config"], formula=res.get("discovered_formula", ""))
    return res

@app.post("/api/dream/optimize")
def run_dream_optimization(req: DreamOptimizeRequest):
    candles = req.candles
    if not candles:
        try:
            m_res = get_binance_klines(symbol=req.symbol or "BTCUSDT", interval="1h", limit=100)
            candles = m_res.get("candles", [])
        except Exception as e:
            logger.warning(f"Failed to fetch market candles for Dream-RSI: {e}")
    
    if not candles or len(candles) < 30:
        raise HTTPException(status_code=400, detail="Insufficient candle data (requires >= 30 bars) to run Dream-RSI replay.")
    
    current_status = paper_trader.get_status()
    current_config = {
        "confidence_gate": current_status.get("confidence_gate", 0.65),
        "sl_atr_mult": current_status.get("sl_atr_mult", 1.5),
        "tp_atr_mult": current_status.get("tp_atr_mult", 3.75)
    }
    
    res = dream_improver.run_dream_optimization(candles, current_config)
    return res

class WalkForwardRequest(BaseModel):
    symbol: Optional[str] = "BTCUSDT"
    capital: Optional[float] = 10000.0
    interval: Optional[str] = "1d"
    re_optimize_days: Optional[int] = 7

@app.post("/api/dream/walk_forward_6mo")
def run_walk_forward_6mo(req: WalkForwardRequest):
    sym = (req.symbol or "BTCUSDT").upper()
    interval = (req.interval or "1d").lower()
    candles = fetch_historical_dataset(sym, interval=interval)

    return dream_improver.run_walk_forward_day_by_day(
        symbol=sym,
        candles=candles,
        initial_capital=req.capital or 10000.0,
        adaptation_frequency_days=req.re_optimize_days or 7,
        interval=interval
    )

class FeedDailyRunRequest(BaseModel):
    symbol: Optional[str] = "BTCUSDT"
    interval: Optional[str] = "1d"
    walk_forward_result: Optional[Dict[str, Any]] = None

@app.post("/api/dream/feed_daily_run")
def feed_daily_run_endpoint(req: FeedDailyRunRequest):
    sym = (req.symbol or "BTCUSDT").upper()
    interval = (req.interval or "1d").lower()
    wf_res = req.walk_forward_result
    candles = None
    
    if not wf_res:
        candles = fetch_historical_dataset(sym, interval=interval)
        wf_res = dream_improver.run_walk_forward_day_by_day(symbol=sym, candles=candles, interval=interval)

    res = dream_improver.feed_daily_run_to_dream(symbol=sym, walk_forward_result=wf_res, candles=candles, interval=interval)
    if "converged_policy" in res and "config" in res["converged_policy"]:
        paper_trader.apply_dream_policy(res["converged_policy"]["config"], formula=res.get("discovered_formula", ""))
    return res

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting LayaQuant Daemon on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


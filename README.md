# 📈 LayaQuant Studio

> **100% Open-Source, Zero-Paid-API Algorithmic Trading Signal & Backtesting Simulation Platform**  
> Powered by the **Laya System 1 Decision Model** (`convaiinnovations/laya`) running locally on CPU/GPU as an ultra-low latency (~33ms) decision microservice.

---

## ⚡ Key Highlights & Constraints Adherence

- **ZERO Paid APIs & ZERO Credit Cards**: No OpenAI, Anthropic, or external subscription dependencies.
- **Model Engine**: Open-source `convaiinnovations/laya` via `pip install laya` running locally as a non-autoregressive, calibrated System 1 decision engine. Includes calibrated quantitative Shannon entropy fallback.
- **Market Data Feeds**: Unauthenticated public endpoints (Binance public REST Klines: `https://api.binance.com/api/v3/klines` for crypto like BTCUSDT, ETHUSDT, SOLUSDT, and `yfinance` for US equities like SPY, NVDA, QQQ).
- **Stochastic Market Simulator**: Client-side Geometric Brownian Motion with jump-diffusion and volatility regimes for instant offline simulation and stress testing.
- **Bloomberg / TradingView Terminal UI**: High-density dark UI with Chart.js, live daemon health probing, confidence gating, strategy criteria editor, trade blotter, audit log, and CSV export.

---

## 🏛️ System Architecture

```
layquant-studio/
├── backend/
│   ├── server.py              # FastAPI microservice with Laya System 1 & market proxies
│   └── requirements.txt       # fastapi, uvicorn, laya, pydantic, requests, pandas, yfinance
├── frontend/
│   └── index.html             # High-density TradingView/Bloomberg UI + Chart.js
├── run.bat                    # Windows one-click launcher
├── run.sh                     # Unix/macOS one-click launcher
└── README.md                  # System documentation
```

### 1. Backend Microservice (`backend/server.py`)
- **FastAPI** daemon running on `http://127.0.0.1:8000`.
- **Endpoints**:
  - `GET /`: Serves the complete web terminal directly from the microservice.
  - `GET /health`: Returns service status, loaded model name, device (CPU/CUDA), and engine type.
  - `POST /v1/systemone`: Ingests market state (ticker, price, RSI, SMA distances, volume spikes) and typed criteria questions; executes non-autoregressive parallel evaluation and returns choice (`buy`, `sell`, `hold`), calibrated confidence score ($1 - H(p)/\ln(k)$), and sub-35ms latency telemetry.
  - `GET /api/market/binance`: Proxy for public Binance Klines with synthetic fallback.
  - `GET /api/market/yfinance`: Historical US equity data using `yfinance`.
  - `GET /api/market/synthetic`: Stochastic jump-diffusion market generator.
  - `GET /api/paper/status`: Live 24h paper trading telemetry, elapsed duration, PnL, and positions.
  - `POST /api/paper/start`: Starts unattended 24-hour background paper trading loop.
  - `POST /api/paper/stop`: Pauses background trading loop.
  - `POST /api/paper/step`: Triggers an immediate step evaluation.
  - `POST /api/paper/reset`: Resets paper wallet to $10,000.00.
  - `GET /api/paper/export`: Downloads the paper trading trade log CSV.

### 2. 24-Hour Live Paper Trading Engine (`backend/paper_trader.py`)
- Continuously polls unauthenticated live market feeds (default: Binance `BTCUSDT` on a 1-hour interval `1h`).
- Computes real-time Wilder RSI(14), Fast SMA (10), Slow SMA (30), and volume spikes.
- Queries the Laya System 1 microservice for non-autoregressive decision making.
- Executes paper orders with realistic maker/taker fees (0.075%), slippage (3 bps), and automated percentage stop-loss execution (3.0%).
- Automatically persists all state and hourly snapshots to `data/paper_trading_state.json` and logs executed trades to `data/paper_trades.csv`.
- Runs unattended for 24 hours (or configurable duration) and can be monitored in real time via the UI.

### 3. Quantitative Decision & Confidence Scoring
Confidence scores are calculated using **Normalized Shannon Entropy**:
$$H(p) = -\sum_{i=1}^k p_i \ln(p_i)$$
$$\text{Confidence} = 1 - \frac{H(p)}{\ln(k)}$$
where $k$ is the number of discrete action choices ($k = 3$ for `buy`, `sell`, `hold`). When the model distribution is confident and peaked, $H(p) \to 0$ and $\text{Confidence} \to 1.0$. When the model is uncertain, $H(p) \to \ln(3)$ and $\text{Confidence} \to 0.0$.

### 3. Stochastic Market Simulator
The simulation engine supports Geometric Brownian Motion with Jump-Diffusion:
$$S_{t+1} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z + J_t \right)$$
where $Z \sim \mathcal{N}(0, 1)$, and $J_t \sim \mathcal{N}(0, \delta^2)$ represents Poisson jump arrivals for modeling black swan volatility shocks and regime changes.

---

## 🚀 Quickstart Guide

### Option 1: One-Click Launchers

#### On Windows:
Double-click `run.bat` or run:
```cmd
.\run.bat
```

#### On Linux / macOS:
```bash
chmod +x run.sh
./run.sh
```

### Option 2: Manual Terminal Commands

1. **Install Dependencies**:
```bash
pip install -r backend/requirements.txt
```

2. **Start the LayaQuant Microservice**:
```bash
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

3. **Access the Terminal**:
Open your browser to:
[http://127.0.0.1:8000](http://127.0.0.1:8000) (or open `frontend/index.html` directly).

---

## 📊 Terminal Features & Controls

| Feature | Description |
| :--- | :--- |
| **Live Daemon Status Pill** | Probes `http://127.0.0.1:8000/health` every 5 seconds, displaying roundtrip latency and active model engine. |
| **Mode Switcher** | Seamlessly toggle between **Laya Daemon**, **Browser Emulator** (client-side Shannon entropy), and **Custom URL**. |
| **Market Selection** | Free Binance public feeds (`BTC/USDT`, `ETH/USDT`, `SOL/USDT`), equities (`SPY`, `NVDA`), and synthetic stochastic regimes. |
| **Risk & Gating** | Configurable confidence gate (50%–95%), technical pre-scanner, capital allocation, commission fees, slippage, and stop-loss. |
| **Strategy Criteria** | Natural language criteria editor for `buy`, `sell`, and `hold` with quick presets (Mean Reversion, Trend Breakout, High Confirmation). |
| **Interactive Chart** | Smooth toggle between Price + SMAs + Signal Triangles and Strategy Equity vs. Buy & Hold Benchmark. |
| **Performance Metrics** | Net Profit, Benchmark Return, Alpha, Annualized Sharpe Ratio ($R_f=2\%$), Max Drawdown (MDD), Win Rate, and Profit Factor. |
| **Trade Blotter & Audit** | Chronological trade journal, step-by-step decision audit logs, raw JSON request/response inspector, and 1-click CSV export. |

---

## 🛡️ License

MIT License — 100% Free and Open Source.

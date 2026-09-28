# 🚀 LayaQuant Studio — Master Project Handover & Architecture Blueprint

> **Document Version**: 2.5 (Production Hardened)  
> **Last Updated**: September 25, 2026  
> **Prepared For**: Claude Opus / Autonomous Senior Quant Engineer  
> **Repository Path**: `c:/Users/mansour/Documents/antigravity/gallant-bell`

---

## 0. READ FIRST — Strategy Status (2026-09-25)

**The 5m dual-model scalper described in §4 has no edge.** `backend/backtest_live_strategy.py` replays its exact
signal code on 90 days of Binance USD-M 5m data (8 symbols): gross expectancy +0.008R/trade (PF 1.03, i.e. noise);
after fees/slippage −0.09R/trade. No exit/filter variant tested was profitable. Its paper P&L looked good only
because of simulator bugs (fixed — see §5.5).

**The live paper trader now defaults to `strategy_mode = "TREND"`** (`backend/trend_strategy.py`):
4h Donchian 120-bar breakout, long-only, 60-bar channel exit, 4×ATR(20) chandelier stop, 1% equity risk/trade,
≤1× equity notional per position, ≤3× gross. Taker fills + slippage, funding charged at each 8h settlement.
The old scalper is still available: `POST /api/paper/set_strategy_mode {"mode": "SCALPER"}`.

Validation (`python -m backend.backtest_trend`, real klines + real funding history, 2020-01 → 2026-09):

| | In-sample 2020-24 | Out-of-sample 2024-07 → 2026-09 |
|---|---|---|
| Selected config (chosen on IS Sharpe only) | CAGR 49%, Sharpe 1.65, maxDD −19% | **CAGR 30%, Sharpe 1.17, maxDD −26%** |
| BTC buy & hold | CAGR 60%, Sharpe 1.07, maxDD −77% | CAGR 14%, Sharpe 0.52, maxDD −53% |
| Same exits, random long entries (5 seeds) | — | Sharpe 0.2–0.9 (avg ≈0.66) |

- All 42 grid configs (1d/4h, N, stop, long/long+short) had positive OOS Sharpe (0.34–1.29) → edge is not a single lucky setting.
- The random-entry control shows a large part of the return is long crypto drift + cut-losers/let-winners-run exits; the breakout entry roughly doubles it.
- Expect long flat/down stretches (2022: −2.5%; 2025 −9% for the daily variant). Win rate is ~35–45%; profits come from a few large trends.
- 2.2 years OOS is still a small sample. Paper-trade it before risking money; do not raise risk above 1% on this evidence.

**Can it do €500 → €10k in months? (`python -m backend.growth_study`)** Same rules, leverage caps raised to 10×/position
and 20× gross, liquidation modelled, a fresh $500 account started every 2 weeks from 2020-03 to 2025-08 (144 overlapping
12-month windows):

| Risk/trade | $10k within 6 mo | within 12 mo | Wiped out | Median after 12 mo |
|---|---|---|---|---|
| 1% (live) | 0% | 0% | 0% | $681 |
| 5% | 6% | 17% | 0% | $1,149 |
| 10% | 11% | 24% | 3% | $1,371 |
| 15% | 17% | 31% | 35% | $598 |
| 20% | 19% | 26% | 58% | $192 |

- Nearly all windows that reached $10k started in 2020 (the 2020-21 bull run); no window starting in 2022 or 2025 got there at any risk level.
- Out of sample (2024-07 → now) no risk level reached $10k; the best was 5% risk → $1,915 with an −81% drawdown on the way.
- Growth peaks around 5–10% risk. Above that the median result falls and the chance of being wiped out rises quickly.

**Does Laya improve entries? (`python -m backend.laya_filter_test`)** No. On all 452 trend entries since 2020-04:
- The prompt `server.py` sends today (RSI/SMA criteria) says "buy" on only 4 breakouts. Used as a gate it would skip
  trades worth +268R and keep 4 losers (−1.8R), because its criteria call RSI > 65 "sell" and breakouts have high RSI.
- A trend-specific prompt ("will this breakout continue?") has zero correlation with the outcome (+0.005). Filtering on it
  did no better than skipping the same number of trades at random.
- Laya is a general text classifier (ModernBERT, trained on email triage / intent / NLI), not a market model. In TREND mode
  the paper trader does not call it at all; in SCALPER mode it is only the fallback when `neural_dream` fails.

**What the goal needs.** At the growth-optimal (Kelly) leverage, a strategy with Sharpe S grows the median account by
about exp(S²/2) per year, and at full Kelly there is a 50% chance of halving along the way. 20× in 12 months needs S ≈ 2.5,
in 6 months S ≈ 3.5. The live trend strategy (S ≈ 1.2 out of sample) tops out near 2× a year, which matches the growth
study (best out-of-sample result: 5% risk, +86%/yr, −81% drawdown).

**News safety brake (`backend/news_guard.py`), shadow mode.** Laya answers yes/no questions per headline (hack, exchange
withdrawal halt/insolvency, delisting, regulator action, depeg, outage); market-wide and outage flags also need a keyword.
Correct on 26/26 hand-labelled headlines, but replayed on 2020-2024 news (`backend/news_guard_backtest.py`, 229k-headline
CoinDesk/CryptoCompare archive) it would have skipped 105 of 344 trend entries whose average was *better* than the rest
(+0.91R vs +0.61R): real news is full of "Celsius exits bankruptcy" / "Tether freezes wallets" noise, and breakouts rarely
happen during real crashes (there was no entry during FTX). So it only logs and shows NEWS FLAG on the scanner;
`POST /api/news_guard/enforce?on=true` makes it block entries. `python -m backend.news_guard --report` judges the live log.

**Other strategies (`backend/strategy_lab.py`, point-in-time universe from `backend/universe_data.py`, delisted coins
included).** Out-of-sample (2024-07 → now) Sharpe; parameters chosen on 2020-06 → 2024-06 only:

| Strategy | In-sample Sharpe | Out-of-sample Sharpe |
|---|---|---|
| Trend, live 8 coins (baseline) | 1.68 | **1.17** |
| Trend, top-30 by volume | 0.88 | 0.54 (coins enter the top 30 during pumps, breakouts buy the top) |
| Cross-sectional momentum, best in-sample setting | 1.11 | −0.02 |
| Funding carry, best in-sample setting | 4.21 | −0.52 (funding premia compressed since 2024) |
| Equal-risk combination of the three above | 3.42 | −0.33 (−28%/yr, −77% drawdown) |
| Trend ensemble (9 lookbacks, vol-sized, Zarattini 2025), 8 coins | 1.74 | 0.42 |
| Same, monthly top 20 | 0.68 | 0.59 |

The combination row is the trap to avoid: in 2024 it looked like the Sharpe-3 system the goal needs, then lost 77%.

**Trained entry filters (`backend/ml_lab.py`).** 1,724 trend trades on the top-30 universe, models retrained each year on
trades that had already closed. LightGBM (AUC 0.49) and logistic regression (0.50) found nothing. Laya's frozen encoder
embeddings of the features as text + a logistic head looked real at trade level (kept +0.29R vs dropped +0.01R; random
does as well 1%; stable across regularisation; not coin identity). But on the live 8-coin strategy with every breakout
signal scored (`--portfolio`, 42% of signals skipped) it is no better than skipping at random: Sharpe 2021-24H1 1.76 vs
1.71 unfiltered, 2024H2+ 1.09 vs 1.17, random skips 0.95–1.19. A skipped breakout usually re-triggers on the next bar,
so the filter mostly delays entries. A full fine-tune of the 421M-parameter model on ~1.7k trades would overfit; not
worth doing. No filter is used live.

**Long/short with separate levers (`backend/levers_lab.py`).** `TrendParams.short_risk_pct` lets shorts use their own
risk (`allow_long` switches longs off). The short book alone loses in both periods (Sharpe −0.14 in-sample, −0.27 out of
sample); it only helps in bear years (+15% 2022, +13% 2026 so far). Adding shorts lowers out-of-sample Sharpe at every
lever: long-only 1.17, +0.25% shorts 1.09, +0.5% 0.98, +1% 0.75. The live bot stays long-only.

**Deposit plan: 500 + 100/month → 10,000 (`levers_lab.py`, paper trader `POST /api/paper/recurring_deposit`).**
At a steady +30%/yr it takes 49 months; +50%/yr 40 months. Bootstrapping 30-day blocks of the out-of-sample returns
only: 2% risk → 50% chance within 36 months, 3% → 67% (median 28 months), with a ~1-in-4 chance of being below the
money deposited after 24 months.

**Hype / attention / sentiment (`backend/hype_lab.py`, `backend/telegram_data.py`).** Measured in the 72h before each
trend entry and split into thirds, with a permutation test: news-headline attention, Laya's tone and hype of those
headlines, English Wikipedia pageviews, the @WatcherGuru Telegram channel (attention, Laya tone/hype) and Whale Alert
exchange in/outflows for the coin and for USDT+USDC. Nothing is both significant and stable. News attention looked
significant before 2024-07 (high attention worse, p≈1%) and flipped after; Wikipedia attention looked very significant
after 2024-07 (rising attention better, p<0.1%) but was the other way round in 2020, 2023 and 2024. Measures that flip
sign between periods would hurt as filters. Public Telegram channels can be read with history via t.me/s pages (no
login); X needs a paid API and Reddit's API refuses these requests.

**Free insight sources (`backend/insight_lab.py`).** Binance futures metrics (open interest, top-trader / all-account
long-short ratios, taker buy/sell ratio; data.binance.vision daily files), Fear & Greed (alternative.me), stablecoin
supply (DefiLlama), Deribit DVOL and the Coinbase premium, each taken the day before every trend entry. Long/short
ratios and 30-day OI change flip sign between periods. Fear & Greed, taker ratio, DVOL, Coinbase premium and stablecoin
growth keep their sign but are only significant in one period each. A blend chosen on 2020-03 .. 2024-06 data only
(`--composite`) gave Sharpe 1.23 vs 1.17 unfiltered after 2024-07, inside the 1.01–1.23 range of skipping the same share
of signals at random: a hint, not an edge. Not used live.

**Attention dilution.** The number of active Binance perps (tokens competing for the same attention and money) sorts
trend outcomes the same way in both periods (fewer perps → better trades), but it only rises, so it is a time trend,
not a filter. It is consistent with the edge shrinking as the market fills up: +0.68R per trade before 2024-07,
+0.43R after. Expect somewhat less than the backtest. Attention measures in `hype_lab.py` also come as shares of
attention across the 8 coins (`*_share`), so a market-wide frenzy is not read as coin-specific hype.

**Shorting new listings (`backend/listing_lab.py`).** Short every new USDT perp 1–7 days after listing, hold 14–30
days, stop at +30/60%. Listings before 2024-07 (220): mean −2.9% to +0.9% per trade, no edge. Listings since (~550, the
2024-26 memecoin/AI flood): +2.3% to +3.3% per trade in every setting. Chosen on the earlier data it would have been
rejected; regime-dependent and exposed to squeezes, so paper-trade only if at all.

**Small-capital edges.** Edges too small for funds are the natural place for a small account; tested:
- Exchange announcements (`backend/event_lab.py`; events from the official Telegram channels @binance_announcements
  2017 on, @upbit_news 2025-05 on, @bithumb_notice 2026-05 on; traded on the Binance perp with 1-minute bars, entry
  60-120 s after the post, 0.7% round-trip costs + funding). Upbit listings: price is already up a median +13% before
  a normal bot can enter; buying loses (−11% over 24h). Fading them (short, hold 24h): +7.2%/trade, 70% win, n=27, but
  +12% in the first half of events vs +2% in the second, stops at 10/20% turn it negative, and it has to sit through
  squeezes of up to +43%. Binance delistings, short and hold 4h: +3.6%/trade after funding, 63% win, n=67, stable
  across halves (+3.4% / +3.8%) — the best find, but ~27 events a year and fat tails (one +180% squeeze inside 4h), so
  sized to survive a 200% squeeze it adds only a few % a year. Binance listing fades and Bithumb listings (n=18): no
  edge. The public Upbit notice API is Cloudflare-blocked from the research server.
- Daily reversal / momentum in the top 50-100 coins (`backend/smallcap_lab.py`): both directions lose after
  0.15%/side costs (turnover 56-104% a day).
- Shorting new listings (`listing_lab.py`, above): only since 2024-07.
The trend strategy already runs fine on €500 (positions of roughly €50-150); for a small account the deposit rate and
the risk level move the result far more than any of these.

**Final hype/attention run** (all Whale Alert history 2019 on, `*_share` measures): still nothing consistent. Share of
attention flips sign between periods for news, Wikipedia and Telegram; whale exchange inflows and stablecoin inflows
keep their sign but are not significant after 2024-07.

**Stocks + crypto: the one thing that raised the Sharpe (`backend/cross_asset_lab.py`).** Time-series momentum
on ETFs (hold an ETF while its 12-month return is positive, volatility-sized, monthly; Moskowitz-Ooi-Pedersen 2012)
earns on its own (12 ETFs 2007-now: +15.5%/yr, Sharpe 0.83, max DD −39%) and is nearly uncorrelated with the crypto
trend book (daily correlation +0.09). Equal-risk mix, weights fixed in advance: out-of-sample Sharpe 1.78 vs 1.17 for
crypto alone (the ETF book had unusually good 2024-26 years). Stricter check, drawing ETF months from 2007-now
(2008 and 2016-18 included) and crypto months from 2024-07 on, 500 + 100/month, at 2% crypto-equivalent risk: 10k
within 36 months 68% (crypto alone 50%), below the deposits at 24 months 9% (crypto alone 21%). Restricted to what
Binance Futures lists as USDT "TradFi" perpetuals since 2026 (SPY, QQQ, gold XAU, silver XAG; operated by
ADGM-regulated Nest Exchange) the numbers are nearly the same (ETF book Sharpe 0.88, correlation +0.08), so the same
bot and account could run both books. Not built into the paper trader yet; TradFi perp funding and regional
availability still to check.

**The plan tool (`frontend/plan.html`, http://127.0.0.1:8000/plan; paper money).** The user picks budget, risk
level (1-3), monthly deposit and target; `POST /api/plan/start` resets the paper account and runs both books:
- Crypto trend (unchanged rules, 8 coins) at risk per trade = 0.68% x level.
- TradFi trend (`backend/tradfi_book.py`): XAUUSDT, XAGUSDT, SPYUSDT, QQQUSDT perps. Once a month, long an asset
  while its ETF's (GLD/SLV/SPY/QQQ, Yahoo Finance) 12-month return is positive, sized to (14.6% x level)/sqrt(4)
  yearly volatility each; caps 1.5x equity per asset, 3x total; positions within 20% of target are left alone.
  Marked at the perps' Binance mark price, funding charged, taker fee + slippage on every change.
- Level mapping makes both books carry equal risk as in `cross_asset_lab.py`. Crypto entries only count crypto
  positions against their leverage cap; the scalper exit manager skips TRADFI positions.
- Recurring deposit every 30 days; `/api/plan/pause`, `/api/plan/resume`, `/api/plan/status`. The plan and its
  risk level persist in `data/paper_trading_state.json`.
If Yahoo or the Binance TradFi quotes are unavailable, the rebalance waits and retries every 15 minutes (shown on the
page). Funding is charged at each symbol's real settlement interval (`/fapi/v1/fundingInfo`: 1h/4h/8h).

**Funding matters for the TradFi book.** Longs paid, Jan-Aug 2026: gold +12.1%/yr of notional, silver +23.0%,
S&P 500 −6.1% (longs were paid), Nasdaq +0.2%. The ETF backtest had left this out. With it (`cross_asset_lab.py`,
`FUNDING_2026`), the TradFi book's 2007-now Sharpe drops from 0.88 to 0.68 (T-bill financing instead: 0.78), and the
combined out-of-sample Sharpe from 1.93 to 1.74, still well above crypto alone (1.17). Level 2 odds (500 + 100/month,
stricter bootstrap): 10k within 36 months 62% (was 70%), below deposits at 24 months 12% (was 9%). Yearly costs at
level 2: crypto ~1.5% fees + ~1.8% funding; TradFi <0.5% fees + ~8% funding at 2026 rates with September 2026 weights. Tested with mocked prices (entries, rebalance, trend flips, deposits, reload) and a live server smoke test;
Binance is geo-blocked from the research server, so live quotes were not exercised there.

**Telegram notifications and the local journal.**
- `backend/notifier.py` holds the bot token and chat id. They are stored only in `data/telegram_config.json` (never
  committed); `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` env vars override the file.
  - `POST /api/telegram/connect {token}` checks the token and waits up to 90 s for the user to message the bot, which
    gives the chat id.
  - Messages go through a background queue, so Telegram never holds up trading.
- `backend/plan_reporter.py` is called from the paper trader on every trade, funding charge, deposit, rebalance and loop
  error, and once per loop (`tick`). It writes:
  - `data/journal/equity.csv`: one row per hour (equity, deposited, trading P&L, exposures, drawdown, fees, funding).
  - `data/journal/events.csv`: every trade, rebalance, deposit, funding charge, alert and plan start/pause/resume.
- Messages sent: plan start/pause/resume, crypto entries and exits, the monthly TradFi rebalance, deposits, one alert
  per newly crossed drawdown level (10/20/30/40%, compared with the backtest's worst drawdown for the risk level), the
  target reached, TradFi data down for over an hour, loop errors (max one per hour), and a weekly summary (Sunday
  18:00 UTC, optional daily). The deposit raises the drawdown peak so it doesn't hide a loss.
- `GET /api/plan/journal` feeds the plan page's equity-vs-deposits chart and events table; `/api/telegram/status`,
  `/settings` and `/test` serve the page's Telegram card.
- Fixed at the same time: the plan used the default 720 h run length and would have stopped itself after 30 days; it
  now runs with a 10-year limit.

**Internet/Reddit survey (2026-09).** LLMs trading on their own lost money in the Alpha Arena live contest (4 of 6
models down; Claude −42%, Gemini −46%); Reddit "ChatGPT trader" stories are weeks-long and unverified; bots
advertising ~1%/day are not credible. AI reading news for small stocks (Lopez-Lira & Tang) is documented but
reportedly decayed after 2023, and free stock-news data (FNSPID) ends in 2023, so it can't be checked on recent years.
Prediction markets were researched and dropped at the user's request (`backend/predmarket_lab.py` unused).

**LLMs reading breaking news (2026-09-27, `backend/news_llm_lab.py`, `news_llm_score.py`, `news_llm_job.py`).**
- **Setup.** Every @WatcherGuru post, 2022-02 .. 2026-09 (12,465 after removing price ticks and ads), was shown to an
  instruction-tuned LLM. The model answered one letter, A (strongly down) .. G (strongly up), for Bitcoin over the
  next few hours; the score is the probability-weighted answer taken from the log-probabilities.
- **Rules fixed before any result.**
  - Test window 2025-01 .. now, after all models' training data.
  - Entry on the BTCUSDT perp 1-2 minutes after the post; 0.15% round-trip costs; one position at a time.
  - Pass: profitable in 2025 and in 2026, p < 0.01 against random directions, and better than following the last
    15 minutes' move.
- **Models.** Qwen2.5 7B, 32B (AWQ) and 72B (AWQ). Then, at the user's request, only models that fit an RTX 3050
  with 4 GB: Qwen2.5-1.5B and 3B (AWQ 4-bit), Llama-3.2-3B, Phi-3.5-mini.
  - Scored unattended on rented Vast.ai GPUs through the Vast REST API; SSH is blocked by the research server's proxy.
  - The job receives posts and serves results over HTTP; plain HTTP to the instance works only through a CONNECT
    tunnel (`curl --proxytunnel`).
  - Total cost $1.37.
- **Result: nothing, for every model size.**
  - Spearman correlation of score with the next 15m / 1h / 4h / 24h return is within ±0.02 (±0.04 at 24h).
  - Every rule lost almost exactly the 0.15% costs per trade.
  - There was no signal even on 2022-24 news inside the models' training data.
  - The scores mostly echo the move that already happened (correlation +0.12 to +0.17 with the 15 minutes before
    entry).
- **Follow-ups (`backend/news_llm_creative.py`), each built on 2022-24 and checked unchanged on 2025+.** All failed:
  - Under- or over-reaction: big news with no move yet, or a big move on news rated minor.
  - Gradient boosting on all models' answer probabilities: train AUC 0.68-0.72, test 0.50-0.52, +0.01% gross per
    trade.
  - Headline embeddings (bge-small) with ridge regression: test IC 0.00.
  - "Impact" as a volatility warning: adds only +0.03 Spearman beyond the last day's volatility.
  - Daily average tone vs the next 1-3 days: sign flips between periods.
  - Altcoin headlines traded on the named coin (one position per coin): nothing clears t = 1 in 2025+ for the
    small models.
  - The 72B model looked promising on altcoin headlines at 4h: +0.15% net per trade, t = 2.1 in 2025+. That is below
    the p < 0.01 bar, came out of many tries, and the model cannot run on the user's PC; it was not pursued.
- The big-model scores are kept in `data/news_llm/too_big_for_3050/`.

**Time-series foundation model (`backend/chronos_lab.py`).**
- Amazon Chronos-Bolt small and base (released 2024-11; run on CPU) forecast the 8 coins' next 24h from 512
  4-hour bars, once a day, 2025-01 .. 2026-08.
- Direction: IC −0.004 to +0.008 (4h: −0.005 / +0.033, inconsistent between halves). Long/flat on the forecast:
  −15% and −25% a year vs −19% holding all 8.
- Volatility: slightly better than the last 30 bars' volatility (Spearman with the next 24h absolute move 0.33 vs 0.31,
  better in both halves), but too small to change position sizing.

**Published / popular ideas checked on recent data (`backend/alt_lab.py`, 2026-09-27).** None is usable:
- **Volatility targeting** (exposure × target / 30-day market volatility, capped 0.25–2×). It does not help: crypto
  trend book Sharpe 1.17 → 1.07 after 2024-07; whole plan 1.74 → 1.79 but with deeper drawdowns (−24% → −28%). The
  trend rules already size each trade by ATR.
- **Bitcoin clock effects** (BTCUSDT 1-minute data, 2022-02 .. now).
  - Quantpedia's "long 22:00–24:00 UTC" (Sharpe 1.58 on 2015–2021) has faded: +0.03% gross per trade 2022–24,
    −0.01% since 2025, a loss even with limit orders.
  - Intraday momentum (the 00:00–00:30 sign held for 23:30–24:00): nothing.
  - "Monday Asia open" (Sun 23:00 → Mon 23:00 UTC): +0.21% above other days 2022–24 (t 0.7) and +0.42% since 2025
    (t 1.5). Tuesday is similar, so it is not significant.
- **Crash rebound** (buy after a 4h bar falls ≥ 4 standard deviations, hold 24h; chosen on 2020–24).
  - On the live 8 coins it looked real: +1.8% then +2.0% per trade.
  - On the point-in-time top 30/50 with delisted coins, and with trades on the same bar grouped, it is gone:
    +0.1% per crash bar 2020–24 (t 0.1).
  - Since 2025, 3 crash bars make more than all of the profit. Losses include LUNA −98% and several 2026 tokens
    −60% to −83%.
  - The 8-coin result was survivorship: those coins are, by construction, the ones that recovered.
- **Funding carry** (long spot, short perp).

  | Coin | 2022 | 2023 | 2024 | 2025 | 2026 Jan–Aug |
  |---|---|---|---|---|---|
  | BTC | +4.2% | +7.9% | +12.0% | +5.1% | +1.7% |
  | ETH | +0.8% | +8.3% | +13.0% | +4.9% | +1.0% |

  That is now about savings-account yield, plus exchange risk.

**Selling Bitcoin volatility (`backend/vol_premium_lab.py`).**
- **Setup.** Each Friday, sell a 7-day BTC straddle, or an iron fly with wings ±10%, priced by Black-Scholes from
  Deribit DVOL. Costs: Deribit fees, a vol-point spread and skew on the bought wings. 2021-04 .. 2026-08.
- **The premium is real but shrinking.** DVOL was above the next week's realised volatility in 82% of weeks before
  2024-07 (68% vs 55% on average) and 71% after (48% vs 43%).
- **Priced at DVOL, it looked like a diversifier.**
  - Iron fly alone after 2024-07: Sharpe 0.93, worst week −7.5%.
  - Correlation with the plan: about 0.
  - Plan plus half iron fly: Sharpe 2.01 vs 1.81, drawdown −11% vs −19%.
- **At realistic prices the edge is gone.**
  - A live Deribit snapshot (2026-09-27) showed 5-day ATM IV at 31% vs DVOL 34.7%, and about 8% bid-ask on weekly
    options.
  - Pricing at DVOL − 3 points and 1.5 points spread per leg: iron fly Sharpe 0.27 after 2024-07, and the plan plus
    iron fly drops to 1.61.
  - At DVOL − 5 points it loses money.
- **Timing with a forecast hurt.** Selling only when implied / forecast volatility (30-day realised, or Chronos-Bolt)
  is above its in-sample median did worse than always selling.
- Not pursued. The premium has been competed away for small traders.

**Drug-trial results and biotech stocks (`backend/biotech_lab.py`, `biotech_score.py`, 2026-09-27).** Chosen because the
user is a pharmacist: a domain edge plus a small local model to read announcements.

- **Data.**
  - SEC EDGAR full-text search: 8-Ks from SIC 2833–2836 / 8731 mentioning "topline results/data", "met (its/the)
    primary endpoint", "did not meet …" or "primary efficacy endpoint", 2015-01 .. 2026-09.
  - The SEC requires a contact in the User-Agent; the user's email is used with their consent.
  - 14,654 filings from 832 companies, 7,748 after dropping earnings releases.
  - Yahoo prices exist for about 55%. Delisted, merged or renamed companies are missing, and the bias direction is
    unknown.
  - Returns are abnormal vs XBI. Entry at the close of the day after the filing; hold 20 / 60 trading days.
- **Classification.**
  - A keyword baseline.
  - Qwen2.5-3B-Instruct-AWQ (fits the user's RTX 3050 4 GB) on a rented GPU. It answered outcome, phase and evidence
    strength ("strong / moderate / weak or spun") from letter log-probabilities.
  - A Phi-3.5-mini run was stopped to save money once the result was clear.
- **Rules fixed in advance: all fail.**
  - Drift in the result's direction: long positive / short negative, 60 days. −1.7% / −5.5% (keywords) and
    −2.4% / −3.3% (Qwen) per trade, 2015–21 / 2022+.
  - The paper's size effect: short large-firm failures, buy small-firm failures. About 0.
  - Short small-firm positives the model rates weak or spun: +4.3% (t 1.1) / +7.0% (t 1.5). No better than shorting
    every small-firm positive; "strong" and "weak" ratings drift the same.
- **The model reads data quality like the market does.** Small-firm positives rated "strong" jumped +20–23% on the
  news; those rated "weak" moved about 0.
- **Found by slicing.** Small-firm positive readouts lag XBI over the next 60 days.
  - Keywords: −5.4% (t −1.5) then −15.1% (t −3.5).
  - Qwen: −3.7% (t −1.1) then −6.7% (t −2.3).
  - The same stocks did not lag in a window months earlier (+2% / +8%).
  - Not explained by share offerings filed within 10 days (424B). It was worse without one.
  - The "lukewarm first reaction" variant does not hold on Qwen's events in 2015–21.
- **As a portfolio it is untradable.** Calendar-time short with an XBI hedge, 0.5% costs and 15%/yr borrow:
  - Keywords: −20%/yr (max DD −89%) in 2015–21, +40%/yr (max DD −64%) in 2022+.
  - Qwen: −12%/yr (−80%) then +6%/yr (−58%).
  - Squeezes in small biotechs dominate.
- **Usable only as a warning.** Don't chase small biotechs after "positive" topline news; on average they lag the
  sector for the next three months.
- Total Vast.ai spend in this session: about $1.90.

**Buying at the start of a breakthrough (`backend/biotech_start_lab.py`).**
- **Setup.** 8-Ks announcing FDA designations (Breakthrough Therapy, Fast Track, Orphan, Priority Review, Rare
  Pediatric, RMAT) or research starts (IND clearance, first patient dosed, Phase 3 start), 2015-01 .. now.
  - The event must be in the headline or lede.
  - 1,320 events, 62% with prices.
  - Returns are XBI-adjusted from the day after the filing, compared with the same stocks' abnormal return in an
    earlier window.
- **Main hypothesis fails.** Breakthrough Therapy: +11–12% on the announcement, then −10% (t −1.3) / −0.6% over 60
  days (2015–21 / 2022+).
- The other designations show no consistent drift; Priority Review is −5% / −8% ("sell the news").
- **Research starts are a lottery.**
  - IND clearance: +25% / +8%. First patient dosed: +5% / +19% (t ≤ 1.4).
  - Median −4% / −10%; without the best 3 events per period, about 0%. The best 3 were +143% to +606%.
  - Long portfolio hedged with XBI: +11% then +26% a year, max DD −86% / −71%.
  - Survivorship (dead companies missing) flatters it. Not usable.

**FDA decision-date run-up (`backend/biotech_pdufa_lab.py`).**
- **Setup.** PDUFA dates were read from 8-K exhibits (680 company/date pairs; 51% tradable, i.e. priced and public
  40+ trading days ahead). Rule: buy 40 trading days before, sell the day before, vs XBI.
- **The run-up is not specific.**
  - +4.4% (t 1.5) / +4.8% (t 2.2), net +3.9% / +4.3% (2015–21 / 2022+).
  - The same stocks' control window was +3.7% / +3.9%, so only about +1% is specific to the run-up; the rest is
    survivorship.
  - Portfolio: −1%/yr (max DD −78%) then +18%/yr (−61%).
- **Decision day itself: −5.4% / −4.6% on average (t −3.7 / −3.5).** Holding through an FDA decision loses on average.

**Insider buying in biotech (`backend/biotech_insider_lab.py`).**
- **Setup.** SEC Insider Transactions Data Sets 2015–2026, restricted to the ~900 drug / biotech CIKs from the labs
  above. Signal: officer or director open-market purchases ≥ $10k; 4,947 signals, 53% with prices. Hold 60 days vs XBI.
- **Result.**
  - 2015–21: +3.1% (t 2.3), but the control was +3.2%. Cluster buys +5.3% (t 2.0).
  - 2022+: −0.6%, cluster −1.6%.
  - Portfolio: +13%/yr (max DD −26%) then −5%/yr. Decayed, and not insider-specific.

**Calendar effects (`backend/calendar_lab.py`).** SPY from 1993, QQQ from 1999, Bitcoin; before 2022 vs 2022+.
- **Overnight.** Before 2022, close→open carried all the gains: SPY overnight Sharpe 1.01 vs daytime 0.07. Since 2022
  overnight is 0.66 vs daytime 0.44, below holding the whole day (0.76). Break-even cost only 2.7–3.9 bp per round
  trip.
- **Turn of month:** +4.5 bp (t 1.3) before 2022, then +0.9 bp.
- **Pre-holiday:** positive but not significant.
- **FOMC announcement day:** +23.5 bp (t 3.0) before 2022, then −4.7 bp (SPY).
- **Bitcoin:** turn of month, weekend and monthly-expiry effects are nothing consistent. The funding-settlement
  30-minute drift was significant in 2022–23 only.

**New-listing shorts, market-controlled (`backend/listing_deep_lab.py`).** Wait 3 days, hold 21, stop +60%.
- Since 2024-07: +2.6% per trade, but shorting a basket of established perps over the same days made +3.2%, so the
  difference is −0.5% (t −0.35). The gain was the altcoin bear market.
- Before 2024-07, new listings did lag established coins (+5.5%, t 2.4), but that reversed in the last 5 quarters.
- Portfolio at 2% per short: −3%/yr then +14%/yr (max DD −23%).

**More TradFi markets (`backend/tradfi_extend_lab.py`).** Binance added oil (CLUSDT, BZUSDT), copper, platinum,
palladium and IWMUSDT in 2026.
- **Funding, longs pay per year:** oil −22% / −16% (longs are paid), copper +13%, platinum +18%, palladium +21%,
  IWM 0%.
- **Adding them does not help.**
  - All 9: TradFi book Sharpe falls from 0.58–0.74 to 0.34–0.51, and the plan's out-of-sample Sharpe from 1.64–1.74
    to 1.45–1.57.
  - Oil + copper only: 1.70–1.78 depending on funding assumptions, with a lower drawdown (−19% vs −24%) but lower
    10k odds (54–60% vs 61–63%).
- **Long/short TradFi** raises the book's own Sharpe (0.82–0.89) but not the plan's.

**Broker ETFs instead of perps for the TradFi half.** Unleveraged ETFs would lift the plan to Sharpe 1.92 and the
level-2 odds to 68%. But the TradFi book runs at about 1.4× total equity at level 1 and 2.5× at level 2. Financed at
margin or futures rates it is the same as perps: Sharpe 1.71–1.75, odds 58–59%. Perps are fine and the only option
for small accounts.

**Limit orders for the trend strategy (`backend/maker_lab.py`).**
- **Setup.** Entries and channel exits as limit orders at the signal bar's close (maker 0.02%, no slippage), resting
  one 4h bar, then market; stops stay market. The taker version reproduces the live results exactly.
- **Result.** All 439 entries filled at the limit. Fees −28%. +0.7–0.9%/yr (52.6→53.3% in-sample, 29.6→30.5%
  out-of-sample), Sharpe +0.02.
- Worth building into the live bot when real orders are added; not a breakthrough.

**Small LLMs reading financial statements (`backend/fin_statement_lab.py`, `fin_statement_score.py`).** After Kim,
Muhn & Nikolaev 2024.
- **Setup.** SEC Financial Statement Data Sets (Q1+Q2 of 2016–2026): 55,815 10-Ks. Target: next year's diluted EPS
  higher than this year's. 27,069 firm-years with an outcome.
- **Baseline.** Logistic regression on standard ratios, trained FY2016–21, tested FY2022–24 (10,472): AUC 0.596,
  accuracy 56.6% (base rate 52.9%).
- **Models.** Qwen2.5-3B-AWQ and Phi-3.5-mini (fit an RTX 3050), zero-shot on anonymised two-year tables, letter
  log-probabilities.
- **Result: worse than random.**
  - Qwen2.5-3B: AUC 0.457, accuracy 46.1%. Phi-3.5-mini: AUC 0.453, accuracy 45.4%.
  - Both say "lower" about 80% of the time.
  - Added to the baseline: +0.001 AUC (95% −0.001 .. +0.002).
- The published result needs a GPT-4-class model with step-by-step reasoning. Stopped at stage 2, no returns test.

**Token unlocks from CoinGecko supply jumps (`backend/unlock_lab.py`).** The DefiLlama emissions API is paid (HTTP
402) and its open-source adapters repo is outside this session's GitHub scope.
- **Setup.** Proxy: CoinGecko market cap ÷ price = circulating supply (keyless API: last 365 days only), 220 coins
  with Binance perps. Events: supply up ≥ 2% in a day; 432 events, 2025-10 .. 2026-09.
- **Next 14 days vs the equal-weight coin basket.**
  - All jumps: −2.9% (t −1.15), median −7.9%.
  - Jumps ≥ 5% (205): −5.8% (t −3.0), median −7.9%.
- **Checks weaken it.**
  - 31% of events fall on days with 3+ jumps (CoinGecko batch updates such as 2026-03-12 and 2026-02-10); clustered
    by day it is −3.2% (t −1.2).
  - First half of the year −8.1% (t −2.9), second half −2.4% (t −1.0).
  - An unhedged short with a +40% stop loses (−1.3% per trade).
  - Hedged (short the coin, long the basket, 3% per trade): +10% over the year, Sharpe 0.64, max DD −19%.
- **Suggestive, not proven.** Needs real unlock schedules (so shorts can start before the unlock) or months of forward
  tracking.

**Binance vs OKX funding arbitrage (`backend/funding_arb_lab.py`).** OKX keeps about 3 months of funding history;
Bybit blocks the server's region.
- 62 coins, 2026-07 .. 08: the average daily funding gap is 5.5%/yr.
- Holding the 7-day-signalled side when the gap is above 7%/yr, with 0.2% per switch: −0.1%/yr on average, positive
  for 10% of coins (small alts). No edge.

**Unlock watch, live and paper only (`backend/unlock_watch.py`).** Started with the server.
- **Daily check.** Once a day (after 01:00 UTC) one CoinGecko /coins/markets call saves price, circulating supply and
  market cap of the top 250 coins to data/unlock_watch/snapshots/.
- **Events.** Supply up ≥ 5% (and < 100%) since the previous snapshot, stablecoins skipped. Days with 3+ jumps are
  flagged "batch" and scored separately.
- **Paper trades.** Each event opens a 14-day paper hedged short (short the coin, long the equal-weight basket of
  tracked coins, 0.2% costs).
- **Alerts.** Telegram per event (with a warning for the trend strategy's 8 coins), per closed trade, and a weekly
  score.
- **Outputs.** data/unlock_watch/events.csv, GET /api/unlock_watch/status, and a card on the plan page.
- **Purpose.** Out-of-sample evidence for the supply-jump effect. Tested with synthetic snapshots and a live server
  smoke test.

**Stock / index perps outside US hours (`backend/tradfi_hours_lab.py`).** Binance's 2026 stock perps (SPY, QQQ, IWM
and 14 single stocks), 5-minute klines, 2,216 symbol-days.
- **At the open the perp matches the real gap.** Correlation +1.00, mean error +0.015%.
- **The rule fixed in advance fails.** Fading closed-hours moves above 1 sd, held to 11:30: −0.53% per trade
  (t −4.7), in both halves, weekends and weeknights (single stocks −0.62%; index perps about 0).
- **Following the move works only in the first five minutes.**
  - Enter 09:30: +0.42% (t 3.3).
  - Enter 09:35: +0.10% (t 0.8; +0.27% t 1.95 clustered by day).
  - Enter 09:40: 0.00%.
  - It is the perp catching up with the real opening auction, a speed game. After that, up-gaps drift up and
    down-gaps reverse (2026 market drift).
- Not tradable for a retail bot.

**Alt-season entry filter (`backend/alt_season_lab.py`).** Alt entries only while the 7-alt equal-weight index
priced in BTC is above its 30-day mean (on 45% of the time).
- 2020–24: 52.6% → 34.7%/yr, Sharpe 1.68 → 1.39.
- 2024-07+: 29.6% → 34.5%, Sharpe 1.17 → 1.36, DD −26% → −19%.
- Regime-dependent, so rejected by the in-sample rule.

**Bitcoin ETF flows:** blocked. Farside is behind a Cloudflare challenge and the alternatives are paid.

**Crypto session effects (`backend/session_lab.py`).** Six 4-hour UTC blocks, 8 coins, before 2024 vs 2024+.
- **Two blocks passed the in-sample test and faded.**
  - 04–08 UTC: +14.4 bp (t 3.4), then +1.4 bp.
  - 20–24 UTC: +10.1 bp (t 2.1), then +5.8 bp.
- **Trading them since 2024:** −11%/yr and +3%/yr with limit orders, much worse as taker.
- Since 2024 Bitcoin's gains came outside US hours (US hours −1.1 bp per 4h bar vs +2.9 bp, t 2.1), but that is too
  small to trade daily.

**Crypto-linked stocks vs Bitcoin (`backend/linked_lab.py`).** COIN, MSTR, HOOD, MARA and RIOT, with Bitcoin priced at
exactly 16:00 New York.
- **Daily lead:** |correlation| ≤ 0.10 and sign-inconsistent.
- **Hourly lead** (Yahoo 60m, last two years): about 0 either way; same-hour correlation 0.52–0.78.
- An earlier +0.50 "lead" was a timestamp bug: Yahoo stamps bars with their start time.
- **Snap-back** of stock/BTC at |20-day z| > 2: about 0 before 2024; only COIN since 2024 (+3.9%, t 2.6). Rejected.

**Mega-cap stock trend book:** not tested. Binance's stock perps are today's winners, so a backtest would be
hindsight-selected. 2026 funding on those perps averages about +5%/yr (CRCL +16%, GOOGL/AMD +11%).

**Plan overlays (`backend/plan_overlay_lab.py`).**
- **Adaptive 90-day inverse-vol weights:** Sharpe 1.59 / 1.70 vs 1.67 / 1.74, drawdowns −36% / −31%. Worse.
- **Drawdown brake (halve exposure below −X%).**
  - 10%: Sharpe 1.66 / 1.71, DD −17% / −18%, lower returns. The same trade-off as a lower risk level.
  - 15%: out-of-sample Sharpe 1.23. 20%: 1.46.
- Not adopted.

**Bitcoin around US macro releases (`backend/macro_lab.py`).** CPI and jobs-report dates from the BLS archive link
names; FOMC from the Fed calendars; Bitcoin 1-minute data, 2022-02 .. now.
- **Volatility.** The release hour moves 2–5× an ordinary hour: CPI 4.9× in 2022–23 and 2.5× since, FOMC 3.0× /
  2.1×, jobs 1.7× / 2.1×.
- **No directional drift passes.** Nothing had |t| ≥ 2 in 2022–23. The CPI reaction since 2024 (+37 bp, t 2.0) had
  the opposite sign before.

**Altcoin / Bitcoin pair trend (`backend/pairs_trend_lab.py`).** The live breakout rules on the log(alt/BTC) ratios of
the 7 alts, both directions, equal volatility per pair, costs on two legs, funding ignored.
- **Alone:** Sharpe 1.39 in-sample, then 0.37 out-of-sample (max DD −40%). Correlation with the crypto trend book
  +0.25 / +0.29.
- **Plan + pairs (equal risk):** Sharpe 2.03 in-sample vs 1.67, but 1.65 out-of-sample vs 1.74. It decayed; rejected.

**Crypto sector momentum (`backend/sector_lab.py`).** CoinGecko categories (today's membership), one sector per coin,
point-in-time top 100 perps.
- **Weekly long top 2 / short bottom 2 by 28-day return:** −2%/yr (Sharpe 0.27) in 2021–23, +7%/yr (Sharpe 0.39)
  since 2024.
- **Long the top 2** vs holding all sectors: the same Sharpe (1.04) in 2021–23, then −37% vs −43%/yr. No edge.

**Faster trend book: 1-hour breakouts with limit orders (`backend/fast_trend_lab.py`).** The live rules on 1h bars
(same bar counts, so 4× faster), 8 coins, maker_lab "limit_chase" execution. Kept only if plan + fast book beat the
plan in both periods.

| Book (own $500 account) | Sharpe in / out | Max DD in / out | Entries |
|---|---|---|---|
| 4h 120/60 (live), limit | 1.70 / 1.19 | −18% / −26% | 439 |
| 1h 120/60, limit | 0.51 / 0.85 | −47% / −52% | 1,810 |
| 1h 120/60, market orders | 0.39 / 0.77 | −51% / −53% | 1,810 |
| 1h 240/120, limit | 0.78 / 1.08 | −24% / −34% | 1,256 |

- **Correlation with the 4h book:** +0.60 / +0.62 (240/120: +0.66 / +0.71). It is the same trade, with more noise.
- **Plan + 1h 120/60:** Sharpe 1.39 / 1.60 vs the plan's 1.68 / 1.76. 240/120: 1.51 / 1.72. Replacing the 4h book with
  the 1h book: 0.86 / 1.51.
- Limit orders help on 1h too (+0.1 Sharpe), but not enough to make up for the whipsaws. Rejected.

**Binance Monitoring Tag shorts (`backend/warning_tag_lab.py`).** 139 coins tagged in 25 announcements
(2023-10 .. 2026-09) on @binance_announcements; 90–93 had a USD-M perp.
- **Pre-registered rule fails.** Short at the announcement day's UTC close, hold 7 days, minus the top-100 average:
  −0.6% (t −0.2). The first half was +4.9%, the second −9.3%; 2026 −9.1%. One squeeze reached +291% in 7 days.
- **The drop happens within hours.** Previous close → announcement-day close: −9.4% (t 5.6, 84% of coins fall).
- **Fast version (post-hoc, event_lab machinery).** Short 60–120 s after the post, 0.7% round-trip costs, funding.
  - 4h: +2.3% per coin. 24h: +4.4% per coin, 67% win, halves +5.5% / +3.4%.
  - Bitcoin-adjusted and averaged per announcement: 24h +2.7% (t 1.8, 24 posts), halves +4.3% / +1.05%.
  - Adverse moves within 24h: 90th percentile +16%, max +62%.
- **Verdict.** A weaker cousin of the delisting short (+3.6% per trade at 4h). About 12 posts a year. Sized to survive
  the squeezes it would add a few % a year. Not adopted; a forward paper log would be the next step.

**TradFi signal blend (`backend/tradfi_signal_lab.py`).** Hurst-Ooi-Pedersen's 1/3/12-month average vs the plan's
12-month rule on GLD/SLV/SPY/QQQ with 2026 funding.

| TradFi signal | Book Sharpe 2007-19 / 2020+ | Plan Sharpe in / out |
|---|---|---|
| 12m (current) | 0.65 / 0.77 | 1.67 / 1.74 |
| 1/3/12 monthly | 0.53 / 0.84 | 1.56 / 1.89 |
| 1/3/12 weekly | 0.47 / 0.77 | 1.57 / 1.77 |
| 3/6/12 monthly | 0.62 / 0.77 | 1.66 / 1.71 |

- The blend is worse before 2020 and in-sample, and better only after mid-2024. Fails the rule; 12m stays.

**Copying Hyperliquid vaults (`backend/hl_vault_lab.py`).** Every vault ever created (9,476; 6,372 closed, so no
survivorship bias), with each vault's on-chain equity and cumulative P&L history (`vaultDetails`). Rule: every 4
weeks, the top 10 by past 12-week return among vaults with ≥ $50k equity, held 4 weeks, minus the leader's 10% profit
share.
- **Data limit.** Hyperliquid keeps ~48 history points per vault, so older vaults have multi-week gaps and the clean
  test only covers 2025-05 .. 2026-09 (19 periods, ~65 eligible vaults).
- **No persistence.** Top 10: −50%/yr vs −2%/yr for the average vault. Past-vs-next rank correlation −0.11 (t −1.8);
  the second half was worse (−0.19, t −3.0). Last quarter's winners are mostly high-risk bets that then lose.
- Filling the gaps by interpolation (`--interp`, back to 2024-11) gives +0.05 (t 0.9). Interpolation biases towards
  persistence, so even that is weak.
- **HLP** (the exchange's own market-making and liquidation vault) was the steadiest: +16%/yr, weekly Sharpe ~1.5–2.3
  since 2024-11. It is a yield-like product with exchange and tail-event risk (e.g. the 2025 JELLY squeeze), not a
  skill signal.
- Rejected.

**Bottom line (2026-09-25).** Nothing tested beats the live 8-coin trend strategy out of sample. Its ceiling is roughly
2× a year at ~5% risk per trade with deep drawdowns; ~1% risk (+30%/yr, −26% drawdown) is the sane setting, 2–3% if
the user accepts deeper drawdowns for the deposit plan.

---

## 1. Executive Summary & Purpose

**LayaQuant Studio** is an autonomous, 100% open-source, zero-paid-API quantitative trading desk and real-time paper trading engine engineered for crypto perpetual futures. 

### Core Mission
Scale an initial bankroll of **€500.00 to €10,000.00** (+1,900% / 20x gain) without crossing the Kelly overbetting cliff, utilizing:
- **Dual-Model Consensus AI**: System 2 Macro Governor (Model A) + System 1 Microstructure Sniper (Model B) with institutional Veto Armor.
- **Dynamic Milestone Tiered Kelly Compounding**: Geometric position scaling modulated across 4 capitalization milestones with automatic defensive throttles.
- **Strict Asymmetric Risk Architecture**: 1.35x ATR Initial Stop Loss, 1.40x ATR Take-Profit 1 (75% scalp lock), and a 2.70x ATR trailing runner.
- **Zero-Paid-API Constraint**: Public Binance REST/WebSocket endpoints and open-source models/scrapers.

---

## 2. Directory Structure & File Inventory

```
c:/Users/mansour/Documents/antigravity/gallant-bell/
├── backend/
│   ├── server.py                        # FastAPI microservice (API routes, startup recovery, UI serving)
│   ├── paper_trader.py                  # Core trading engine (state machine, order lifecycle, blotter, scanner)
│   ├── dual_model_engine.py             # Model A (Macro Governor) + Model B (Micro Sniper) + Consensus Synthesizer
│   ├── neural_dream.py                  # 18-D PyTorch Neural Policy Net & offline dream learning loop
│   ├── news_engine.py                   # Zero-cost live RSS & Twitter sentiment scraper with NLP catalyst scoring
│   ├── research_engine.py               # Microstructure math: Lopez de Prado VPIN, Kyle lambda, behavioral metrics
│   ├── institutional_compounding_engine.py # Multi-asset compounding futures backtester & squeeze detector
│   ├── walk_forward_5m_futures.py       # Out-of-sample walk-forward validation framework
│   └── requirements.txt                 # Python dependencies
├── frontend/
│   └── index.html                       # Real-time Bloomberg/TradingView terminal UI (vanilla JS, Tailwind, Chart.js)
├── data/
│   ├── paper_trading_state.json         # Real-time persisted state (equity, positions, radar, indicators)
│   ├── paper_trades.csv                 # Detailed trade blotter log (fills, gross, fees, net PnL, RRR, reasons)
│   ├── live_sessions_history.json       # Historical session archive for post-trade equity curve telemetry
│   └── neural_market_model.pt           # PyTorch weights for the 18-D neural policy model
├── run.bat                              # Windows one-click launcher
├── run.sh                               # Unix launcher
└── CLAUDE_HANDOVER.md                   # This master developer document
```

---

## 3. High-Level Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data Feeds [Zero-Paid Public Feeds]
        B1["Binance Public REST Klines (5m, 1h, 4h)"]
        B2["Binance Public L2 Depth (Bid/Ask Ratio & Spread)"]
        B3["Binance Public WebSocket (@ticker stream)"]
        N1["RSS News & Whale Feeds (CoinDesk, Decrypt, Yahoo)"]
    end

    subgraph Intelligence Layer [backend/dual_model_engine.py]
        MA["Model A: Macro Governor (HTF Trend, News Catalyst, F&G)"]
        MB["Model B: Micro Sniper (OFI Depth, VPIN Toxicity, Squeeze)"]
        DCS["Dual Consensus Synthesizer (Veto Protocol & Score 0-100)"]
        MA --> DCS
        MB --> DCS
    end

    subgraph Execution & Risk [backend/paper_trader.py]
        SC["Market Universe Scanner (BTC, ETH, SOL, DOGE, XRP, BNB, AVAX, SUI)"]
        KC["Dynamic Tiered Kelly Sizer (Risk 2.5% - 8.0%, Buffer >20%)"]
        RM["Asymmetric Risk Manager (SL: 1.35x ATR | TP1: 1.4x 75% | Trail)"]
        POS["Multi-Slot Portfolio (Up to 3 Concurrent Trades, 5x Iso)"]
    end

    subgraph Persistence & Frontend
        STATE["data/paper_trading_state.json & paper_trades.csv"]
        UI["frontend/index.html (Real-Time Dashboard & Blotter)"]
    end

    B1 & B2 --> MB
    B1 & N1 --> MA
    DCS --> SC
    SC --> KC --> RM --> POS
    POS --> STATE
    B3 --> UI
    STATE --> UI
```

---

## 4. Detailed Component Breakdown

### A. The Dual-Model AI Engine (`backend/dual_model_engine.py`)

#### 1. Model A — Macro Governor (`LayaMacroGovernor`)
- **Higher Timeframe Regime Detection**: Classifies market into `TREND_BULL`, `TREND_BEAR`, `RANGE_BOUND`, or `TOXIC_CHOP` using 4-hour / 1-hour 50/200 EMA alignment and ATR expansion.
- **Directional Permit**: Emits `PERMIT_LONG`, `PERMIT_SHORT`, `PERMIT_BOTH`, or `ENFORCE_CASH`.
- **Governor Hard Veto**: Automatically blocks all trades if:
  - Macro regime is `TOXIC_CHOP` (high ATR expansion + flat moving average spread).
  - High-impact FUD catalyst is detected against the asset (e.g., hacks, regulatory actions).
  - Andrew Lo Adaptive Market Hypothesis (AMH) extreme sentiment throttle: caps leverage when Fear & Greed is extreme.

#### 2. Model B — Micro Sniper (`LayaMicroSniper`)
- **Microstructure Order Flow (OFI)**: Evaluates bid/ask order book depth ratio from public Binance order books:
  - If buying: requires Bid/Ask ratio >= 0.85 (OFI veto triggered if ask wall heavily outweighs bids).
  - If shorting: requires Bid/Ask ratio <= 1.15 (veto if heavy bid wall resting support).
- **VPIN Flow Toxicity (Lopez de Prado)**: Calculates Volume-Synchronized Probability of Toxicity. If VPIN >= 0.70, triggers toxicity veto to prevent adverse selection.
- **Bollinger-Keltner Volatility Squeeze**: Detects volatility compression when 20-period Bollinger Bands contract entirely inside 1.5x ATR Keltner Channels.

#### 3. Consensus Synthesizer (`DualConsensusSynthesizer`)
Combines Model A and Model B into a composite score:
`Score = 50.0 + (MacroConfidence * 25.0) + (TacticalConfidence * 25.0)`
- **Score >= 88.0/100 + Dual Consensus** is strictly required to open a trade.

---

### B. Core Portfolio & Execution Engine (`backend/paper_trader.py`)

#### 1. Multi-Asset Radar Universe
Scans 8 high-liquidity perpetual pairs on every cycle:
`["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "SUIUSDT"]`
Holds up to **3 concurrent positions** across decorrelated pairs.

#### 2. Dynamic Milestone Tiered Kelly Sizing
Positions are sized via a fraction of the Kelly Criterion based on win rate p ~= 0.65 and payoff ratio b >= 1.5:
`Kelly* = (p * (b + 1) - 1) / b`
Modulated across 4 Capital Milestones:
- **Phase 1 (€500 – €1,500)**: Scrappy Growth (Risk = 4.0% - 8.5% of equity).
- **Phase 2 (€1,500 – €3,500)**: Capital Expansion (Risk = 3.0% - 6.5%).
- **Phase 3 (€3,500 – €5,000)**: Milestone Capital Lock (Risk = 2.0% - 4.5%).
- **Phase 4 (€5,000 – €10,000)**: Strict Quarter-Kelly (Risk = 2.0% - 2.5%).
- **Defensive Throttle**: After >= 1 loss, risk is automatically cut by 40% until a win is banked.
- **Liquidity Buffer**: Always preserves a minimum **20% liquid unencumbered cash buffer** to eliminate liquidation risk.

#### 3. Calibrated Risk/Reward Rules
- **Stop Loss**: Strict 1.35x ATR_14 from entry.
- **Take-Profit 1 (Scalp Lock)**: Closes **75% of position** at +1.40x ATR_14 (~1:1 R:R).
- **Runner**: Remaining 25% runner has its stop ratcheted to `entry + 0.2x ATR` (guaranteed risk-free gain) and targets 2.70x ATR_14.
- **Stale Timeout**: 60-minute holding limit if a trade fails to hit TP1.
- **Anti-Churn Cooldown**: 10-minute (600s) mandatory desk pause after closing trades.
- **Maker Post-Only Execution**: Uses limit orders at 0.02% fee rate (saving 73% vs. 0.075% taker).
- **Strict Leverage Cap**: Hard-capped at **5.0x Isolated Margin** across all symbols.

---

## 5. Critical Debugging History & Lessons Learned

> **DO NOT RE-INTRODUCE THESE BUGS.** Claude Opus must be aware of the exact failure modes discovered and resolved:

### 1. The Cross-Symbol Price Contamination Bug (Fixed)
- **Symptom**: Historical log showed account exploding from €500 to $147,195.34 in fake profits.
- **Cause**: In `fetch_klines()`, a fallback line `p = self.current_price` was used during Binance API timeouts. When XRP timed out, it was assigned BNB's $792 price, creating a fake +$95k gain.
- **Fix**: Isolated symbol caches completely. Added the **Absolute Anomaly Guard**: any price tick deviating >25% from entry price is instantly dropped as a corrupted tick.

### 2. The Inverted Risk/Reward Bleed (-€56 with 68% Win Rate) (Fixed)
- **Symptom**: User ran live trading and lost €56 despite winning 51 out of 75 trades (68% win rate).
- **Cause**: Stop loss was 2.2x ATR (loss = -$5.30 on 100% position size) while TP1 closed only 50% at 1.2x ATR (win = +$1.38). Win/loss ratio was 0.26 (3.8x larger losses).
- **Fix**: Tightened SL to 1.35x ATR (cuts loss by 44%), increased TP1 to 75% at 1.4x ATR (lifts avg win to +$2.65). Expectancy flipped from -$0.76 to **+$0.86/trade**.

### 3. The 15-Minute Chokehold & Overnight Chop Bleed (Fixed)
- **Symptom**: Overnight session lost €94 across 165 trades, paying $40.60 in fees.
- **Cause**:
  1. A rigid 15-minute / 25-minute `adverse_atr >= 0.8` scratch dumped 42 trades at market inside normal sideways range noise.
  2. In `backend/paper_trader.py` (line 1438), `intelligent_dream_trainer` was silently overwriting `self.atr_multiplier_stop` back to 2.1x ATR every 10 seconds!
  3. In `RANGE_BOUND` regimes, `score >= 40.0` was permitted as a fallback, machine-gunning 76 entries into a flat 0% market.
- **Fix**:
  - Hard-locked `self.atr_multiplier_stop = 1.35` (banned neural overwrite).
  - Removed premature 25m -0.8x ATR scratches (giving trades 60m to develop).
  - Required `score >= 88.0` AND Volatility Squeeze for `RANGE_BOUND` entries.
  - Increased inter-trade cooldown to 10 minutes (600s).
  - Enforced post-only maker fills (0.02%).

### 4. The Trade Blotter 'Entry = Exit' UI Bug (Fixed)
- **Symptom**: All trade rows in the web UI displayed identical numbers for Entry and Exit.
- **Cause**: Frontend mapped `entryPrice: t.price || 0, exitPrice: t.price || 0`.
- **Fix**: Backend now emits distinct `entry_price` and `exit_price` on all trade objects; frontend displays `--` on open entries and exact execution prices on exits.

### 5. Paper-Simulator Optimism Bugs (Fixed 2026-09-25)
- Entry used the radar's display-rounded price (2dp): up to ±0.5% error on ~$1 coins (SUI/XRP) ≈ 1 ATR of fake P&L. Now uses `price_raw`/`atr_raw`.
- Entry fee was charged on margin, not notional (5× too small at 5× leverage).
- Stops/timeouts/forced/manual exits were billed as maker with zero slippage. Now taker (0.05%) + 2bps, filled at the stop or worse.
- Exits were triggered by the 5m bar's high/low, which includes prices from BEFORE entry and from earlier polls. Now only prices observed since the previous evaluation count.
- TP was checked before SL in the same window; now the stop wins. Trailing ratchet no longer raises the stop from a high and then "hits" it with an earlier low of the same window.
- Kelly sizing had hard floors (4–8.5%) that bet big even with negative Kelly; entry rows (`SELL_SHORT`) counted as losses. Now realized-leg stats, phase values are caps, 0.5% probe size without an edge.
- `evaluate_step` was not locked: the background loop and `/api/paper/step` could close the same position twice (double cash credit). Now serialized with `self.lock`.

---

## 6. How to Run, Test, and Verify

### Prerequisites
- Python 3.10+ (Current runtime: Python 3.11 with PyTorch + CUDA support).
- Windows OS (PowerShell / Command Prompt).

### Launching the System
```powershell
# From repository root:
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```
- Web Dashboard: `http://127.0.0.1:8000`
- API Health Check: `GET http://127.0.0.1:8000/health`
- Paper Trader Status: `GET http://127.0.0.1:8000/api/paper/status`

### Starting / Stopping Paper Trading
- **Via Dashboard**: Click `▶ RECORD LIVE` / `⏹ PAUSE`.
- **Via API**:
  ```powershell
  # Start live loop:
  Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/api/paper/start"
  
  # Stop live loop:
  Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/api/paper/stop"
  ```

### Quick Diagnostic Verification Command
```powershell
python -c "import urllib.request, json; res = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/paper/status').read()); print('Status:', res.get('is_running'), 'Equity:', res.get('total_equity'), 'Cash:', res.get('cash'), 'SL mult:', res.get('sl_atr_mult'), 'TP1 mult:', res.get('tp1_atr_mult'), 'Positions:', len(res.get('positions', [])))"
```

---

## 7. Current System State (As of Handover)

| Parameter | Current Value | Rationale |
| :--- | :--- | :--- |
| **Cash & Total Equity** | **€500.00** | Pristine clean baseline ready for fresh live run |
| **Open Positions** | **0** | Clean slate |
| **Stop Loss ATR Multiplier** | **1.35x** | Hard-locked (cuts losing size by 44%) |
| **Take Profit 1 Multiplier** | **1.40x** | Locks **75%** of position |
| **Runner Multiplier** | **2.70x** | 25% rides risk-free with stop at entry + 0.2x ATR |
| **Max Concurrent Positions**| **3** | Decorrelated multi-asset diversification |
| **Max Leverage** | **5.0x** | Hard ceiling on all assets |
| **Trade Cooldown** | **600 seconds (10 min)** | Anti-churn discipline |
| **Max Holding Timeout** | **3,600 seconds (60 min)**| Gives 5m breakout space without premature choking |
| **Fee Rate** | **0.02% (Maker)** | Post-only limit execution |
| **Scanner Universe** | **8 Top Altcoins** | BTC, ETH, SOL, DOGE, XRP, BNB, AVAX, SUI |

---

## 8. Immediate Next Steps & Strategic Roadmap for Claude Opus

1. **Live Autonomous Run**:
   - Trigger `POST /api/paper/start` or have the user click `▶ RECORD LIVE` to accumulate fresh, clean trade history under the hardened 1.35x ATR rules.
2. **Monitor Fee Structure**:
   - Inspect `data/paper_trades.csv` periodically to verify that Maker fee savings are active (`fee_rate = 0.0002`).
3. **Adaptive Range Governor Enhancement**:
   - If sideways crypto market conditions persist, consider implementing an **ADX (Average Directional Index)** or **Choppiness Index** filter to completely sleep the bot when ADX < 20.
4. **Milestone Progress Tracking**:
   - As equity approaches **€1,500** (Milestone 1), monitor the automatic transition from Phase 1 Scrappy Growth to Phase 2 Growth Expansion in `backend/paper_trader.py`.

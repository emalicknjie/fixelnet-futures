# Fixelnet

A personal BTC options paper trading system driven by a PyTorch convolutional neural network. The model watches daily OHLCV data, predicts market direction, selects the best 3-DTE Deribit option, sizes the position against a risk budget, and records the trade locally — all without touching a real broker. A self-improving agent reviews performance every morning and proposes code fixes via Telegram.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     run_trader.py  (9:00 AM UTC)                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
           ┌────────────────────┴─────────────────────┐
           ▼                                          ▼
┌──────────────────────┐              ┌───────────────────────────────┐
│   Signal Engine      │              │   Deribit Price Feed          │
│   fixelnet/          │              │   trading/deribit.py          │
│                      │              │                               │
│  Kraken REST API     │              │  Public REST — no auth        │
│  → OHLCV history     │              │  → BTC/USD index price        │
│  → StockStats fixels │              │  → Daily option chain         │
│    (RSI,CCI,WR,RSV,  │              │    (strikes, bid/ask USD)     │
│     ATR,VR × 15d)    │              │  → Per-instrument ticker      │
│  → Conv2d network    │              │    for mark-to-market         │
│  → 0 = LONG          │              └──────────────┬────────────────┘
│    1 = HEAVY_SHORT   │                             │
│    2 = SHORT         │                             │
└──────────┬───────────┘                             │
           │ signal + confidence                     │ chain + spot
           └──────────────────┬──────────────────────┘
                              ▼
           ┌──────────────────────────────────────────┐
           │   Strategy + Risk                        │
           │   trading/strategy.py                    │
           │   trading/risk.py                        │
           │                                          │
           │  Select nearest ATM strike               │
           │  Target 3 DTE  (sort by abs(DTE − 3))    │
           │  Mid-price from live bid/ask             │
           │  Confidence ≥ 65%                        │
           │  Max 3 open positions                    │
           │  Budget: $500 LONG/SHORT · $1500 HEAVY   │
           └──────────────────┬───────────────────────┘
                              │ OrderParams
                              ▼
           ┌──────────────────────────────────────────┐
           │   Paper Trade Simulator                  │
           │   trading/simulator.py                   │
           │                                          │
           │  record_trade()    → trades.json         │
           │  mark_to_market()    daily               │
           │  close_expired_positions()               │
           │  send_weekly_summary_if_due()            │
           └──────────────────┬───────────────────────┘
                              │ Telegram
                              ▼
           ┌──────────────────────────────────────────┐
           │   Notifications                          │
           │   trading/notify.py                      │
           │                                          │
           │  Trade entered / blocked / closed        │
           │  Daily MTM with unrealized P&L           │
           │  Weekly performance summary              │
           └──────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│             run_improvement_agent.sh  (9:30 AM UTC)                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│   Self-Improving Agent    trading/improvement_agent.py              │
│                                                                     │
│  1. Collect context — trades (7d), positions, stats, log tail,      │
│     source files (first 80 lines each), prior agent decisions       │
│  2. Call claude-sonnet-4-6 → 1–3 prioritized improvement proposals  │
│     each with: problem · fix · estimated impact · code patch        │
│  3. Deliver each proposal to Telegram with interactive buttons:     │
│           [ ✅ ACCEPT ]   [ ✏️ MODIFY ]   [ ❌ DECLINE ]           │
│  4. ACCEPT  → auto-apply find→replace patch to source file          │
│     MODIFY  → prompt for input, regenerate proposal, re-present     │
│     DECLINE → log and move on                                       │
│  5. All decisions written to data/improvement_agent_log.json        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Signal Engine (`fixelnet/`)

The core prediction pipeline. Fetches daily BTC/USD OHLCV from the Kraken public REST API, computes six technical indicators over a 15-day rolling window — ATR, CCI, RSI, RSV, VR, WR via `stockstats` — and runs them through a pretrained `Conv2d` network to predict one of three directional classes:

| Class | Signal | Action | Budget |
|---|---|---|---|
| 0 | **LONG** | Buy call | $500 |
| 1 | **HEAVY_SHORT** | Buy put, high conviction | $1,500 |
| 2 | **SHORT** | Buy put | $500 |

Input tensor shape: `(1, 1, 6, 15)` — one sample, one channel, six indicators, fifteen days. Hyperparameters are `exec()`'d from `neural_properties.txt` at load time; weights are loaded from a `.pth` state dict (not the pickled full model, so there's no class-name coupling at load time).

### Deribit Price Feed (`trading/deribit.py`)

Pulls live BTC option market data from Deribit's public REST API — no API key, no account required. Returns an option chain compatible with `strategy.signal_to_order_params()` with USD-denominated bid/ask prices. Deribit quotes premiums in BTC; this module multiplies by the live index price so callers always see dollars.

Key functions:

| Function | What it does |
|---|---|
| `get_btc_spot()` | Current BTC/USD index price |
| `get_option_chain(dte_min, dte_max)` | Full chain filtered to a DTE window |
| `get_quote(instrument_name)` | Single instrument bid/ask for MTM |

Deribit offers daily BTC expirations, which makes precise 3-DTE targeting reliable.

### Paper Trade Simulator (`trading/simulator.py`)

All trades are tracked locally in JSON — nothing is sent to a real broker.

```
data/
  trades.json                  full ledger (open + closed)
  positions.json               open positions with live MTM prices
  last_weekly.txt              date of last weekly summary
  improvement_agent_log.json   history of agent proposals and decisions
```

Settlement at expiry uses intrinsic value: `max(0, strike − BTC_spot)` for puts, `max(0, BTC_spot − strike)` for calls, using the Deribit index price at close.

### Self-Improving Agent (`trading/improvement_agent.py`)

The most interesting part of the project. Every morning at 9:30 AM the agent reads the current system state, calls the Anthropic API to get concrete improvement proposals, and delivers each one to Telegram with interactive buttons. Accepted proposals are auto-patched into the source files using exact string find→replace. All decisions are logged so the agent avoids proposing the same fix twice.

The agent uses `claude-sonnet-4-6` with a detailed system prompt that constrains proposal scope: no breaking changes, no changes to model architecture, no Deribit authentication, only safe auto-applicable patches.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/emalicknjie/fixelnet.git
cd fixelnet
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `EN_lib` is a private internal library not on PyPI. Install it manually from its source before running the Jupyter notebooks. It is not required for `run_trader.py`.

### 3. Model weights

Place the pretrained model directory at the path set by `MODEL_DIR`. The directory must contain:

```
neural_properties.txt   hyperparameter definitions (exec'd at load time)
test_weights.pth        PyTorch state dict
```

### 4. Environment variables

Copy the template below to `.env` in the project root and fill in your values:

```ini
# Required — path to pretrained model directory
MODEL_DIR=/path/to/nnet_state_saves/test

# Anthropic API — required for the improvement agent
ANTHROPIC_API_KEY=sk-ant-...

# Telegram — optional; all notifications are silently skipped if absent
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Tastytrade — only needed if re-enabling live broker execution
TASTYTRADE_USERNAME=
TASTYTRADE_PASSWORD=
TASTYTRADE_CHALLENGE_ANSWER=
TASTYTRADE_OTP=
ACCOUNT_NUMBER=
```

The Deribit price feed requires no credentials.

---

## Running

### Daily trader

```bash
# Dry run — shows signal, selected strike, risk verdict; records nothing
python3 run_trader.py --dry-run

# Live run — records trade to data/trades.json
python3 run_trader.py
```

Example dry-run output:

```
══════════════════════════════════════════════════════════
  BTC Signal: SHORT (BUY PUT)
  Scores  →  Long 0.121  Heavy Short 0.203  Short 0.676
  BTC spot    $83,412.00
══════════════════════════════════════════════════════════

  Symbol      : BTC-19APR26-84000-P
  Strike      : $     84,000  (0.7% OTM)
  Expiry      : 2026-04-19  (3 DTE)
  Limit price : $    259.69
  Quantity    : 1 contract(s)
  Total debit : $    259.69
  Budget      : $259.69 / $500  [PASS]
  Verdict     : APPROVED

[DRY RUN] Trade NOT recorded.
```

### Performance dashboard

```bash
python3 show_performance.py
```

Prints a box-drawn table of open positions, closed trades, and aggregate P&L stats directly in the terminal.

### Improvement agent (manual run)

```bash
python3 -c "from trading.improvement_agent import run; run()"
```

---

## Cron schedule

Both scripts use shell wrappers that append timestamped output to daily log files under `logs/`. The improvement agent wrapper fires a Telegram crash alert if the process exits non-zero.

```cron
# Daily trader — 9:00 AM UTC
0  9 * * * /root/fixelnet/run_trader.sh

# Improvement agent — 9:30 AM UTC (30 min after main trader)
30 9 * * * /root/fixelnet/run_improvement_agent.sh
```

Logs land at:

```
logs/trader_YYYY-MM-DD.log
logs/improvement_agent_YYYY-MM-DD.log
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Signal model | PyTorch `Conv2d`, `stockstats`, scikit-learn `MinMaxScaler` |
| Historical data | Kraken public REST API (daily OHLCV) |
| Options data | Deribit public REST API (chain, quotes, index price) |
| AI agent | Anthropic API — `claude-sonnet-4-6` |
| Notifications | Telegram Bot API — messages + inline keyboard callbacks |
| Persistence | JSON flat files |
| Scheduling | cron + bash wrappers |
| Language | Python 3.10+, bash |

---

## Project layout

```
fixelnet/
├── fixelnet/                   signal engine
│   ├── data.py                 Kraken OHLCV fetch
│   ├── features.py             fixel computation (stockstats indicators)
│   ├── models.py               FixelNet Conv2d architecture + loader
│   └── signals.py              run_predictions(), plot_signals(), export_table()
├── trading/                    execution layer
│   ├── deribit.py              Deribit public REST client
│   ├── simulator.py            paper trade recorder, MTM, settlement
│   ├── strategy.py             signal → OrderParams (strike selection, pricing)
│   ├── risk.py                 confidence, position cap, budget checks
│   ├── notify.py               Telegram notification helpers
│   ├── improvement_agent.py    daily Claude-powered self-improvement loop
│   └── execution.py            Tastytrade session + DXLink (inactive / reference)
├── data/                       runtime JSON (not committed)
├── logs/                       daily cron log files
├── nnet_state_saves/           model weights (not committed)
├── run_trader.py               main entry point
├── run_trader.sh               cron wrapper
├── run_improvement_agent.sh    cron wrapper with Telegram crash alert
├── show_performance.py         terminal P&L dashboard
└── .env                        secrets (not committed)
```

---

## Disclaimer

Personal research project. All trades are simulated — no real money is at risk. Nothing here is financial advice.

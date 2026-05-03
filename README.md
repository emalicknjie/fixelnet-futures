# Fixelnet Futures

A BTC paper futures trading system driven by the Fixelnet neural network. The same Conv2d signal engine that powers the [fixelnet](https://github.com/emalicknjie/fixelnet) options trader is wired here to unleveraged spot futures positions — long or short, sized by conviction, auto-closed after 72 hours or on a 2% stop-loss. Everything runs locally with JSON persistence and Telegram notifications.

---

## How it differs from fixelnet (options)

| | fixelnet (options) | fixelnet-futures (this repo) |
|---|---|---|
| Instrument | Deribit BTC options (3 DTE) | Simulated BTC spot futures |
| LONG signal | Buy call, $1,500 budget | Enter long, $500 notional |
| SELL signal | Buy put, $1,500 budget | Enter short, $1,000 notional (2×) |
| HOLD signal | No trade | Close any open position |
| Price feed | Deribit public REST (options chain + index) | Kraken public REST (spot ticker) |
| Position close | Option expires at intrinsic value | 72h window or 2% stop-loss |
| Broker layer | None (paper simulator) | None (paper simulator) |
| Signal engine | `fixelnet/` — unchanged | `fixelnet/` — unchanged |

The fixelnet signal engine (`fixelnet/`) is byte-for-byte identical between the two repos. Only the execution layer differs.

---

## Signal mapping

The neural network outputs three classes. Their original notebook intent is preserved here:

| Class | Label | Notebook name | Action | Notional |
|---|---|---|---|---|
| 0 | `LONG` | BUY | Enter long | $500 |
| 1 | `HEAVY_SHORT` | SELL | Enter short | $1,000 (2×) |
| 2 | `HOLD` | HOLD | Close open position, no new entry | — |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  run_futures_trader.py  (9:00 AM UTC)               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
┌───────────────────────┐         ┌─────────────────────────────────┐
│   Signal Engine       │         │   Kraken Spot Price             │
│   fixelnet/           │         │   (public ticker, no auth)      │
│                       │         │                                 │
│  Kraken REST → OHLCV  │         │  Last trade price: XBTUSD       │
│  → StockStats fixels  │         │  Fallback: OHLCV last close     │
│    RSI, CCI, WR, RSV, │         └──────────────┬──────────────────┘
│    ATR, VR × 15 days  │                        │ btc_spot
│  → Conv2d network     │                        │
│  → class 0/1/2        │                        │
└──────────┬────────────┘                        │
           │ signal + confidence scores           │
           └──────────────────┬───────────────────┘
                              ▼
           ┌──────────────────────────────────────────┐
           │   Strategy + Risk                        │
           │   trading/futures_strategy.py            │
           │   trading/futures_risk.py                │
           │                                          │
           │  LONG  → long,  1× notional ($500)       │
           │  SELL  → short, 2× notional ($1,000)     │
           │  HOLD  → close open position             │
           │                                          │
           │  Confidence ≥ 65%                        │
           │  Max 1 open position                     │
           │  Stop-loss: ±2% from entry               │
           └──────────────────┬───────────────────────┘
                              │ FuturesOrderParams
                              ▼
           ┌──────────────────────────────────────────┐
           │   Paper Futures Simulator                │
           │   trading/futures_simulator.py           │
           │                                          │
           │  record_futures_trade()                  │
           │   → data/futures_trades.json             │
           │                                          │
           │  mark_to_market_futures()                │
           │   → update current_price + unrealized    │
           │   → auto-close on stop-loss              │
           │   → auto-close after 72h window          │
           │                                          │
           │  close_open_on_hold()                    │
           │   → settle at market on HOLD signal      │
           │                                          │
           │  get_futures_summary()                   │
           │   → P&L, win rate, best/worst trade      │
           └──────────────────┬───────────────────────┘
                              │ Telegram
                              ▼
           ┌──────────────────────────────────────────┐
           │   Notifications                          │
           │   trading/futures_notify.py              │
           │                                          │
           │  Trade entered (direction, size, stop)   │
           │  Trade closed (reason, P&L, result)      │
           │  Daily MTM with unrealized P&L           │
           │  HOLD signal close                       │
           │  Weekly performance summary              │
           └──────────────────────────────────────────┘
```

---

## Position lifecycle

```
Entry
 └─ record_futures_trade()
     ├─ entry_price   = Kraken spot at signal time
     ├─ size_btc      = notional / entry_price
     ├─ stop_loss     = entry × (1 − 0.02)  for long
     │                = entry × (1 + 0.02)  for short
     └─ close_after   = entry_time + 72 hours

Daily MTM  (mark_to_market_futures)
 ├─ Fetch fresh Kraken spot
 ├─ Update current_price + unrealized_pnl
 ├─ Check stop-loss → auto-close if triggered
 └─ Check close_after → auto-close if 72h elapsed

Signals
 ├─ HOLD signal → close_open_on_hold() → settle at market
 └─ LONG/SELL   → check risk → record new entry (if approved)

Close
 └─ close_futures_trade(reason)
     ├─ reason = "stop_loss"   | −2% hit
     ├─ reason = "72h_expiry"  | prediction window elapsed
     ├─ reason = "hold_signal" | model emitted HOLD
     └─ P&L = (close − entry) × size_btc   (long)
            = (entry − close) × size_btc   (short)
```

---

## Risk parameters

| Parameter | Value | Notes |
|---|---|---|
| Base notional | $500 | LONG trades |
| HEAVY_SHORT notional | $1,000 | 2× base; higher conviction short |
| Max open positions | 1 | Only one futures position at a time |
| Stop-loss | 2% | Auto-triggered on MTM check |
| Prediction window | 72 hours | Matches Fixelnet's 3-day horizon |
| Min confidence | 65% | Softmax score for predicted class |
| Leverage | 1× | Unleveraged; notional = cash at risk |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/emalicknjie/fixelnet-futures.git
cd fixelnet-futures
```

### 2. Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Model weights

The model directory must contain:

```
neural_properties.txt   hyperparameter definitions (exec'd at load time)
test_weights.pth        PyTorch state dict
```

Point `MODEL_DIR` at it in `.env`.

### 4. Environment

```ini
# .env
MODEL_DIR=/path/to/nnet_state_saves/test

# Telegram — optional; all notifications are silently skipped if absent
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

No broker credentials. No API keys. Kraken and Deribit data is all public.

---

## Running

```bash
# Dry run — shows today's signal, proposed order, risk verdict; records nothing
python3 run_futures_trader.py --dry-run

# Live paper trade
python3 run_futures_trader.py
```

Example dry-run output:

```
══════════════════════════════════════════════════════════════
  Fixelnet Futures Paper Trader [DRY RUN]
  2026-05-03 09:00:41
══════════════════════════════════════════════════════════════

[1/3] Running Fixelnet on BTC ...

  ┌────────────────────────────────────────────────────────┐
  │              FIXELNET SIGNAL — BTC/USD                 │
  ├────────────────────────────────────────────────────────┤
  │  Signal      : SELL (SHORT 2×)                         │
  │  Confidence  : 74.23%                                  │
  │  BTC spot    :      $95,142.00                         │
  ├────────────────────────────────────────────────────────┤
  │  Long score        : 0.1204                            │
  │  Heavy short score : 0.7423                            │
  │  Short/hold score  : 0.1373                            │
  └────────────────────────────────────────────────────────┘

[2/3] Fetching BTC spot price and running daily maintenance ...
  BTC/USD spot : $95,142.00
  No open positions after MTM.

[3/3] Evaluating order ...

  ┌────────────────────────────────────────────────────────┐
  │                  PROPOSED FUTURES ORDER                │
  ├────────────────────────────────────────────────────────┤
  │  Direction   : SHORT (sell)                            │
  │  Entry price :       $95,142.00                        │
  │  Notional    :       $1,000.00  (2× base)              │
  │  Size        : 0.010511 BTC                            │
  │  Stop loss   :       $97,044.84  (+2%)                 │
  │  Auto-close  : 72h  (Fixelnet window)                  │
  ├────────────────────────────────────────────────────────┤
  │                     RISK CHECK                         │
  ├────────────────────────────────────────────────────────┤
  │    Confidence          : 74.23%               [PASS]   │
  │    Open positions      : 0/1                  [PASS]   │
  │    Spot price          : $95,142.00           [PASS]   │
  ├────────────────────────────────────────────────────────┤
  │    Verdict       : APPROVED                            │
  └────────────────────────────────────────────────────────┘

  [DRY RUN] Trade NOT recorded (dry-run mode).
══════════════════════════════════════════════════════════════
  Done.
══════════════════════════════════════════════════════════════
```

---

## Cron

```cron
# Daily futures trader — 9:00 AM UTC
0 9 * * * cd /root/fixelnet-futures && python3 run_futures_trader.py >> logs/futures_$(date +\%F).log 2>&1
```

---

## Project layout

```
fixelnet-futures/
├── fixelnet/                       signal engine (unchanged from main repo)
│   ├── data.py                     Kraken OHLCV fetch + price window
│   ├── features.py                 fixel computation (6 indicators × 15 days)
│   ├── models.py                   FixelNet Conv2d + loader
│   └── signals.py                  LONG=0, HEAVY_SHORT=1, SHORT/HOLD=2
├── trading/
│   ├── futures_strategy.py         signal → FuturesOrderParams (direction, size_multiplier)
│   ├── futures_risk.py             notional sizing, stop-loss price, pre-trade checks
│   ├── futures_simulator.py        JSON persistence, MTM, stop-loss, 72h close
│   └── futures_notify.py           Telegram notifications for all trade events
├── data/
│   └── futures_trades.json         trade ledger (open + closed)
├── run_futures_trader.py           main entry point
├── requirements.txt
└── .env                            secrets (not committed)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Signal model | PyTorch `Conv2d`, `stockstats`, scikit-learn `MinMaxScaler` |
| Historical data | Kraken public REST API (daily OHLCV) |
| Live spot price | Kraken public ticker (`/0/public/Ticker?pair=XBTUSD`) |
| Notifications | Telegram Bot API |
| Persistence | JSON flat file (`data/futures_trades.json`) |
| Scheduling | cron |
| Language | Python 3.10+ |

---

## Related

- [emalicknjie/fixelnet](https://github.com/emalicknjie/fixelnet) — the original repo; same signal engine wired to Deribit BTC options

---

## Disclaimer

Personal research project. All trades are simulated — no real money is at risk. Nothing here is financial advice.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fixelnet Futures** is a BTC paper futures trading system driven by the Fixelnet neural network signal engine.
It fetches daily OHLCV data from the **Kraken public REST API**, runs inference through a pre-trained PyTorch Conv2d network, and maps signals to unleveraged BTC spot futures paper trades.

**Signal classes (original notebook intent):**
- Class 0 (LONG/BUY)          → enter long, $500 notional
- Class 1 (HEAVY_SHORT/SELL)  → enter short, $1000 notional (2×)
- Class 2 (HOLD)              → close any open position, no new trade

## Running

```bash
pip install -r requirements.txt

# Dry run — show today's signal and proposed order without recording
python3 run_futures_trader.py --dry-run

# Live paper trade
python3 run_futures_trader.py
```

Set up `.env` with `MODEL_DIR`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Architecture

### Futures Pipeline

```
Kraken public API (daily OHLCV) ← fixelnet/data.py
    → fetch_price_history()
    → get_price_window()
    ↓
fixelnet/features.py → get_latest_fixel()
    ↓
fixelnet/models.py → predict()
    → (signal_class, long_score, heavy_short_score, short_score)
    ↓
trading/futures_strategy.py
    → signal_to_futures_order()   # LONG→long 1×, HEAVY_SHORT→short 2×, HOLD→None
    ↓
trading/futures_risk.py
    → check_futures_risk()        # confidence ≥ 65%, ≤ 1 open position
    → compute_stop_loss()         # ±2% from entry
    ↓
trading/futures_simulator.py
    → record_futures_trade()      # write to data/futures_trades.json
    → mark_to_market_futures()    # Kraken spot, auto stop-loss & 72h close
    → get_futures_summary()       # P&L stats
    ↓
trading/futures_notify.py         # Telegram notifications for all events
```

### Package Layout

| Module | Responsibility |
|---|---|
| `fixelnet/` | Unchanged signal engine (Kraken OHLCV → FixelNet → class 0/1/2) |
| `trading/futures_strategy.py` | Signal → FuturesOrderParams (direction, size_multiplier) |
| `trading/futures_risk.py` | Position sizing, stop-loss price, pre-trade checks |
| `trading/futures_simulator.py` | JSON persistence, MTM, stop-loss, 72h auto-close |
| `trading/futures_notify.py` | Telegram notifications for all trade events |
| `run_futures_trader.py` | Main cron runner |

### Trade Lifecycle

1. **Entry** — `record_futures_trade()` writes to `data/futures_trades.json`
2. **Daily MTM** — `mark_to_market_futures()` fetches Kraken spot, updates unrealized P&L,
   auto-closes on stop-loss or 72h expiry
3. **HOLD close** — `close_open_on_hold()` closes position at market if model emits class 2
4. **Weekly summary** — `send_futures_weekly_if_due()` fires every Monday

### Risk Parameters

| Parameter | Value |
|---|---|
| Base notional | $500 (LONG), $1000 (HEAVY_SHORT 2×) |
| Max open positions | 1 |
| Stop loss | 2% adverse move |
| Auto-close | 72 hours (Fixelnet prediction window) |
| Min confidence | 65% |

### Neural Network (unchanged)

- Input tensor: `(1, 1, 6, 15)` — batch × channel × indicators × days
- Prediction classes: `0=LONG`, `1=HEAVY_SHORT/SELL`, `2=HOLD`
- Model dir: set via `MODEL_DIR` in `.env`

### Paths

Model dir: `MODEL_DIR` env var (default: `/root/fixelnet-futures/nnet_state_saves/test/`)
Trade data: `data/futures_trades.json`

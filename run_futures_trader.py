#!/usr/bin/env python3
"""
run_futures_trader.py — Fixelnet → BTC futures paper trader.

Workflow
--------
1. Load the pre-trained FixelNet model and compute today's BTC signal.
2. Fetch current BTC/USD spot price from Kraken.
3. Run daily maintenance (MTM, stop-loss check, 72h expiry closes).
4. Apply HOLD logic: if signal is class 2, close any open position, no new entry.
5. For LONG (class 0) or SELL/SHORT (class 1), apply risk rules and record trade.

Usage
-----
    # Dry run — show signal and proposed order without recording:
    python3 run_futures_trader.py --dry-run

    # Live paper trade:
    python3 run_futures_trader.py

Environment variables (set in .env)
-------------------------------------
    MODEL_DIR          Path to the FixelNet model directory
    TELEGRAM_BOT_TOKEN Telegram bot token (optional)
    TELEGRAM_CHAT_ID   Telegram chat/user ID (optional)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Fixelnet signal engine ────────────────────────────────────────────────────
from fixelnet import load_model, predict
from fixelnet.data import fetch_price_history, get_price_window
from fixelnet.features import get_latest_fixel
from fixelnet.signals import LONG, HEAVY_SHORT, SHORT as HOLD

# ── Futures trading modules ───────────────────────────────────────────────────
from trading.futures_strategy import (
    SIGNAL_NAMES,
    signal_to_futures_order,
)
from trading.futures_risk import (
    MAX_OPEN_POSITIONS,
    MIN_CONFIDENCE,
    STOP_LOSS_PCT,
    check_futures_risk,
    notional_for_signal,
)
from trading.futures_simulator import (
    _btc_spot,
    close_open_on_hold,
    get_futures_summary,
    load_open_futures_trades,
    mark_to_market_futures,
    record_futures_trade,
    send_futures_weekly_if_due,
)
from trading.futures_notify import (
    notify_error,
    notify_futures_hold_close,
    notify_futures_no_trade,
)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR:    str = os.environ.get("MODEL_DIR", "/root/fixelnet-futures/nnet_state_saves/test")
BTC_ASSET:    str = "BTCUSDT"
_HISTORY_START    = "1 Jan, 2025"


# ── Argument parsing ───────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixelnet BTC futures paper trader")
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show signal and proposed order without recording any trade.",
    )
    return p.parse_args()


# ── Step 1: Fixelnet signal ────────────────────────────────────────────────────
def _get_btc_signal() -> tuple[int, float, float, float]:
    """Load model, fetch BTC history, run inference.

    Returns (signal_class, long_score, heavy_short_score, short_score).
    """
    print(f"  Model dir : {MODEL_DIR}")
    model = load_model(MODEL_DIR)

    today_str = datetime.today().strftime("%d %b, %Y")
    print(f"  Fetching BTC/USD price history from Kraken (since {_HISTORY_START}) ...")
    price_history = fetch_price_history(BTC_ASSET, _HISTORY_START, today_str)

    if price_history.empty:
        raise RuntimeError("Kraken returned empty price history for BTCUSDT.")

    today  = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    window, _, _ = get_price_window(price_history, today, interval_days=5)

    fixel = get_latest_fixel(window)
    signal, long_s, heavy_s, short_s = predict(model, fixel)
    return signal, long_s, heavy_s, short_s


def _print_signal(signal: int, long_s: float, heavy_s: float, short_s: float, btc_spot: float) -> None:
    confidence_map = {LONG: long_s, HEAVY_SHORT: heavy_s, HOLD: short_s}
    confidence = confidence_map[signal]
    width = 56

    print()
    print("  ┌" + "─" * width + "┐")
    print(f"  │{'FIXELNET SIGNAL — BTC/USD':^{width}}│")
    print("  ├" + "─" * width + "┤")
    print(f"  │  Signal      : {SIGNAL_NAMES[signal]:<{width - 17}}│")
    print(f"  │  Confidence  : {f'{confidence:.2%}':<{width - 17}}│")
    print(f"  │  BTC spot    : ${btc_spot:>12,.2f}{'':>{width - 31}}│")
    print("  ├" + "─" * width + "┤")
    print(f"  │  Long score        : {long_s:.4f}{'':>{width - 28}}│")
    print(f"  │  Heavy short score : {heavy_s:.4f}{'':>{width - 28}}│")
    print(f"  │  Short/hold score  : {short_s:.4f}{'':>{width - 28}}│")
    print("  └" + "─" * width + "┘")
    print()


# ── Step 3: Evaluate and maybe place ──────────────────────────────────────────
def _evaluate_and_maybe_place(
    signal:   int,
    long_s:   float,
    heavy_s:  float,
    short_s:  float,
    btc_spot: float,
    dry_run:  bool,
) -> None:
    confidence_map = {LONG: long_s, HEAVY_SHORT: heavy_s, HOLD: short_s}
    confidence = confidence_map[signal]
    signal_name = SIGNAL_NAMES[signal]
    width = 56

    # ── HOLD — close open position, no new entry ──────────────────────────────
    if signal == HOLD:
        print("  Signal is HOLD — closing any open position, no new entry.")
        if not dry_run:
            closed = close_open_on_hold(btc_spot)
            closed_ids = [t["id"] for t in closed]
            notify_futures_hold_close(signal_name, confidence, btc_spot, closed_ids)
            if closed:
                for t in closed:
                    sign = "+" if (t.get("pnl") or 0) >= 0 else ""
                    print(f"  Closed {t['id']}  {t['direction']}  "
                          f"P&L {sign}${t.get('pnl', 0):,.2f}")
            else:
                print("  No open positions to close.")
        else:
            open_t = load_open_futures_trades()
            print(f"  [DRY RUN] Would close {len(open_t)} open position(s).")
        return

    # ── Build order params ────────────────────────────────────────────────────
    order = signal_to_futures_order(signal, confidence)
    if order is None:
        return   # shouldn't happen for LONG/HEAVY_SHORT

    notional = notional_for_signal(signal)

    # ── Risk check ────────────────────────────────────────────────────────────
    open_trades = load_open_futures_trades()
    risk = check_futures_risk(signal, confidence, open_trades, btc_spot)

    direction_label = "LONG  (buy)" if order.direction == "long" else "SHORT (sell)"
    size_btc   = round(notional / btc_spot, 8) if btc_spot > 0 else 0.0
    stop_price = round(
        btc_spot * (1 - STOP_LOSS_PCT) if order.direction == "long"
        else btc_spot * (1 + STOP_LOSS_PCT),
        2,
    )
    close_after_h = 72

    print()
    print("  ┌" + "─" * width + "┐")
    print(f"  │{'PROPOSED FUTURES ORDER':^{width}}│")
    print("  ├" + "─" * width + "┤")
    print(f"  │  Direction   : {direction_label:<{width - 17}}│")
    print(f"  │  Entry price : ${btc_spot:>12,.2f}{'':>{width - 31}}│")
    print(f"  │  Notional    : ${notional:>10,.2f}  ({order.size_multiplier:.0f}× base){'':>{width - 37}}│")
    print(f"  │  Size        : {size_btc:.6f} BTC{'':>{width - 26}}│")
    print(f"  │  Stop loss   : ${stop_price:>12,.2f}  (−2%){'':>{width - 35}}│")
    print(f"  │  Auto-close  : {close_after_h}h  (Fixelnet window){'':>{width - 35}}│")
    print("  ├" + "─" * width + "┤")
    print(f"  │{'RISK CHECK':^{width}}│")
    print("  ├" + "─" * width + "┤")
    checks = [
        ("Confidence",     f"{confidence:.2%}",                confidence >= MIN_CONFIDENCE),
        ("Open positions", f"{len(open_trades)}/{MAX_OPEN_POSITIONS}", len(open_trades) < MAX_OPEN_POSITIONS),
        ("Spot price",     f"${btc_spot:,.2f}",                btc_spot > 0),
    ]
    for label, value, passed in checks:
        status = "PASS" if passed else "FAIL"
        line = f"  {label:<18}: {value}"
        print(f"  │  {line:<{width - 5}}[{status}]  │")

    verdict = "APPROVED" if risk.approved else f"BLOCKED — {risk.reason}"
    print("  ├" + "─" * width + "┤")
    print(f"  │  {'Verdict':<14}: {verdict[:width - 19]:<{width - 19}}│")
    print("  └" + "─" * width + "┘")
    print()

    if not risk.approved:
        notify_futures_no_trade(signal_name, confidence, btc_spot, risk.reason)
        return

    if dry_run:
        print("  [DRY RUN] Trade NOT recorded (dry-run mode).")
        return

    # ── Record trade ──────────────────────────────────────────────────────────
    print("  Recording simulated futures trade ...")
    trade_id = record_futures_trade(
        order=order,
        signal_name=signal_name,
        entry_price=btc_spot,
        notional_usd=risk.notional_usd,
        size_btc=risk.size_btc,
        confidence=confidence,
    )
    print(f"  Trade recorded.  ID={trade_id}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    args    = _parse_args()
    dry_run = args.dry_run

    tag = " [DRY RUN]" if dry_run else ""
    print("=" * 62)
    print(f"  Fixelnet Futures Paper Trader{tag}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # ── Step 1: Signal ────────────────────────────────────────────────────────
    print("\n[1/3] Running Fixelnet on BTC ...")
    try:
        signal, long_s, heavy_s, short_s = _get_btc_signal()
    except Exception as exc:
        print(f"\n[ERROR] Fixelnet failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        notify_error("Fixelnet inference", exc)
        return 1

    # ── Step 2: BTC spot + daily maintenance ─────────────────────────────────
    print("\n[2/3] Fetching BTC spot price and running daily maintenance ...")
    btc_spot = _btc_spot()
    if btc_spot is None:
        print("[ERROR] Could not fetch BTC spot price.", file=sys.stderr)
        notify_error("BTC spot fetch", RuntimeError("Kraken spot unavailable"))
        return 1
    print(f"  BTC/USD spot : ${btc_spot:,.2f}")

    _print_signal(signal, long_s, heavy_s, short_s, btc_spot)

    if not dry_run:
        print("  Running mark-to-market (stop-loss + 72h expiry check) ...")
        still_open = mark_to_market_futures()
        if still_open:
            for t in still_open:
                sign = "+" if t["unrealized_pnl"] >= 0 else ""
                print(f"  {t['id']}  {t['direction']}"
                      f"  @ ${t['current_price']:,.2f}"
                      f"  unrealized {sign}${t['unrealized_pnl']:,.2f}")
        else:
            print("  No open positions after MTM.")

        send_futures_weekly_if_due()

    # ── Step 3: Risk + order ──────────────────────────────────────────────────
    print("\n[3/3] Evaluating order ...")
    try:
        _evaluate_and_maybe_place(
            signal, long_s, heavy_s, short_s, btc_spot,
            dry_run=dry_run,
        )
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        notify_error("Order evaluation/placement", exc)
        return 1

    print("=" * 62)
    print("  Done.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())

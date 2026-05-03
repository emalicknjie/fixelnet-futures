"""
trading/futures_simulator.py — Local paper futures trading simulator.

All trade data is persisted to data/futures_trades.json.

Trade lifecycle
---------------
record_futures_trade()       on signal day        → entry written to futures_trades.json
mark_to_market_futures()     daily (live)         → current price updated, stop-loss / 72h
                                                     expiry auto-closed, Telegram notified
close_futures_trade()        stop-loss / 72h /    → trade settled, P&L finalised
                             hold-signal close
send_futures_weekly_if_due() Mondays              → aggregate stats sent to Telegram

Stop-loss rule
--------------
Long  position stops out when current_price ≤ stop_loss_price  (−2% from entry)
Short position stops out when current_price ≥ stop_loss_price  (+2% from entry)

72-hour expiry
--------------
Fixelnet's prediction horizon is 3 days.  Any open position is force-closed at
market price once close_after (entry_time + 72h) has elapsed.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── File paths ─────────────────────────────────────────────────────────────────
_DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
_TRADES_FILE = _DATA_DIR / "futures_trades.json"
_WEEKLY_FILE = _DATA_DIR / "futures_last_weekly.txt"

_CLOSE_AFTER_HOURS = 72   # Fixelnet prediction window


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _ensure() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_futures_trades() -> list[dict[str, Any]]:
    """Return all futures trade records (open and closed)."""
    _ensure()
    if not _TRADES_FILE.exists():
        return []
    return json.loads(_TRADES_FILE.read_text())


def _save_futures_trades(trades: list[dict]) -> None:
    _ensure()
    _TRADES_FILE.write_text(json.dumps(trades, indent=2, default=str))


def load_open_futures_trades() -> list[dict[str, Any]]:
    """Return only open futures trade records."""
    return [t for t in load_futures_trades() if t.get("status") == "open"]


# ── BTC spot price ─────────────────────────────────────────────────────────────

def _btc_spot() -> float | None:
    """
    Current BTC/USD price.

    Priority:
    1. Kraken public ticker (real-time)
    2. Kraken OHLCV last close (fallback via fixelnet.data)
    """
    import requests as _req

    try:
        resp = _req.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": "XBTUSD"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("error"):
            result = data.get("result", {})
            pair_data = (
                result.get("XXBTZUSD")
                or result.get("XBTUSD")
                or (next(iter(result.values()), None) if result else None)
            )
            if pair_data:
                price = float(pair_data["c"][0])  # last trade price
                log.debug("Kraken ticker BTC/USD = %.2f", price)
                return price
    except Exception as exc:
        log.warning("Kraken ticker failed: %s — trying OHLCV fallback", exc)

    try:
        from fixelnet.data import fetch_price_history
        from datetime import datetime as _dt
        today = _dt.today().strftime("%d %b, %Y")
        hist  = fetch_price_history("BTCUSDT", "1 Jan, 2025", today)
        price = float(hist["close"].iloc[-1])
        log.debug("Kraken OHLCV fallback BTC/USD = %.2f", price)
        return price
    except Exception as exc:
        log.error("Both BTC spot sources failed: %s", exc)
        return None


# ── P&L helpers ────────────────────────────────────────────────────────────────

def _calc_pnl(direction: str, entry_price: float, close_price: float, size_btc: float) -> tuple[float, float]:
    """Return (pnl_usd, pnl_pct) for a closed futures position."""
    if direction == "long":
        pnl_usd = (close_price - entry_price) * size_btc
        pnl_pct = (close_price - entry_price) / entry_price
    else:  # short
        pnl_usd = (entry_price - close_price) * size_btc
        pnl_pct = (entry_price - close_price) / entry_price
    return round(pnl_usd, 2), round(pnl_pct, 6)


def _stop_triggered(direction: str, current_price: float, stop_loss_price: float) -> bool:
    if direction == "long":
        return current_price <= stop_loss_price
    return current_price >= stop_loss_price


# ── Public API ─────────────────────────────────────────────────────────────────

def record_futures_trade(
    order: Any,          # trading.futures_strategy.FuturesOrderParams
    signal_name: str,
    entry_price: float,
    notional_usd: float,
    size_btc: float,
    confidence: float,
) -> str:
    """
    Record a simulated futures trade entry.

    Appends a new record to futures_trades.json and fires a Telegram notification.

    Parameters
    ----------
    order        : FuturesOrderParams from signal_to_futures_order().
    signal_name  : Human-readable signal label (e.g. "LONG (BUY)").
    entry_price  : BTC/USD spot at entry.
    notional_usd : USD notional size.
    size_btc     : BTC quantity (notional / entry_price).
    confidence   : Softmax probability for predicted class.

    Returns
    -------
    8-character hex trade ID.
    """
    from trading.futures_risk import compute_stop_loss

    trade_id   = uuid.uuid4().hex[:8]
    now        = datetime.now()
    close_after = (now + timedelta(hours=_CLOSE_AFTER_HOURS)).isoformat(timespec="seconds")
    stop_price  = compute_stop_loss(order.direction, entry_price)

    trade: dict[str, Any] = {
        "id":             trade_id,
        "entry_date":     date.today().isoformat(),
        "entry_time":     now.isoformat(timespec="seconds"),
        "signal":         signal_name,
        "signal_class":   order.signal,
        "direction":      order.direction,
        "entry_price":    entry_price,
        "size_btc":       size_btc,
        "notional_usd":   notional_usd,
        "leverage":       1,
        "stop_loss_price": stop_price,
        "close_after":    close_after,
        "confidence":     round(confidence, 4),
        "status":         "open",
        "current_price":  entry_price,
        "unrealized_pnl": 0.0,
        "last_updated":   now.isoformat(timespec="seconds"),
        "close_price":    None,
        "close_time":     None,
        "close_reason":   None,
        "pnl":            None,
        "pnl_pct":        None,
    }
    trades = load_futures_trades()
    trades.append(trade)
    _save_futures_trades(trades)

    from trading.futures_notify import notify_futures_trade_entered
    notify_futures_trade_entered(
        trade_id=trade_id,
        signal_name=signal_name,
        direction=order.direction,
        entry_price=entry_price,
        size_btc=size_btc,
        notional_usd=notional_usd,
        stop_loss_price=stop_price,
        close_after=close_after,
        confidence=confidence,
    )
    log.info(
        "Futures trade entered: %s %s @ $%.2f  %.6f BTC  notional $%.2f  stop $%.2f",
        trade_id, order.direction, entry_price, size_btc, notional_usd, stop_price,
    )
    return trade_id


def close_futures_trade(
    trade: dict[str, Any],
    close_price: float,
    reason: str,
) -> dict[str, Any]:
    """
    Settle a single open futures trade.

    Updates the trade record in-place with close_price, pnl, etc., persists
    the full trades list, and fires a Telegram notification.

    Parameters
    ----------
    trade       : The trade dict (must be the live object from load_futures_trades()).
    close_price : BTC/USD price at close.
    reason      : "stop_loss" | "72h_expiry" | "hold_signal" | "manual"

    Returns
    -------
    The updated trade dict.
    """
    pnl_usd, pnl_pct = _calc_pnl(
        trade["direction"], trade["entry_price"], close_price, trade["size_btc"]
    )
    now = datetime.now()
    trade.update(
        status="closed",
        close_price=close_price,
        close_time=now.isoformat(timespec="seconds"),
        close_reason=reason,
        pnl=pnl_usd,
        pnl_pct=pnl_pct,
        current_price=close_price,
    )
    sign = "+" if pnl_usd >= 0 else ""
    log.info(
        "Futures trade closed: %s %s  entry $%.2f → close $%.2f  "
        "P&L %s$%.2f (%s%.2f%%)  reason=%s",
        trade["id"], trade["direction"],
        trade["entry_price"], close_price,
        sign, pnl_usd, sign, pnl_pct * 100, reason,
    )

    # Persist
    trades = load_futures_trades()
    for i, t in enumerate(trades):
        if t["id"] == trade["id"]:
            trades[i] = trade
            break
    _save_futures_trades(trades)

    from trading.futures_notify import notify_futures_trade_closed
    notify_futures_trade_closed(trade)
    return trade


def mark_to_market_futures() -> list[dict[str, Any]]:
    """
    Refresh current_price and unrealized_pnl for every open futures position.

    Also auto-closes any position that has:
    - Hit its stop-loss price, or
    - Exceeded its 72-hour prediction window (close_after).

    Sends a daily Telegram MTM notification for remaining open positions.

    Returns
    -------
    List of trade dicts that are still open after this run.
    """
    trades = load_futures_trades()
    now    = datetime.now()

    spot = _btc_spot()
    if spot is None:
        log.warning("MTM skipped: could not fetch BTC spot price.")
        return [t for t in trades if t.get("status") == "open"]

    still_open: list[dict] = []
    auto_closed: list[dict] = []

    for trade in trades:
        if trade.get("status") != "open":
            continue

        # Update current price and unrealized P&L
        trade["current_price"] = spot
        pnl_usd, _ = _calc_pnl(
            trade["direction"], trade["entry_price"], spot, trade["size_btc"]
        )
        trade["unrealized_pnl"] = pnl_usd
        trade["last_updated"]   = now.isoformat(timespec="seconds")

        # Check stop-loss
        if _stop_triggered(trade["direction"], spot, trade["stop_loss_price"]):
            close_futures_trade(trade, spot, "stop_loss")
            auto_closed.append(trade)
            continue

        # Check 72-hour expiry
        try:
            close_after = datetime.fromisoformat(trade["close_after"])
        except (KeyError, ValueError):
            close_after = None

        if close_after and now >= close_after:
            close_futures_trade(trade, spot, "72h_expiry")
            auto_closed.append(trade)
            continue

        still_open.append(trade)

    # Persist updated prices for trades that are still open
    _save_futures_trades(trades)

    # Log + notify MTM summary
    if still_open:
        total_upnl = sum(t.get("unrealized_pnl", 0.0) for t in still_open)
        sign = "+" if total_upnl >= 0 else ""
        log.info(
            "Futures MTM — %d open position(s)  total unrealized P&L: %s$%.2f",
            len(still_open), sign, total_upnl,
        )
        for t in still_open:
            upnl = t.get("unrealized_pnl", 0.0)
            usign = "+" if upnl >= 0 else ""
            log.info(
                "  %s  %s  entry $%.2f → current $%.2f  (%s$%.2f)",
                t["id"], t["direction"],
                t["entry_price"], t["current_price"],
                usign, upnl,
            )
    else:
        log.info("Futures MTM — no open positions.")

    from trading.futures_notify import notify_futures_mtm
    notify_futures_mtm(still_open, spot)

    return still_open


def close_open_on_hold(btc_spot: float) -> list[dict[str, Any]]:
    """
    Close all open futures positions because the model emitted HOLD.

    Returns the list of trade dicts that were closed.
    """
    open_trades = load_open_futures_trades()
    closed = []
    for trade in open_trades:
        close_futures_trade(trade, btc_spot, "hold_signal")
        closed.append(trade)
    return closed


def get_futures_summary() -> dict[str, Any]:
    """
    Compute aggregate performance statistics across all recorded futures trades.

    Returns a dict with keys:
        total_trades, open_trades, closed_trades,
        wins, losses, win_rate,
        realized_pnl, unrealized_pnl, total_pnl,
        best_trade, worst_trade, open_positions.
    """
    trades = load_futures_trades()

    closed = [t for t in trades if t["status"] == "closed" and t.get("pnl") is not None]
    open_t = [t for t in trades if t["status"] == "open"]
    wins   = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]

    realized   = round(sum(t["pnl"]                  for t in closed), 2)
    unrealized = round(sum(t.get("unrealized_pnl", 0) for t in open_t), 2)

    return {
        "total_trades":   len(trades),
        "open_trades":    len(open_t),
        "closed_trades":  len(closed),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       len(wins) / len(closed) if closed else 0.0,
        "realized_pnl":   realized,
        "unrealized_pnl": unrealized,
        "total_pnl":      round(realized + unrealized, 2),
        "best_trade":     max(closed, key=lambda t: t["pnl"]) if closed else None,
        "worst_trade":    min(closed, key=lambda t: t["pnl"]) if closed else None,
        "open_positions": open_t,
    }


def send_futures_weekly_if_due() -> bool:
    """
    Send a Telegram weekly summary if today is Monday and one hasn't been sent yet.

    Returns True if a summary was sent this call.
    """
    today = date.today()
    if today.weekday() != 0:   # 0 = Monday
        return False
    _ensure()
    if _WEEKLY_FILE.exists():
        try:
            last = date.fromisoformat(_WEEKLY_FILE.read_text().strip())
            if last >= today:
                return False
        except ValueError:
            pass
    from trading.futures_notify import notify_futures_weekly_summary
    sent = notify_futures_weekly_summary(get_futures_summary())
    if sent:
        _WEEKLY_FILE.write_text(today.isoformat())
    return sent

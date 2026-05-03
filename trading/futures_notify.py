"""Telegram notification helpers for the Fixelnet futures paper trader.

All public functions return True on success, False on failure/unconfigured.
They never raise — a notification failure must never abort a trade.
"""
from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime
from typing import Any

import requests

_API_BASE = "https://api.telegram.org"
_TIMEOUT  = 10

log = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def send(text: str, parse_mode: str = "HTML") -> bool:
    """Send a plain message to the configured Telegram chat."""
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID",   "").strip()
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"{_API_BASE}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception:
        log.warning("Telegram send failed:\n%s", traceback.format_exc())
        return False


def notify_futures_trade_entered(
    trade_id:        str,
    signal_name:     str,
    direction:       str,
    entry_price:     float,
    size_btc:        float,
    notional_usd:    float,
    stop_loss_price: float,
    close_after:     str,
    confidence:      float,
) -> bool:
    """Fired when a new futures paper trade is recorded."""
    dir_label = "🟢 LONG" if direction == "long" else "🔴 SHORT"
    text = (
        f"<b>Fixelnet Futures — Trade Entered</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>ID</b>          <code>{trade_id}</code>\n"
        f"<b>Signal</b>      {signal_name}\n"
        f"<b>Direction</b>   {dir_label}\n"
        f"<b>Entry</b>       ${entry_price:,.2f}\n"
        f"<b>Size</b>        {size_btc:.6f} BTC  (${notional_usd:,.2f} notional)\n"
        f"<b>Stop loss</b>   ${stop_loss_price:,.2f}  (−2%)\n"
        f"<b>Auto-close</b>  {close_after}  (72h)\n"
        f"<b>Confidence</b>  {confidence:.2%}"
    )
    return send(text)


def notify_futures_trade_closed(trade: dict[str, Any]) -> bool:
    """Fired when a futures position is closed for any reason."""
    pnl    = trade.get("pnl", 0.0) or 0.0
    pnl_pct = trade.get("pnl_pct", 0.0) or 0.0
    sign   = "+" if pnl >= 0 else ""
    result = "WIN ✅" if pnl > 0 else ("LOSS ❌" if pnl < 0 else "SCRATCH")
    reason_labels = {
        "stop_loss":   "Stop-loss triggered",
        "72h_expiry":  "72-hour prediction window elapsed",
        "hold_signal": "HOLD signal — closed by model",
        "manual":      "Manual close",
    }
    reason_label = reason_labels.get(trade.get("close_reason", ""), trade.get("close_reason", ""))
    dir_label = "LONG" if trade.get("direction") == "long" else "SHORT"
    text = (
        f"<b>Fixelnet Futures — Position Closed</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>ID</b>          <code>{trade['id']}</code>\n"
        f"<b>Signal</b>      {trade.get('signal', '—')}\n"
        f"<b>Direction</b>   {dir_label}\n"
        f"<b>Entry</b>       ${trade['entry_price']:,.2f}\n"
        f"<b>Close</b>       ${trade.get('close_price', 0):,.2f}\n"
        f"<b>Size</b>        {trade.get('size_btc', 0):.6f} BTC\n\n"
        f"<b>P&amp;L</b>         {sign}${pnl:,.2f}  ({sign}{pnl_pct*100:.2f}%)\n"
        f"<b>Result</b>      {result}\n"
        f"<b>Reason</b>      {reason_label}"
    )
    return send(text)


def notify_futures_hold_close(
    signal_name: str,
    confidence:  float,
    btc_spot:    float,
    closed_ids:  list[str],
) -> bool:
    """Fired when HOLD signal closes open position(s) without opening a new one."""
    ids_str = ", ".join(f"<code>{i}</code>" for i in closed_ids) if closed_ids else "none"
    text = (
        f"<b>Fixelnet Futures — HOLD Signal</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Signal</b>      {signal_name}\n"
        f"<b>Confidence</b>  {confidence:.2%}\n"
        f"<b>BTC spot</b>    ${btc_spot:,.2f}\n\n"
        f"Closed position(s): {ids_str}\n"
        f"No new trade placed."
    )
    return send(text)


def notify_futures_no_trade(
    signal_name: str,
    confidence:  float,
    btc_spot:    float,
    reason:      str,
) -> bool:
    """Fired when a directional signal is blocked (risk rules or no spot price)."""
    text = (
        f"<b>Fixelnet Futures — Trade Blocked</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Signal</b>      {signal_name}\n"
        f"<b>Confidence</b>  {confidence:.2%}\n"
        f"<b>BTC spot</b>    ${btc_spot:,.2f}\n\n"
        f"<b>Reason</b>      {reason}"
    )
    return send(text)


def notify_futures_mtm(open_trades: list[dict[str, Any]], btc_spot: float) -> bool:
    """Fired once per daily MTM run for all open futures positions."""
    if not open_trades:
        return send(
            f"<b>Fixelnet Futures — Daily MTM</b>\n<i>{_ts()}</i>\n\n"
            f"BTC spot ${btc_spot:,.2f}\n\nNo open positions."
        )
    lines: list[str] = []
    total_upnl = 0.0
    for t in open_trades:
        upnl  = t.get("unrealized_pnl", 0.0)
        total_upnl += upnl
        sign  = "+" if upnl >= 0 else ""
        arrow = "▲" if upnl > 0 else ("▼" if upnl < 0 else "·")
        dir_label = "LONG" if t.get("direction") == "long" else "SHORT"
        lines.append(
            f"  <code>{t['id']}</code>  {dir_label}  "
            f"entry ${t['entry_price']:,.0f} → ${t['current_price']:,.0f}  "
            f"{sign}${upnl:,.2f} {arrow}"
        )
    total_sign = "+" if total_upnl >= 0 else ""
    text = (
        f"<b>Fixelnet Futures — Daily MTM</b>\n<i>{_ts()}</i>\n\n"
        f"BTC spot ${btc_spot:,.2f}\n\n"
        + "\n".join(lines)
        + f"\n\n<b>Total unrealized: {total_sign}${total_upnl:,.2f}</b>"
    )
    return send(text)


def notify_futures_weekly_summary(stats: dict[str, Any]) -> bool:
    """Fired every Monday with aggregate performance across all futures trades."""
    win_rate = stats.get("win_rate", 0.0)
    r_pnl    = stats.get("realized_pnl", 0.0)
    u_pnl    = stats.get("unrealized_pnl", 0.0)
    total    = stats.get("total_pnl", 0.0)

    def _fmt(v: float) -> str:
        return f"{'+'if v >= 0 else ''}${v:,.2f}"

    best  = stats.get("best_trade")
    worst = stats.get("worst_trade")
    best_str  = f"{_fmt(best['pnl'])}  ({best['id']})" if best else "—"
    worst_str = f"{_fmt(worst['pnl'])}  ({worst['id']})" if worst else "—"

    text = (
        f"<b>Fixelnet Futures — Weekly Summary</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Total trades</b>   {stats.get('total_trades', 0)}\n"
        f"<b>Closed</b>         {stats.get('closed_trades', 0)}"
        f"  ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)\n"
        f"<b>Win rate</b>       {win_rate:.1%}\n\n"
        f"<b>Realized P&amp;L</b>   {_fmt(r_pnl)}\n"
        f"<b>Unrealized P&amp;L</b> {_fmt(u_pnl)}\n"
        f"<b>Total P&amp;L</b>      {_fmt(total)}\n\n"
        f"<b>Best trade</b>     {best_str}\n"
        f"<b>Worst trade</b>    {worst_str}"
    )
    return send(text)


def notify_error(context: str, exc: BaseException) -> bool:
    """Fired on any unhandled exception during the futures trading pipeline."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb)[-800:]
    text = (
        f"<b>Fixelnet Futures — Error</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Context</b>  {context}\n\n"
        f"<pre>{tb_str}</pre>"
    )
    return send(text)

"""Telegram notification helpers for the Fixelnet paper trader.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment.
If either is absent, every send() call is a silent no-op so the trader
continues to work without Telegram configured.

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
_TIMEOUT = 10  # seconds

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core send primitive
# ---------------------------------------------------------------------------

def send(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a plain message to the configured Telegram chat.

    Parameters
    ----------
    text       : Message body. HTML tags (<b>, <i>, <code>, <pre>) are
                 supported with the default parse_mode.
    parse_mode : "HTML" (default) or "MarkdownV2".

    Returns
    -------
    True if the message was delivered, False otherwise.
    """
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
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


# ---------------------------------------------------------------------------
# Structured notification functions
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def notify_trade_placed(
    signal_name: str,
    option_type: str,
    option_symbol: str,
    strike: float,
    expiry: str,
    dte: int,
    quantity: int,
    limit_price: float,
    total_cost: float,
    order_id: Any,
    order_status: str,
    btc_spot: float,
    dry_run: bool = False,
) -> bool:
    """Fired when an order is successfully submitted (or would be in dry-run)."""
    label = "[DRY RUN] " if dry_run else ""
    direction = "CALL (Bullish)" if option_type == "call" else "PUT (Bearish)"
    text = (
        f"<b>{label}Fixelnet — Trade Placed</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Signal</b>      {signal_name}\n"
        f"<b>Direction</b>   {direction}\n"
        f"<b>Symbol</b>      <code>{option_symbol}</code>\n"
        f"<b>Strike</b>      ${strike:,.0f}\n"
        f"<b>Expiry</b>      {expiry}  ({dte} DTE)\n"
        f"<b>BTC spot</b>    ${btc_spot:,.2f}\n\n"
        f"<b>Qty</b>         {quantity} contract(s)\n"
        f"<b>Limit</b>       ${limit_price:,.2f}\n"
        f"<b>Total debit</b> ${total_cost:,.2f}\n\n"
        f"<b>Order ID</b>    <code>{order_id}</code>\n"
        f"<b>Status</b>      {order_status}"
    )
    return send(text)


def notify_skipped_hold(
    signal_name: str,
    confidence: float,
    btc_spot: float,
    long_s: float,
    heavy_s: float,
    short_s: float,
) -> bool:
    """Fired when the signal is HOLD (SHORT class) — no trade taken."""
    text = (
        f"<b>Fixelnet — No Trade (HOLD)</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Signal</b>      {signal_name}\n"
        f"<b>Confidence</b>  {confidence:.2%}\n"
        f"<b>BTC spot</b>    ${btc_spot:,.2f}\n\n"
        f"Long {long_s:.3f}  |  Heavy short {heavy_s:.3f}  |  Short {short_s:.3f}"
    )
    return send(text)


def notify_skipped_risk(
    signal_name: str,
    confidence: float,
    btc_spot: float,
    reason: str,
) -> bool:
    """Fired when a directional signal is blocked by risk rules."""
    text = (
        f"<b>Fixelnet — Trade Blocked (Risk)</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Signal</b>      {signal_name}\n"
        f"<b>Confidence</b>  {confidence:.2%}\n"
        f"<b>BTC spot</b>    ${btc_spot:,.2f}\n\n"
        f"<b>Reason</b>  {reason}"
    )
    return send(text)


def notify_auth_renewal_needed() -> bool:
    """Fired when the cached session has expired and interactive re-auth is required."""
    text = (
        f"<b>Fixelnet — Auth Renewal Required</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"The Tastytrade session token has expired. "
        f"Re-authenticate by running:\n\n"
        f"<pre>python3 /root/fixelnet/test_auth.py</pre>\n\n"
        f"Then set <code>TASTYTRADE_OTP</code> in <code>.env</code> and run again."
    )
    return send(text)


def notify_trade_entered(
    trade_id: str,
    signal_name: str,
    symbol: str,
    strike: float,
    expiry: str,
    option_type: str,
    quantity: int,
    entry_price: float,
    budget: float,
    total_cost: float,
) -> bool:
    """Fired when the simulator records a new simulated trade entry."""
    direction = "PUT (Bearish)" if option_type == "put" else "CALL (Bullish)"
    text = (
        f"<b>Fixelnet — Simulated Trade Entered</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>ID</b>          <code>{trade_id}</code>\n"
        f"<b>Signal</b>      {signal_name}\n"
        f"<b>Direction</b>   {direction}\n"
        f"<b>Symbol</b>      <code>{symbol}</code>\n"
        f"<b>Strike</b>      ${strike:,.2f}/share\n"
        f"<b>Expiry</b>      {expiry}\n\n"
        f"<b>Qty</b>         {quantity} contract(s)\n"
        f"<b>Entry price</b> ${entry_price:,.2f}/share\n"
        f"<b>Total cost</b>  ${total_cost:,.2f}  (budget ${budget:,.0f})"
    )
    return send(text)


def notify_mark_to_market(positions: list[dict]) -> bool:
    """Fired once per daily MTM run with unrealized P&L for all open positions."""
    if not positions:
        return send(
            f"<b>Fixelnet — Daily Mark-to-Market</b>\n<i>{_ts()}</i>\n\nNo open positions."
        )
    lines = []
    total_upnl = 0.0
    for p in positions:
        upnl = p.get("unrealized_pnl", 0.0)
        total_upnl += upnl
        sign = "+" if upnl >= 0 else ""
        arrow = "▲" if upnl > 0 else ("▼" if upnl < 0 else "·")
        lines.append(
            f"  <code>{p['trade_id']}</code>  {p['symbol']}  "
            f"@ ${p['current_price']:.2f}  {sign}${upnl:,.2f} {arrow}"
        )
    total_sign = "+" if total_upnl >= 0 else ""
    text = (
        f"<b>Fixelnet — Daily Mark-to-Market</b>\n<i>{_ts()}</i>\n\n"
        + "\n".join(lines)
        + f"\n\n<b>Total unrealized P&L: {total_sign}${total_upnl:,.2f}</b>"
    )
    return send(text)


def notify_position_closed(trade: dict, btcw_spot: float | None = None) -> bool:
    """Fired when a position expires and is settled at intrinsic value."""
    pnl        = trade.get("pnl", 0.0)
    sign       = "+" if pnl >= 0 else ""
    result     = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "SCRATCH")
    spot_line  = f"\n<b>BTCW spot</b>    ${btcw_spot:.3f}/share" if btcw_spot else ""
    text = (
        f"<b>Fixelnet — Position Closed at Expiry</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>ID</b>          <code>{trade['id']}</code>\n"
        f"<b>Signal</b>      {trade['signal']}\n"
        f"<b>Symbol</b>      <code>{trade['symbol']}</code>\n"
        f"<b>Strike</b>      ${trade['strike']:,.2f}/share\n"
        f"<b>Expiry</b>      {trade['expiry']}{spot_line}\n\n"
        f"<b>Entry price</b> ${trade['entry_price']:,.2f}/share\n"
        f"<b>Close price</b> ${trade.get('close_price', 0):,.2f}/share  (intrinsic)\n"
        f"<b>Qty</b>         {trade['quantity']} contract(s)\n\n"
        f"<b>Final P&L</b>   {sign}${pnl:,.2f}  —  {result}"
    )
    return send(text)


def notify_weekly_summary(stats: dict) -> bool:
    """Fired every Monday with aggregate performance across all recorded trades."""
    win_rate = stats.get("win_rate", 0.0)
    r_pnl    = stats.get("realized_pnl", 0.0)
    u_pnl    = stats.get("unrealized_pnl", 0.0)
    total    = stats.get("total_pnl", 0.0)

    def _fmt(v: float) -> str:
        return f"{'+'if v >= 0 else ''}${v:,.2f}"

    text = (
        f"<b>Fixelnet — Weekly Summary</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Total trades</b>   {stats.get('total_trades', 0)}\n"
        f"<b>Closed</b>         {stats.get('closed_trades', 0)}"
        f"  ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)\n"
        f"<b>Win rate</b>       {win_rate:.1%}\n\n"
        f"<b>Realized P&L</b>   {_fmt(r_pnl)}\n"
        f"<b>Unrealized P&L</b> {_fmt(u_pnl)}\n"
        f"<b>Total P&L</b>      {_fmt(total)}"
    )
    return send(text)


def notify_error(context: str, exc: BaseException) -> bool:
    """Fired on any unhandled exception during the trading pipeline."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb)[-800:]  # cap at 800 chars to stay within Telegram's 4096 limit
    text = (
        f"<b>Fixelnet — Error</b>\n"
        f"<i>{_ts()}</i>\n\n"
        f"<b>Context</b>  {context}\n\n"
        f"<pre>{tb_str}</pre>"
    )
    return send(text)

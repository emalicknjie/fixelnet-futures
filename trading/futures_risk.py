"""Position sizing and pre-trade risk checks for Fixelnet futures paper trading.

Signal sizing:
    LONG  (0)  →  1× base notional  ($500)
    SELL  (1)  →  2× base notional  ($1000)
    HOLD  (2)  →  no new trade

Risk rules (all must pass for a trade to be placed):
    1. Confidence must be >= MIN_CONFIDENCE (0.65).
    2. No more than MAX_OPEN_POSITIONS (1) open position at once.
    3. Trade notional must be > 0 (needs positive BTC spot price).

Stop loss:
    Long  positions: stop at entry_price × (1 − STOP_LOSS_PCT)  = −2%
    Short positions: stop at entry_price × (1 + STOP_LOSS_PCT)  = −2%
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── Risk constants ─────────────────────────────────────────────────────────────
BASE_NOTIONAL_USD:      float = 500.0   # base position size in USD notional
HEAVY_SHORT_MULTIPLIER: float = 2.0    # HEAVY_SHORT is 2× the base
MAX_OPEN_POSITIONS:     int   = 1      # only 1 futures position at a time
STOP_LOSS_PCT:          float = 0.02   # 2% adverse move triggers stop-loss close
MIN_CONFIDENCE:         float = 0.65   # minimum softmax probability to trade

# Signal class constants (mirrors futures_strategy.py)
_LONG        = 0
_HEAVY_SHORT = 1
_HOLD        = 2


@dataclass
class FuturesRiskCheck:
    """Outcome of a pre-trade futures risk evaluation."""
    approved:     bool
    reason:       str
    notional_usd: float   # position size in USD
    size_btc:     float   # position size in BTC (notional / spot)


def notional_for_signal(signal: int) -> float:
    """Return USD notional for the given signal class."""
    return BASE_NOTIONAL_USD * (HEAVY_SHORT_MULTIPLIER if signal == _HEAVY_SHORT else 1.0)


def compute_stop_loss(direction: str, entry_price: float) -> float:
    """
    Compute the stop-loss trigger price for a futures position.

    Long:  stop = entry × (1 − STOP_LOSS_PCT)
    Short: stop = entry × (1 + STOP_LOSS_PCT)
    """
    if direction == "long":
        return round(entry_price * (1.0 - STOP_LOSS_PCT), 2)
    return round(entry_price * (1.0 + STOP_LOSS_PCT), 2)


def check_futures_risk(
    signal: int,
    confidence: float,
    open_trades: list[dict[str, Any]],
    btc_spot: float,
) -> FuturesRiskCheck:
    """
    Evaluate all pre-trade risk rules.

    Parameters
    ----------
    signal      : Fixelnet class (LONG=0, HEAVY_SHORT=1).
    confidence  : Softmax probability of the predicted class (0–1).
    open_trades : Currently open futures trade records from the simulator.
    btc_spot    : Current BTC/USD spot price.

    Returns
    -------
    FuturesRiskCheck with approved=True only if every rule passes.
    """
    # Rule 1 — confidence threshold
    if confidence < MIN_CONFIDENCE:
        return FuturesRiskCheck(
            approved=False,
            reason=(
                f"Confidence {confidence:.2%} is below minimum {MIN_CONFIDENCE:.0%}. "
                "Skipping trade."
            ),
            notional_usd=0.0,
            size_btc=0.0,
        )

    # Rule 2 — open positions cap
    open_count = len(open_trades)
    if open_count >= MAX_OPEN_POSITIONS:
        return FuturesRiskCheck(
            approved=False,
            reason=(
                f"Already have {open_count} open position(s) "
                f"(max {MAX_OPEN_POSITIONS}). Skipping new entry."
            ),
            notional_usd=0.0,
            size_btc=0.0,
        )

    # Rule 3 — valid spot price
    if btc_spot <= 0:
        return FuturesRiskCheck(
            approved=False,
            reason="BTC spot price unavailable or zero — cannot size position.",
            notional_usd=0.0,
            size_btc=0.0,
        )

    notional = notional_for_signal(signal)
    size_btc = round(notional / btc_spot, 8)

    return FuturesRiskCheck(
        approved=True,
        reason=(
            f"All risk checks passed. Confidence={confidence:.2%}, "
            f"open positions={open_count}/{MAX_OPEN_POSITIONS}, "
            f"notional=${notional:.2f} ({size_btc:.6f} BTC @ ${btc_spot:,.2f})."
        ),
        notional_usd=notional,
        size_btc=size_btc,
    )

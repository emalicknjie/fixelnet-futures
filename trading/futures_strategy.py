"""Signal-to-futures-order mapping for Fixelnet paper futures trading.

Signal classes (original notebook intent):
    LONG         = 0  →  Enter long   (1× base notional)   [notebook: BUY]
    HEAVY_SHORT  = 1  →  Enter short  (2× base notional)   [notebook: SELL]
    HOLD         = 2  →  Close any open position, no new trade [notebook: HOLD]

All positions are unleveraged (leverage=1).  Size is purely notional/spot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Mirror fixelnet.signals constants
LONG        = 0
HEAVY_SHORT = 1
HOLD        = 2

SIGNAL_NAMES: dict[int, str] = {
    LONG:        "LONG (BUY)",
    HEAVY_SHORT: "SELL (SHORT 2×)",
    HOLD:        "HOLD (CLOSE)",
}

# Direction strings used in trade records
DIRECTION_LONG  = "long"
DIRECTION_SHORT = "short"


@dataclass
class FuturesOrderParams:
    """Parameters for a single futures paper trade."""
    direction:        str    # "long" or "short"
    size_multiplier:  float  # 1.0 for LONG, 2.0 for HEAVY_SHORT
    signal:           int    # original Fixelnet class
    confidence:       float  # softmax probability for predicted class


def signal_to_futures_order(
    signal: int,
    confidence: float,
) -> FuturesOrderParams | None:
    """
    Convert a Fixelnet signal to a FuturesOrderParams.

    Returns None for HOLD (class 2) — the caller should close any open
    position and record no new entry.

    Parameters
    ----------
    signal     : Fixelnet class (LONG=0, HEAVY_SHORT=1, HOLD=2).
    confidence : Softmax probability for the predicted class (0–1).

    Returns
    -------
    FuturesOrderParams, or None when signal is HOLD.
    """
    if signal == HOLD:
        return None

    direction       = DIRECTION_LONG  if signal == LONG else DIRECTION_SHORT
    size_multiplier = 1.0             if signal == LONG else 2.0

    return FuturesOrderParams(
        direction=direction,
        size_multiplier=size_multiplier,
        signal=signal,
        confidence=confidence,
    )

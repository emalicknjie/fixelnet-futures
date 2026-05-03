"""Technical indicator computation and fixel extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from stockstats import StockDataFrame as sdf

LOOKBACK_DAYS = 15
# All six indicators used by fixelnet; order here is arbitrary — columns are
# sorted alphabetically before reshape so the network always sees the same layout.
_INDICATOR_NAMES = ['atr', 'cci', 'rsi', 'rsv', 'vr', 'wr']

# 90 column names in the exact order produced by sort_index(axis=1):
# lexicographic sort, e.g. atr_1, atr_10, atr_11, ..., atr_9, cci_1, ...
_FIXEL_COLUMNS: list[str] = sorted(
    f'{ind}_{day}'
    for ind in _INDICATOR_NAMES
    for day in range(1, LOOKBACK_DAYS + 1)
)


def compute_fixels(historical_data: pd.DataFrame) -> np.ndarray:
    """
    Compute MinMax-normalised fixels for every row in historical_data.

    Each fixel is a (6, 15) array: 6 indicator groups × 15 lookback positions.
    The indicator axis order matches alphabetical sort: atr, cci, rsi, rsv, vr, wr.
    NaN values (e.g. weekends with missing closes) are zeroed out.

    Returns array of shape (n_rows, 6, 15).
    """
    stock_df = sdf.retype(historical_data.copy())

    # Trigger StockStats computation for every indicator at every lookback window
    for day in range(1, LOOKBACK_DAYS + 1):
        w = str(day)
        for ind in _INDICATOR_NAMES:
            _ = stock_df[f'{ind}_{w}']

    selected = pd.DataFrame(stock_df)[_FIXEL_COLUMNS]

    scaler = MinMaxScaler(feature_range=(0, 1))
    fixels: list[list] = []
    for i in range(selected.shape[0]):
        row = selected.iloc[i:i + 1].sort_index(axis=1)
        row = row.replace([np.inf, -np.inf], np.nan)
        arr = np.asarray(row).reshape(6, -1)
        arr = scaler.fit_transform(arr)
        fixels.append(arr.tolist())

    result = np.asarray(fixels)
    result[np.isnan(result)] = 0.0
    return result


def get_latest_fixel(historical_data: pd.DataFrame) -> np.ndarray:
    """
    Return a single (6, 15) fixel for a windowed price history slice.

    Uses index [1] rather than [0] because the first row of a short window
    commonly has NaN-heavy rolling indicators; index 1 is the first row with
    meaningful values in a ±5-day window.
    """
    fixels = compute_fixels(historical_data)
    return fixels[1]

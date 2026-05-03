"""Buy/sell/hold signal orchestration, statistics, and output artefacts."""
from __future__ import annotations

from calendar import monthrange
from datetime import datetime
from pathlib import Path
from typing import Iterator

import dataframe_image as dfi
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import fetch_price_history, get_price_window
from .features import get_latest_fixel
from .models import FixelNet, predict

# Prediction class constants (match neural network output indices)
LONG = 0
HEAVY_SHORT = 1
SHORT = 2

ASSETS = ['BTCUSDT', 'SOLUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT']

_PRED_COLUMNS = [
    'timestamp', 'Prediction',
    'Long neural strength', 'Heavy short neural strength', 'Short neural strength',
    'timestamp receipt', 'close',
]


def _iter_year_dates(year: int) -> Iterator[tuple[str, str, str]]:
    """Yield (month_str, day_str, year_str) for every calendar day in year."""
    for month in range(1, 13):
        days_in_month = monthrange(year, month)[1]
        for day in range(1, days_in_month + 1):
            yield str(month), str(day), str(year)


def run_predictions(
    model: FixelNet,
    year: int = 2025,
    assets: list[str] | None = None,
    output_dir: str | Path = '/home/emalick/Desktop/Trading/fixelnet_server',
    interval_days: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Run fixelnet predictions for every day in `year` for each asset.

    For each asset:
    - Fetches price history from Kraken (no auth required)
    - Builds a fixel for every day and runs model inference
    - Saves CSV predictions and PNG plot/table artefacts to output_dir

    Returns a dict mapping asset symbol → full predictions DataFrame.
    """
    if assets is None:
        assets = ASSETS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for asset in assets:
        current_date = datetime.today().strftime('%d %b, %Y')
        price_history = fetch_price_history(asset, f'1 Jan, {year}', current_date)

        records: list[tuple] = []
        for month_s, day_s, year_s in _iter_year_dates(year):
            centroid = datetime.strptime(
                f'{month_s}-{day_s}-{year_s} 00:00:00', '%m-%d-%Y %H:%M:%S'
            )
            try:
                window, receipt_ts, receipt_close = get_price_window(
                    price_history, centroid, interval_days
                )
                fixel = get_latest_fixel(window)
                pred, long_s, heavy_s, short_s = predict(model, fixel)
                records.append((
                    centroid.strftime('%m-%d-%Y'),
                    pred, long_s, heavy_s, short_s,
                    receipt_ts, receipt_close,
                ))
            except Exception:
                break

        df = pd.DataFrame(records, columns=_PRED_COLUMNS)
        df.to_csv(output_dir / f'centroid_date_fixel_pred_{asset}_{year}.csv', index=False)

        plot_signals(df, output_dir / f'trading_plot_{asset}.png')
        export_table(df, output_dir / f'trading_table_{asset}.png', output_dir / f'trading_table_{asset}.csv')

        results[asset] = df

    return results


def compute_signal_stats(df: pd.DataFrame) -> dict[str, float | int]:
    """
    Return prediction counts, percentages, and mean/SEM neural strengths
    for each signal class.
    """
    buy = df.loc[df['Prediction'] == LONG]
    sell = df.loc[df['Prediction'] == HEAVY_SHORT]
    hold = df.loc[df['Prediction'] == SHORT]
    total = len(df)

    return {
        'buy_count': len(buy),
        'sell_count': len(sell),
        'hold_count': len(hold),
        'buy_pct': int(len(buy) / total * 100),
        'sell_pct': int(len(sell) / total * 100),
        'hold_pct': int(len(hold) / total * 100),
        'buy_strength_mean': float(buy['Long neural strength'].mean()),
        'buy_strength_sem': float(buy['Long neural strength'].sem()),
        'sell_strength_mean': float(sell['Heavy short neural strength'].mean()),
        'sell_strength_sem': float(sell['Heavy short neural strength'].sem()),
        'hold_strength_mean': float(hold['Short neural strength'].mean()),
        'hold_strength_sem': float(hold['Short neural strength'].sem()),
    }


def plot_signals(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save a Long-Short binary vs Heavy Short line plot for the last 4 predictions.
    Long-short binary = long_strength − short_strength (positive = bullish).
    Heavy Short = −heavy_short_strength (plotted negative to show bearish pressure).
    """
    recent = df.tail(4)[
        ['timestamp', 'Long neural strength', 'Short neural strength', 'Heavy short neural strength']
    ].copy()
    recent['Long-short binary'] = recent['Long neural strength'] - recent['Short neural strength']
    recent['Heavy Short'] = recent['Heavy short neural strength'] * -1

    plt.rcParams['figure.dpi'] = 100
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(recent['timestamp'], recent['Long-short binary'], color='black', linestyle='-', label='Long-short binary')
    ax.plot(recent['timestamp'], recent['Heavy Short'], color='red', linestyle='-')

    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis='y', labelsize=12, labelweight='bold')
    ax.grid(True)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    plt.xticks(recent['timestamp'], rotation=45, fontsize=12, fontweight='bold')

    ax.text(x=-0.5, y=0.5, s='Long', va='center', ha='right', fontsize=12, color='black', fontweight='bold')
    ax.text(x=-0.5, y=-0.5, s='Short', va='center', ha='right', fontsize=12, color='black', fontweight='bold')
    ax.text(1.02, 0.5, 'Heavy Short', transform=ax.transAxes, color='red', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def export_table(
    df: pd.DataFrame,
    png_path: str | Path,
    csv_path: str | Path,
) -> None:
    """
    Export the last 4 prediction rows as a styled PNG table and a CSV.
    The PNG uses background-gradient styling; the CSV contains the computed
    Long-short binary and Heavy Short columns.
    """
    recent = df.tail(4).copy()

    # PNG: full predictions with background gradient
    styled = recent.style.background_gradient()
    dfi.export(styled, str(png_path))

    # CSV: derived signal columns for downstream use
    recent['Long-short binary'] = recent['Long neural strength'] - recent['Short neural strength']
    recent['Heavy Short'] = recent['Heavy short neural strength'] * -1
    recent[['timestamp', 'Prediction', 'Long-short binary', 'Heavy Short']].to_csv(csv_path, index=False)

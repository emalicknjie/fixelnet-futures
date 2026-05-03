"""Kraken data fetching and price-window extraction."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

DATE_FMT = '%m-%d-%Y'

# Map Binance-style symbols to Kraken pair names
_KRAKEN_PAIR: dict[str, str] = {
    'BTCUSDT':  'XBTUSD',
    'SOLUSDT':  'SOLUSD',
    'ETHUSDT':  'ETHUSD',
    'ADAUSDT':  'ADAUSD',
    'DOGEUSDT': 'DOGEUSD',
}

_OHLCV_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']


def _parse_date(date_str: str) -> datetime:
    """Parse a date string in Binance format '1 Jan, 2025' to datetime."""
    return datetime.strptime(date_str.strip(), '%d %b, %Y')


def fetch_price_history(
    asset: str,
    begin_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV from the Kraken public REST API (no auth required).

    Parameters
    ----------
    asset:      Binance-style symbol, e.g. 'BTCUSDT'.
    begin_date: Start date string in Binance format, e.g. '1 Jan, 2025'.
    end_date:   End date string in Binance format, e.g. '10 Apr, 2026'.

    Returns a DataFrame with columns:
        timestamp (str MM-DD-YYYY), open, high, low, close, volume
    """
    pair = _KRAKEN_PAIR.get(asset, asset)
    since = int(_parse_date(begin_date).replace(tzinfo=timezone.utc).timestamp())
    end_dt = _parse_date(end_date)

    resp = requests.get(
        'https://api.kraken.com/0/public/OHLC',
        params={'pair': pair, 'interval': 1440, 'since': since},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get('error'):
        raise RuntimeError(f'Kraken API error for {asset}: {payload["error"]}')

    key = next(k for k in payload['result'] if k != 'last')
    rows = payload['result'][key]

    if not rows:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(rows, columns=_OHLCV_COLUMNS)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
    df = df[df['timestamp'] <= pd.Timestamp(end_dt)].copy()
    df[['open', 'high', 'low', 'close', 'volume']] = (
        df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    )
    df['timestamp'] = df['timestamp'].dt.strftime(DATE_FMT)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)


def get_price_window(
    price_history: pd.DataFrame,
    centroid_date: datetime,
    interval_days: int = 5,
) -> tuple[pd.DataFrame, str, float]:
    """
    Extract rows within ±interval_days of centroid_date.

    Returns:
        windowed_df: rows in the symmetric window around centroid_date
        receipt_timestamp: centroid date string, or 'future' if centroid is ahead of today
        receipt_close: closing price on centroid_date, or 0.0 for future dates
    """
    begin_dt = centroid_date - timedelta(days=interval_days)
    end_dt = centroid_date + timedelta(days=interval_days)

    df = price_history.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    mask = (df['timestamp'] > begin_dt) & (df['timestamp'] <= end_dt)
    df = df.loc[mask].copy()
    df['timestamp'] = df['timestamp'].dt.strftime(DATE_FMT)

    if centroid_date <= datetime.today():
        centroid_str = centroid_date.strftime(DATE_FMT)
        row = df.loc[df['timestamp'] == centroid_str]
        receipt_ts = str(row['timestamp'].iloc[0])
        receipt_close = float(row['close'].iloc[0])
    else:
        receipt_ts, receipt_close = 'future', 0.0

    return df, receipt_ts, receipt_close

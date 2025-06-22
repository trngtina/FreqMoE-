"""
data_processing.py

Functions to fetch, preprocess, normalize, and window financial time series using Finnhub.
"""

import os
import pandas as pd
import numpy as np
import torch
import finnhub


def fetch_data(
    tickers,
    start_date: str,
    end_date: str,
    interval: str = '1d',
    api_key: str = "key",
    tz: str = 'UTC'
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for given tickers via Finnhub API only.
    Splits long date ranges into <=1-year chunks to respect API limits.

    Args:
        tickers (str or list[str]): One or more ticker symbols.
        start_date (str): 'YYYY-MM-DD'.
        end_date   (str): 'YYYY-MM-DD'.
        interval   (str): '1d' for daily, '1h' for hourly, '15m', etc.
        api_key    (str): Finnhub API key or None to read from FINNHUB_API_KEY env var.
        tz         (str): Timezone for the returned index (e.g. 'US/Eastern', 'UTC').

    Returns:
        pd.DataFrame: Adjusted close prices (dates × tickers), timezone-aware.

    Raises:
        ValueError: If no data could be fetched for any tickers.
    """
    # Ensure list of tickers
    if isinstance(tickers, str):
        tickers = [tickers]

    # Get API key
    key = api_key or os.getenv('FINNHUB_API_KEY')
    if not key:
        raise ValueError('Finnhub API key is required.')
    client = finnhub.Client(api_key=key)

    # Map interval to resolution
    if interval.endswith('d'):
        resolution = 'D'
    elif interval.endswith('h'):
        resolution = str(int(interval[:-1]) * 60)
    elif interval.endswith('m'):
        resolution = interval[:-1]
    else:
        resolution = interval

    # Compute UNIX timestamps
    start_ts = int(pd.to_datetime(start_date).timestamp())
    end_ts   = int(pd.to_datetime(end_date).timestamp())
    max_span = 365 * 24 * 3600  # one year in seconds

    frames = []
    for sym in tickers:
        sym_chunks = []
        chunk_start = start_ts
        while chunk_start < end_ts:
            chunk_end = min(chunk_start + max_span, end_ts)
            try:
                resp = client.stock_candles(sym, resolution, chunk_start, chunk_end)
            except Exception:
                break
            if resp.get('s') != 'ok' or 'c' not in resp:
                break
            df_chunk = pd.DataFrame({sym: resp['c'], 't': resp['t']})
            df_chunk.index = pd.to_datetime(df_chunk['t'], unit='s', utc=True)
            df_chunk.index = df_chunk.index.tz_convert(tz)
            df_chunk = df_chunk[[sym]]
            sym_chunks.append(df_chunk)
            if chunk_end >= end_ts:
                break
            chunk_start = chunk_end + 1
        if sym_chunks:
            df_sym = pd.concat(sym_chunks)
            # Remove duplicate timestamps
            df_sym = df_sym[~df_sym.index.duplicated(keep='first')]
            frames.append(df_sym)

    if not frames:
        raise ValueError(f"fetch_data(): No data fetched for tickers {tickers}")

    # Combine all symbols into one DataFrame
    df_all = pd.concat(frames, axis=1).sort_index()
    # Forward-fill gaps and drop any remaining NaNs
    df_all = df_all.ffill().dropna(how='any')
    return df_all

    # ----- Fallback to yfinance if Finnhub fails for all symbols -----
    import yfinance as yf
    yf_frames = []
    for sym in tickers:
        try:
            data = yf.download(sym,
                               start=start_date,
                               end=end_date,
                               interval=interval,
                               progress=False,
                               auto_adjust=True)
        except Exception:
            continue
        if data is None or data.empty or 'Adj Close' not in data.columns:
            continue
        df_t = data['Adj Close'].to_frame(name=sym)
        df_t.index = pd.to_datetime(df_t.index).tz_localize(tz, ambiguous='infer')
        df_t = df_t.ffill().dropna(how='any')
        yf_frames.append(df_t)
    if yf_frames:
        df_all = pd.concat(yf_frames, axis=1).sort_index()
        df_all = df_all.ffill().dropna(how='any')
        return df_all

    # If both fail, raise error
    raise ValueError(f'fetch_data(): No data fetched for tickers {tickers} via Finnhub or yfinance.')


def compute_log_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns: r_t = log(p_t / p_{t-1}).

    Args:
        df_prices (pd.DataFrame): Prices DataFrame.
    Returns:
        pd.DataFrame: Log returns.
    """
    returns = np.log(df_prices / df_prices.shift(1)).dropna()
    return returns


def normalize_series(returns: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Normalize each channel to zero mean and unit variance.

    Args:
        returns (pd.DataFrame): Log returns (T × C).
    Returns:
        X_norm (np.ndarray): (C, T) normalized data.
        means (np.ndarray): (C,) channel means.
        stds (np.ndarray): (C,) channel std deviations.
    """
    X = returns.to_numpy(dtype=float)
    if X.shape[0] < 1:
        raise ValueError(f'normalize_series(): Not enough samples ({X.shape})')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_norm2d = scaler.fit_transform(X)
    means = scaler.mean_
    stds = np.sqrt(scaler.var_)
    return X_norm2d.T, means, stds


def create_windows(
    X: np.ndarray,
    lookback: int,
    horizon: int
) -> (torch.Tensor, torch.Tensor):
    """
    Create sliding windows from multivariate series X (C, T).

    Returns:
        x_windows: (N, C, lookback)
        y_windows: (N, C, horizon)
    """
    C, T = X.shape
    N = T - lookback - horizon + 1
    if N < 1:
        raise ValueError(f'create_windows(): need at least lookback+horizon={lookback+horizon}, got T={T}')
    x_list, y_list = [], []
    for i in range(N):
        x_list.append(X[:, i:i+lookback])
        y_list.append(X[:, i+lookback:i+lookback+horizon])
    x_arr = np.stack(x_list, axis=0)
    y_arr = np.stack(y_list, axis=0)
    return (
        torch.tensor(x_arr, dtype=torch.float32),
        torch.tensor(y_arr, dtype=torch.float32)
    )


def train_val_test_split(
    x_windows: torch.Tensor,
    y_windows: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> tuple:
    """
    Chronological train/val/test split.

    Returns:
        (x_train,y_train), (x_val,y_val), (x_test,y_test)
    """
    N = x_windows.shape[0]
    n_train = int(N * train_ratio)
    n_val   = int(N * (train_ratio + val_ratio)) - n_train
    return (
        (x_windows[:n_train], y_windows[:n_train]),
        (x_windows[n_train:n_train+n_val], y_windows[n_train:n_train+n_val]),
        (x_windows[n_train+n_val:], y_windows[n_train+n_val:])
    )

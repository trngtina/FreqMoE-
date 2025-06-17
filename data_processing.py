import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import yfinance as yf

def fetch_data(tickers, start_date, end_date, interval='1d'):
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
        df = data['Adj Close'].ffill().dropna(axis=0, how='any')
    else:
        df = data['Adj Close'].to_frame().ffill().dropna(axis=0, how='any')
    return df

def compute_log_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(df_prices / df_prices.shift(1)).dropna()

def normalize_series(returns: pd.DataFrame):
    scaler = StandardScaler()
    X = returns.values
    X_norm2d = scaler.fit_transform(X)
    means = scaler.mean_
    stds = np.sqrt(scaler.var_)
    return X_norm2d.T, means, stds

def create_windows(X: np.ndarray, lookback: int, horizon: int):
    C, T = X.shape
    N = T - lookback - horizon + 1
    x_list, y_list = [], []
    for i in range(N):
        x_i = X[:, i : i + lookback]
        y_i = X[:, i + lookback : i + lookback + horizon]
        x_list.append(x_i)
        y_list.append(y_i)
    x_arr = np.stack(x_list, axis=0)
    y_arr = np.stack(y_list, axis=0)
    return torch.tensor(x_arr, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.float32)

def train_val_test_split(x_windows, y_windows, train_ratio=0.7, val_ratio=0.15):
    N = x_windows.shape[0]
    n_train = int(N * train_ratio)
    n_val = int(N * (train_ratio + val_ratio)) - n_train

    x_train = x_windows[:n_train]; y_train = y_windows[:n_train]
    x_val = x_windows[n_train : n_train + n_val]; y_val = y_windows[n_train : n_train + n_val]
    x_test = x_windows[n_train + n_val :]; y_test = y_windows[n_train + n_val :]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

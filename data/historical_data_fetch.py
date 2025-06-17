\"\"\"
historical_data_fetch.py

Utility to fetch historical price data via yfinance.
\"\"\"

import yfinance as yf
import pandas as pd

def fetch_data(tickers, start_date, end_date, interval='1d'):
    \"\"\"
    Fetch adjusted close prices for given tickers.
    Returns a DataFrame indexed by date.
    \"\"\"
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
        df = data['Adj Close'].ffill().dropna(axis=0, how='any')
    else:
        df = data['Adj Close'].to_frame().ffill().dropna(axis=0, how='any')
    return df

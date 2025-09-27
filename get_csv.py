

import os
import finnhub
import pandas as pd
from datetime import datetime, timedelta

def fetch_5yr_prices(
    tickers,
    api_key="d1btmrpr01qsbpuf7h00d1btmrpr01qsbpuf7h0g",
    interval='1d',
    tz='UTC',
    output_csv='finnhub_5yr_prices.csv'
):
    # API key
    key = "d1btmrpr01qsbpuf7h00d1btmrpr01qsbpuf7h0g"
    if not key:
        raise ValueError("Set FINNHUB_API_KEY environment variable or pass api_key")
    client = finnhub.Client(api_key=key)

    # time range: today minus 5 years
    end_dt   = pd.Timestamp.today().tz_localize(tz)
    start_dt = end_dt - pd.DateOffset(years=5)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

    # Finnhub resolution: daily
    resolution = 'D'

    all_frames = []
    for sym in tickers:
        try:
            resp = client.stock_candles(sym, resolution, start_ts, end_ts)
        except Exception as e:
            print(f"❌ {sym}: API error: {e}")
            continue
        if resp.get('s') != 'ok' or 'c' not in resp:
            print(f"❌ {sym}: No data returned (status={resp.get('s')})")
            continue

        df = pd.DataFrame({
            'close': resp['c'],
            't': resp['t']
        })
        # convert epoch → datetime, localize & convert timezone
        df.index = pd.to_datetime(df['t'], unit='s', utc=True).tz_convert(tz)
        df = df[['close']].rename(columns={'close': sym})
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No data could be fetched for any tickers.")

    # merge on timestamps
    df_all = pd.concat(all_frames, axis=1).sort_index()
    # forward‐fill and drop any remaining gaps
    df_all = df_all.ffill().dropna(how='any')

    # save to CSV
    df_all.to_csv(output_csv, index_label='Date')
    print(f"✅ Saved {len(df_all)} rows × {len(tickers)} tickers to {output_csv}")

if __name__ == "__main__":
    # Example usage:
    tickers = ['AAPL', 'MSFT', 'SPY']
    fetch_5yr_prices(
        tickers=tickers,
        api_key=None,               # or put your key here
        interval='1d',
        tz='US/Eastern',
        output_csv='finnhub_5yr_prices.csv'
    )
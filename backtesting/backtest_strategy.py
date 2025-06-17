\"\"\"
backtest_strategy.py

Simple long/short strategy: go long if forecast > 0, short if < 0.
\"\"\"

import pandas as pd

def generate_signals(returns: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    shifted_forecasts = forecasts.shift(1).dropna()
    signals = shifted_forecasts.copy()
    signals[signals > 0] = 1.0
    signals[signals < 0] = -1.0
    signals[signals == 0] = 0.0
    signals = signals.reindex(returns.index).fillna(0.0)
    return signals

def backtest_returns(returns: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
    strat_ret = (signals * returns).mean(axis=1)
    return strat_ret

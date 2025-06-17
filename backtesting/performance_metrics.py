\"\"\"
performance_metrics.py

Compute cumulative returns, Sharpe ratio, max drawdown, hit rate.
\"\"\"

import numpy as np
import pandas as pd
from empyrical import stats

def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    cum_log = returns.cumsum()
    cum_ret = np.exp(cum_log) - 1.0
    return pd.Series(cum_ret, index=returns.index)

def compute_sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    return stats.sharpe_ratio(returns, period=freq, annualization=freq)

def compute_max_drawdown(returns: pd.Series) -> float:
    cum_ret = compute_cumulative_returns(returns)
    rolling_max = cum_ret.cummax()
    drawdown = cum_ret - rolling_max
    return drawdown.min()

def compute_hit_rate(returns: pd.Series) -> float:
    positives = (returns > 0).sum()
    return positives / len(returns)

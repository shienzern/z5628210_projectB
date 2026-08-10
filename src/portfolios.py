"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Build at least a combined equity-plus-crypto fund with two optimisation methods.
Backtest rules: walk-forward, no look-ahead, weights from past data only, annualise
with 252 (equity) or 365 (crypto). See the brief, Part B.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.repeat(1 / n, n)
    bounds = [(0, 1)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    def objective(w):
        return w @ cov @ w

    res = minimize(objective, x0, bounds=bounds, constraints=constraints)
    return res.x


def _max_sharpe_weights(mean_ret: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mean_ret)
    x0 = np.repeat(1 / n, n)
    bounds = [(0, 1)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    def neg_sharpe(w):
        port_ret = w @ mean_ret
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return 0
        return -port_ret / port_vol

    res = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
    return res.x


def _risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.repeat(1 / n, n)
    bounds = [(1e-6, 1)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    def objective(w):
        port_vol = np.sqrt(w @ cov @ w)
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib / port_vol
        target = port_vol / n
        return np.sum((risk_contrib - target) ** 2)

    res = minimize(objective, x0, bounds=bounds, constraints=constraints)
    return res.x


def solve_weights(returns_window: pd.DataFrame, method: str) -> pd.Series:
    """Solve weights from a trailing window of returns (rows=dates, cols=tickers)."""
    cov = returns_window.cov().values
    mean_ret = returns_window.mean().values

    if method == "min_variance":
        w = _min_variance_weights(cov)
    elif method == "max_sharpe":
        w = _max_sharpe_weights(mean_ret, cov)
    elif method == "risk_parity":
        w = _risk_parity_weights(cov)
    else:
        raise ValueError(f"Unknown method: {method}")

    return pd.Series(w, index=returns_window.columns)


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance",
                  window: int = 252, rebalance: str = "MS"):
    """Walk-forward out-of-sample backtest.

    returns: wide frame, date index x ticker columns, daily returns.
    window: trailing days used to estimate weights at each rebalance (no look-ahead).
    rebalance: pandas offset alias, 'MS' = first trading day of month.

    Returns: daily_returns (Series), weights_over_time (DataFrame indexed by
    rebalance date), growth_of_dollar (Series).
    """
    returns = returns.sort_index()
    rebalance_dates = pd.date_range(returns.index.min(), returns.index.max(), freq=rebalance)
    rebalance_dates = [returns.index[returns.index >= d][0]
                        for d in rebalance_dates if (returns.index >= d).any()]
    rebalance_dates = sorted(set(rebalance_dates))

    weights_over_time = {}
    daily_port_returns = []

    for i, reb_date in enumerate(rebalance_dates):
        history = returns.loc[:reb_date].iloc[:-1]  # strictly before reb_date
        if len(history) < window:
            continue
        window_data = history.tail(window)
        w = solve_weights(window_data, method)
        weights_over_time[reb_date] = w

        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index.max()
        period = returns.loc[reb_date:next_reb]
        if reb_date == next_reb:
            continue
        period = period.iloc[:-1] if next_reb != returns.index.max() else period
        port_ret = period[w.index].fillna(0) @ w
        daily_port_returns.append(port_ret)

    daily_returns = pd.concat(daily_port_returns).sort_index()
    daily_returns = daily_returns[~daily_returns.index.duplicated(keep='first')]
    growth_of_dollar = (1 + daily_returns).cumprod()
    weights_df = pd.DataFrame(weights_over_time).T

    return daily_returns, weights_df, growth_of_dollar


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised return, annualised volatility, Sharpe, and max drawdown."""
    ann_return = (1 + daily_returns.mean()) ** periods_per_year - 1
    ann_vol = daily_returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    growth = (1 + daily_returns).cumprod()
    running_max = growth.cummax()
    drawdown = (growth - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'annualised_return': ann_return,
        'annualised_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
    }

"""Station 3 (extension) - fuse sentiment into the funds.

Tilt or factor: combine your sentiment signal with the portfolio weights,
look-ahead safe, then test whether it adds value. An honest negative result,
explained, is good work.
"""
import pandas as pd


def ticker_sentiment_lagged(ticker_scores: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Lag per-ticker-day sentiment scores by lag_days trading days.

    Ticker-days with no headline are treated as neutral (0), consistent with
    the sector index. Mirrors sentiment.sector_sentiment_index but at ticker
    granularity, since the fusion tilt is applied per-stock.
    """
    wide = ticker_scores.pivot(index='trading_date', columns='ticker', values='sentiment')
    wide = wide.sort_index().fillna(0.0)
    lagged = wide.shift(lag_days)
    return lagged


def apply_sentiment(weights: pd.DataFrame, sentiment_lagged: pd.DataFrame,
                     strength: float = 0.5) -> pd.DataFrame:
    """Tilt fund weights by lagged ticker sentiment, look-ahead safe.

    weights: DataFrame indexed by rebalance_date, columns=tickers (e.g. from
        portfolios.oos_backtest).
    sentiment_lagged: DataFrame indexed by trading_date, columns=tickers,
        already lagged (e.g. from ticker_sentiment_lagged).
    strength: how strongly sentiment scales weight. Each stock's weight is
        multiplied by (1 + strength * sentiment), then the row is renormalised
        to sum to 1 (long-only assumed, so weights are clipped at 0 first).

    For each rebalance date, uses the most recent available lagged sentiment
    on or before that date - never a future value.
    """
    tilted = weights.copy()

    for reb_date in weights.index:
        available = sentiment_lagged.loc[:reb_date]
        if available.empty:
            continue
        latest_sentiment = available.iloc[-1]

        row = weights.loc[reb_date]
        common = row.index.intersection(latest_sentiment.index)
        scale = 1 + strength * latest_sentiment.reindex(row.index).fillna(0.0)
        scale = scale.clip(lower=0.1)  # avoid zeroing out / flipping sign

        new_row = (row * scale).clip(lower=0)
        if new_row.sum() > 0:
            new_row = new_row / new_row.sum()
        tilted.loc[reb_date] = new_row

    return tilted


def backtest_from_weights(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Compute daily portfolio returns given a weights-over-time DataFrame.

    Mirrors the return-calculation logic inside portfolios.oos_backtest, but
    takes pre-computed (e.g. sentiment-tilted) weights instead of solving them.
    Holds each rebalance's weights until the next rebalance date.
    """
    rebalance_dates = sorted(weights.index)
    daily_returns = []

    for i, reb_date in enumerate(rebalance_dates):
        w = weights.loc[reb_date]
        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index.max()
        period = returns.loc[reb_date:next_reb]
        if reb_date == next_reb:
            continue
        period = period.iloc[:-1] if next_reb != returns.index.max() else period
        port_ret = period[w.index].fillna(0) @ w
        daily_returns.append(port_ret)

    result = pd.concat(daily_returns).sort_index()
    return result[~result.index.duplicated(keep='first')]

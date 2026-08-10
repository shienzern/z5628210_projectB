"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.
"""
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply VADER to the assembled headlines panel.

    Expects panel to have 'title', 'ticker', 'sector', 'trading_date' columns
    (the output of features.assemble_headline_panel). Scores each headline's
    raw title text (VADER needs casing/punctuation/negation intact), then
    averages to one compound score per ticker-day (a ticker can have several
    headlines on the same trading day).
    """
    sia = _get_analyzer()
    df = panel.copy()
    df['compound'] = df['title'].fillna('').apply(lambda t: sia.polarity_scores(t)['compound'])

    ticker_day = (
        df.groupby(['trading_date', 'ticker', 'sector'])['compound']
        .mean()
        .reset_index()
        .rename(columns={'compound': 'sentiment'})
    )
    return ticker_day


def sector_sentiment_index(scores: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Build a daily sentiment index per sector (equal-weight across tickers).

    Ticker-days with no headlines are treated as neutral (0) when building the
    sector average, rather than dropped or forward-filled - a missing headline
    is not evidence of positive or negative sentiment, just silence.

    The index is lagged by `lag_days` trading days before being usable, so a
    signal computed from day t's headlines is only available from day t+lag_days
    onward - this avoids look-ahead when the index is later fused into a fund.
    """
    sector_daily = (
        scores.groupby(['trading_date', 'sector'])['sentiment']
        .mean()
        .reset_index()
    )

    wide = sector_daily.pivot(index='trading_date', columns='sector', values='sentiment')
    wide = wide.sort_index().fillna(0.0)  # ticker-days with no headlines -> neutral

    lagged = wide.shift(lag_days)

    lagged = lagged.reset_index().melt(
        id_vars='trading_date', var_name='sector', value_name='sentiment_lagged'
    )
    return lagged.dropna(subset=['sentiment_lagged']).reset_index(drop=True)

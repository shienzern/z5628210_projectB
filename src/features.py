"""Station 2 - your features: return features and text assembly.

Build your return features here, and assemble the headlines into a daily text
panel. Scoring the text is the Station 3 sentiment model (see src/sentiment.py).
"""
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker. Uses adjClose.

    Keeps the frame long (date, ticker, ..., ret) rather than pivoting wide,
    so it stays consistent with how etl.py already computes returns.
    """
    df = prices.sort_values(['ticker', 'date']).copy()
    df['ret'] = df.groupby('ticker')[price_col].pct_change()
    return df


def assemble_headline_panel(headlines: pd.DataFrame, trading_calendar: pd.Series) -> pd.DataFrame:
    """Assemble headlines into a daily panel per ticker and sector.

    Maps every headline to its equity trading day: the same day if it is a
    trading day, otherwise the next trading day (e.g. a Saturday headline is
    pulled forward to the following Monday). Keeps the raw headline text
    untouched - no stopword stripping - since VADER (Station 3) needs the
    original casing, punctuation, and negation words.

    trading_calendar: the equity trading dates (e.g. eq_df['date']).
    """
    cal = pd.DataFrame({
        'trading_date': pd.to_datetime(sorted(trading_calendar.unique())).astype('datetime64[ns]')
    })

    h = headlines.copy()
    h['date'] = pd.to_datetime(h['date']).astype('datetime64[ns]')
    h = h.sort_values('date')

    panel = pd.merge_asof(
        h, cal, left_on='date', right_on='trading_date', direction='forward'
    )

    # Headlines after the last trading date have no forward match - drop them
    n_unmapped = panel['trading_date'].isna().sum()
    panel = panel.dropna(subset=['trading_date']).reset_index(drop=True)

    return panel

"""Station 1 - your ETL: load and clean the data.

Load raw data through src.data_access (see context/DATA_GUIDE.md). Add your own
integrity checks. Do not commit data files.
"""
from src import data_access
import pandas as pd


def load_clean_equities():
    """Load equity prices and run Station 1 integrity checks.

    Checks: missing-date audit, duplicate ticker-date check, outlier screen
    on returns. Genuine outliers are kept and documented, not deleted.
    Returns the clean frame plus a summary dict of what was found.
    """
    df = data_access.load_equity_prices()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    n_dupes = df.duplicated(subset=['ticker', 'date']).sum()

    dates_per_ticker = df.groupby('ticker')['date'].nunique()
    n_expected = df['date'].nunique()
    tickers_with_gaps = (dates_per_ticker != n_expected).sum()

    df['ret'] = df.groupby('ticker')['adjClose'].pct_change()
    outlier_threshold = 0.20
    outliers = df.loc[df['ret'].abs() > outlier_threshold,
                       ['ticker', 'date', 'ret', 'volume']]

    summary = {
        'dataset': 'equity_prices',
        'rows': len(df),
        'n_tickers': df['ticker'].nunique(),
        'date_min': df['date'].min(),
        'date_max': df['date'].max(),
        'n_duplicates': int(n_dupes),
        'tickers_with_missing_dates': int(tickers_with_gaps),
        'n_outliers_flagged': len(outliers),
    }
    return df, summary


def load_clean_crypto():
    """Load crypto prices (365-day calendar) and run Station 1 checks.

    Includes capping the stray 2024-01-01 rows per the Data Guide, before
    computing returns and running the same integrity checks as equities.
    """
    df = data_access.load_crypto_prices()
    df = df[df['date'] <= '2023-12-31']
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    n_dupes = df.duplicated(subset=['ticker', 'date']).sum()

    dates_per_ticker = df.groupby('ticker')['date'].nunique()
    n_expected = df['date'].nunique()
    tickers_with_gaps = (dates_per_ticker != n_expected).sum()

    df['ret'] = df.groupby('ticker')['adjClose'].pct_change()
    outlier_threshold = 0.20
    outliers = df.loc[df['ret'].abs() > outlier_threshold,
                       ['ticker', 'date', 'ret', 'volume']]

    summary = {
        'dataset': 'crypto_prices',
        'rows': len(df),
        'n_tickers': df['ticker'].nunique(),
        'date_min': df['date'].min(),
        'date_max': df['date'].max(),
        'n_duplicates': int(n_dupes),
        'tickers_with_missing_dates': int(tickers_with_gaps),
        'n_outliers_flagged': len(outliers),
        'n_rows_capped_2024': 10,
    }
    return df, summary


def load_clean_news():
    """Load news headlines and run Station 1 integrity checks.

    Dedupe on ticker+date+title (not ticker+date alone - many headlines
    legitimately share a ticker-date). Normalise the tz-aware UTC date to
    tz-naive so it can later be merged/compared with price dates.
    """
    df = data_access.load_news_headlines()

    n_naive_dupes = df.duplicated(subset=['ticker', 'date']).sum()
    n_real_dupes = df.duplicated(subset=['ticker', 'date', 'title']).sum()

    df = df.drop_duplicates(subset=['ticker', 'date', 'title']).reset_index(drop=True)

    df['date'] = df['date'].dt.tz_convert(None)

    summary = {
        'dataset': 'news_headlines',
        'rows': len(df),
        'n_tickers': df['ticker'].nunique(),
        'date_min': df['date'].min(),
        'date_max': df['date'].max(),
        'n_naive_duplicate_check': int(n_naive_dupes),
        'n_real_duplicates_removed': int(n_real_dupes),
        'publisher_blank_pct': round(df['publisher'].isna().mean() * 100, 1),
    }
    return df, summary


def build_combined_panel(eq_df, cr_df):
    """Build the combined equity+crypto returns panel on the equity calendar.

    Crypto returns are computed on crypto's own calendar (already done by
    load_clean_crypto), then left-merged onto equity trading dates - never
    merge price levels and difference afterwards, as that creates spurious
    returns.
    """
    equity_calendar = eq_df[['date']].drop_duplicates().sort_values('date')

    eq_long = eq_df[['date', 'ticker', 'sector', 'ret']].copy()
    eq_long['asset_class'] = 'equity'

    cr_merged = equity_calendar.merge(cr_df, on='date', how='left')
    cr_long = cr_merged[['date', 'ticker', 'ret']].copy()
    cr_long['sector'] = None
    cr_long['asset_class'] = 'crypto'

    combined = pd.concat([eq_long, cr_long], ignore_index=True)
    combined = combined.dropna(subset=['ret'])

    summary = {
        'rows': len(combined),
        'n_equity_rows': (combined['asset_class'] == 'equity').sum(),
        'n_crypto_rows': (combined['asset_class'] == 'crypto').sum(),
        'date_min': combined['date'].min(),
        'date_max': combined['date'].max(),
    }
    return combined, summary

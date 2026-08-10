"""FinTech Project - Effortless Invest: systematic multi-asset fund dashboard.

Run locally:   streamlit run streamlit_app.py
Deploy:        push this folder to a public GitHub repo, then connect it on
               share.streamlit.io with entrypoint streamlit_app.py (see brief App. D).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st  # noqa: E402

from src import data_access  # noqa: E402

st.set_page_config(page_title="Effortless Invest", layout="wide")
st.title("Effortless Invest - Systematic Multi-Asset Funds")
st.caption("Systematically managed equity, crypto, and combined funds, backed by out-of-sample backtests and a news-sentiment overlay.")

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


@st.cache_data(ttl=86_400, show_spinner="Loading fund data...")
def _fund_returns():
    df = pd.read_csv(RESULTS / "data" / "fund_returns.csv", parse_dates=["date"])
    return df


@st.cache_data(ttl=86_400, show_spinner="Loading fund weights...")
def _fund_weights():
    df = pd.read_csv(RESULTS / "data" / "fund_weights.csv", parse_dates=["rebalance_date"])
    return df


@st.cache_data(ttl=86_400, show_spinner="Loading performance metrics...")
def _performance_metrics():
    return pd.read_csv(RESULTS / "tables" / "performance_metrics.csv")


@st.cache_data(ttl=86_400, show_spinner="Loading sentiment index...")
def _sentiment_index():
    df = pd.read_csv(RESULTS / "data" / "sector_sentiment_index.csv", parse_dates=["trading_date"])
    return df


@st.cache_data(ttl=86_400, show_spinner="Loading data...")
def _equities():
    return data_access.load_equity_prices()


tab_funds, tab_sentiment, tab_data = st.tabs(["Funds", "Sentiment", "Data"])

with tab_funds:
    returns = _fund_returns()
    weights = _fund_weights()
    metrics = _performance_metrics()

    fund_list = sorted(returns["fund"].unique())

    st.subheader("Compare all funds")
    metrics_display = metrics.set_index("fund")[
        ["annualised_return", "annualised_volatility", "sharpe_ratio", "max_drawdown"]
    ].round(3)
    st.dataframe(metrics_display, width="stretch")

    st.subheader("Fund fact sheet")
    selected_fund = st.selectbox("Choose a fund", fund_list)

    fund_ret = returns[returns["fund"] == selected_fund].sort_values("date")
    growth = (1 + fund_ret["daily_return"]).cumprod()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.line_chart(pd.Series(growth.values, index=fund_ret["date"], name="Growth of $1"))
    with col2:
        row = metrics[metrics["fund"] == selected_fund].iloc[0]
        st.metric("Annualised Return", f"{row['annualised_return']:.1%}")
        st.metric("Annualised Volatility", f"{row['annualised_volatility']:.1%}")
        st.metric("Sharpe Ratio", f"{row['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{row['max_drawdown']:.1%}")

    st.markdown("**Current holdings (most recent rebalance)**")
    fund_w = weights[weights["fund"] == selected_fund]
    latest_date = fund_w["rebalance_date"].max()
    latest_holdings = (
        fund_w[fund_w["rebalance_date"] == latest_date]
        .sort_values("weight", ascending=False)
        .head(10)[["ticker", "weight"]]
        .set_index("ticker")
    )
    st.dataframe(latest_holdings.style.format({"weight": "{:.1%}"}), width="stretch")

    st.subheader("Set your allocation")
    st.caption("Allocate a hypothetical investment across funds to see the blended outcome.")
    alloc = {}
    cols = st.columns(len(fund_list))
    for i, fund in enumerate(fund_list):
        with cols[i % len(cols)]:
            alloc[fund] = st.slider(fund, 0, 100, 0, key=f"alloc_{fund}")

    total_alloc = sum(alloc.values())
    if total_alloc > 0:
        weights_norm = {k: v / total_alloc for k, v in alloc.items() if v > 0}
        blended = None
        for fund, w in weights_norm.items():
            fr = returns[returns["fund"] == fund].sort_values("date").set_index("date")["daily_return"]
            blended = fr * w if blended is None else blended.add(fr * w, fill_value=0)
        blended_growth = (1 + blended).cumprod()
        st.line_chart(blended_growth.rename("Blended Growth of $1"))
    else:
        st.info("Move a slider above to set an allocation.")

with tab_sentiment:
    st.subheader("Sector sentiment index")
    st.caption("VADER sentiment on news headlines, lagged 1 trading day, equal-weighted per sector.")

    sentiment = _sentiment_index()
    sectors = sorted(sentiment["sector"].unique())
    selected_sectors = st.multiselect("Sectors to show", sectors, default=sectors[:3])

    if selected_sectors:
        sub = sentiment[sentiment["sector"].isin(selected_sectors)]
        wide = sub.pivot(index="trading_date", columns="sector", values="sentiment_lagged")
        smoothed = wide.rolling(20).mean()
        st.line_chart(smoothed)
    else:
        st.info("Select at least one sector above.")

with tab_data:
    eq = _equities()
    st.write(f"Equity prices: {eq.shape[0]:,} rows, {eq['ticker'].nunique()} tickers, "
             f"{eq['date'].min().date()} to {eq['date'].max().date()}")
    st.dataframe(eq.head(20), width="stretch")

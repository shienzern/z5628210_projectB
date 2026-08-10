"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from src import etl, features, portfolios, sentiment, fusion  # noqa: E402

RESULTS_DATA = pathlib.Path(__file__).resolve().parent.parent / "results" / "data"
RESULTS_TABLES = pathlib.Path(__file__).resolve().parent.parent / "results" / "tables"
METHODS = ["min_variance", "max_sharpe", "risk_parity"]
FUSION_STRENGTHS = [0.5, 2.0]


def build_fund(name: str, wide_returns: pd.DataFrame, method: str, periods_per_year: int):
    daily_ret, weights, growth = portfolios.oos_backtest(wide_returns, method=method)
    metrics = portfolios.performance_metrics(daily_ret, periods_per_year=periods_per_year)
    metrics["fund"] = name

    ret_df = daily_ret.rename("daily_return").reset_index()
    ret_df.columns = ["date", "daily_return"]
    ret_df["fund"] = name

    w_df = weights.reset_index().melt(id_vars="index", var_name="ticker", value_name="weight")
    w_df.columns = ["rebalance_date", "ticker", "weight"]
    w_df["fund"] = name

    return ret_df, w_df, metrics, daily_ret, weights


def main():
    print("Loading and cleaning data...")
    eq, _ = etl.load_clean_equities()
    cr, _ = etl.load_clean_crypto()
    news, _ = etl.load_clean_news()
    combined, _ = etl.build_combined_panel(eq, cr)

    combined_wide = combined.pivot_table(index="date", columns="ticker", values="ret")
    equity_wide = features.daily_returns(eq).pivot_table(index="date", columns="ticker", values="ret")
    crypto_wide = features.daily_returns(cr).pivot_table(index="date", columns="ticker", values="ret")

    fund_families = {
        "combined": (combined_wide, 252),
        "equity": (equity_wide, 252),
        "crypto": (crypto_wide, 365),
    }

    all_returns, all_weights, all_metrics = [], [], []
    saved_funds = {}  # keep equity funds' raw outputs for the fusion step

    for family, (wide, ppy) in fund_families.items():
        for method in METHODS:
            fund_name = f"{family}_{method}"
            print(f"Backtesting {fund_name}...")
            ret_df, w_df, metrics, daily_ret, weights = build_fund(fund_name, wide, method, ppy)
            all_returns.append(ret_df)
            all_weights.append(w_df)
            all_metrics.append(metrics)
            if family == "equity":
                saved_funds[fund_name] = (daily_ret, weights)

    fund_returns = pd.concat(all_returns, ignore_index=True)
    fund_weights = pd.concat(all_weights, ignore_index=True)
    performance_metrics = pd.DataFrame(all_metrics)

    print("Assembling headline panel and scoring sentiment...")
    panel = features.assemble_headline_panel(news, eq["date"])
    ticker_scores = sentiment.score_headlines(panel)
    sector_index = sentiment.sector_sentiment_index(ticker_scores)
    ticker_lagged = fusion.ticker_sentiment_lagged(ticker_scores)

    print("Running sentiment fusion on equity_min_variance...")
    base_name = "equity_min_variance"
    base_daily, base_weights = saved_funds[base_name]
    base_metrics = portfolios.performance_metrics(base_daily)

    fusion_rows = [{"variant": f"base_{base_name}", **base_metrics}]
    for strength in FUSION_STRENGTHS:
        tilted_weights = fusion.apply_sentiment(base_weights, ticker_lagged, strength=strength)
        tilted_daily = fusion.backtest_from_weights(equity_wide, tilted_weights)
        tilted_metrics = portfolios.performance_metrics(tilted_daily)
        fusion_rows.append({"variant": f"sentiment_tilt_{strength}", **tilted_metrics})
    fusion_comparison = pd.DataFrame(fusion_rows)

    RESULTS_DATA.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)

    fund_returns.to_csv(RESULTS_DATA / "fund_returns.csv", index=False)
    fund_weights.to_csv(RESULTS_DATA / "fund_weights.csv", index=False)
    performance_metrics.to_csv(RESULTS_TABLES / "performance_metrics.csv", index=False)
    sector_index.to_csv(RESULTS_DATA / "sector_sentiment_index.csv", index=False)
    fusion_comparison.to_csv(RESULTS_TABLES / "fusion_comparison.csv", index=False)

    print("\nSaved:")
    print(" -", RESULTS_DATA / "fund_returns.csv", fund_returns.shape)
    print(" -", RESULTS_DATA / "fund_weights.csv", fund_weights.shape)
    print(" -", RESULTS_TABLES / "performance_metrics.csv", performance_metrics.shape)
    print(" -", RESULTS_DATA / "sector_sentiment_index.csv", sector_index.shape)
    print(" -", RESULTS_TABLES / "fusion_comparison.csv", fusion_comparison.shape)

    print("\nPerformance summary:")
    print(performance_metrics.set_index("fund").round(3))
    print("\nFusion comparison:")
    print(fusion_comparison.set_index("variant").round(4))

    # TODO: save figures under results/figures/ for the report


if __name__ == "__main__":
    main()

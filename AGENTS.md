# AGENTS.md — Effortless Invest, Part B

## Project
Effortless Invest — a systematic multi-asset investment app (equities, crypto,
combined) with a news-sentiment overlay. Built for FINS3645 Part B. Full spec:
PROJECT_BRIEF.md. Data guide: context/DATA_GUIDE.md. Builds directly on the
Part A data foundation (z5628210_projectA).

## Folder conventions
- src/ — reusable code (etl.py, features.py, portfolios.py, sentiment.py, fusion.py)
- scripts/ — runnable entry points (run_part_b.py, check_handin.py)
- results/data/, results/tables/, results/figures/ — committed outputs only,
  never raw parquet
- streamlit_app.py — reads only precomputed results/, never recomputes
  backtests or scores sentiment live

## Hard rules (enforced throughout)
- No look-ahead bias: backtest weights use only data strictly before each
  rebalance date (current day's return excluded from the estimation window).
- Sentiment is lagged at least 1 trading day before use in the sector index
  or the fusion tilt.
- Equity calendar = 252 days/yr, crypto-only funds = 365 days/yr; the combined
  fund uses 252 because its panel is left-merged onto the equity calendar.
- Never commit raw .parquet or secrets — only derived artifacts in results/.
- Required output filenames match exactly: fund_returns.csv, fund_weights.csv,
  sector_sentiment_index.csv, performance_metrics.csv.

## How I used AI this session
- I asked the assistant to write and debug code (ETL, portfolio optimisation,
  sentiment scoring, fusion, the Streamlit app) — I ran every function myself
  in the PyCharm console and checked the output before moving on.
- Every non-trivial number in my report was independently verified by running
  code myself, not taken from the assistant's estimate (e.g. drawdown date/
  value, exact weight percentages, growth-of-$1 final values, sentiment
  distribution stats).
- I caught and corrected a real analytical error myself mid-session: an
  initial claim that "combined funds beat equity-only in two of three
  methods" was wrong — checking the actual numbers showed only max-Sharpe
  showed a clear combined benefit; min-variance and risk parity favoured
  equity-only. I rewrote the finding to match the real data.
- I write all report analysis and interpretation myself. The assistant
  reviews my drafts for factual accuracy (checked against my own code output)
  and grammar, and gives structured outlines of what each section needs to
  cover — it does not draft report prose for me.
- Prompt log kept in ai/prompt_log.md.

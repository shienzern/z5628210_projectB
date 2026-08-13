# Prompt log — Effortless Invest Part B

## Entry 1: Building the portfolio backtest engine

### What I wanted
A walk-forward, out-of-sample backtest function supporting three optimisation
methods (min-variance, max-Sharpe, risk parity) with no look-ahead bias.

### Prompt(s)
Asked the assistant to design and implement `oos_backtest()` in src/portfolios.py,
using a 252-day trailing estimation window, monthly rebalancing, and long-only
constraints (weights sum to 1, no shorting).

### What the assistant produced
A full implementation with separate weight-solving functions for each method,
plus a walk-forward loop that iterates over rebalance dates.

### What was wrong or risky
Needed to verify: (1) the estimation window genuinely excludes the current day's
return, (2) weights actually differ meaningfully across methods rather than
silently converging to equal-weight (a known optimiser-stall risk flagged in the
brief), (3) crypto vs equity annualisation was applied correctly.

### What I changed and why
Ran the backtest for all three methods myself in the PyCharm console and checked
the resulting weights by hand — confirmed min-variance concentrated in
low-volatility large-caps (not equal-weight), and max-Sharpe/risk-parity produced
visibly different, sensible risk/return profiles (Sharpe 0.51 / 0.83 / 1.18).
This confirmed the optimiser wasn't stalling.

---

## Entry 2: Sentiment scoring and the sector index

### What I wanted
Score ~147k news headlines with VADER, aggregate to a sector-day index, lagged
to prevent look-ahead.

### Prompt(s)
Asked for score_headlines() and sector_sentiment_index() in src/sentiment.py.

### What the assistant produced
Functions using plain VADER, averaging per-headline scores to ticker-day then
sector-day, with a configurable lag.

### What was wrong or risky
Needed to confirm the neutral-treatment choice (missing headline-days = neutral
0.0, not dropped/forward-filled) was actually implemented as intended, and that
the lag genuinely prevented look-ahead — including the two-step case of a
weekend headline (mapped to Monday, then lagged again to Tuesday).

### What I changed and why
Traced a real weekend headline through the pipeline myself in the console
(a Saturday 2020-01-04 DIS headline → mapped to Monday 2020-01-06 → confirmed
usable only from Tuesday 2020-01-07). This caught that my draft report text had
this wrong (I'd written "usable Monday," not Tuesday) before it went in the report.

---

## Entry 3: Sentiment fusion tilt

### What I wanted
A weight-tilting mechanism that scales fund weights by lagged ticker sentiment,
tested at multiple strengths.

### Prompt(s)
Asked for apply_sentiment() in src/fusion.py, tilting weights by
(1 + strength × sentiment), floored to avoid negative/zero weights.

### What was wrong or risky
Needed to verify the tilt was genuinely look-ahead-safe (reusing the same lagged
signal from Section 3, not a fresh unlagged one), and needed to independently
verify the actual percentage effect of different strength values rather than
trusting a stated example.

### What I changed and why
Worked through the arithmetic myself: at strength 0.5 and sentiment 0.2, the
scale factor is 1 + (0.5 × 0.2) = 1.1, a 10% boost — not the 40% I'd
initially misstated in an early report draft. Caught and corrected this before
finalising Section 4.

---

## Entry 4: Report fact-checking

### What I wanted
An accurate, evidence-based report where every specific number/claim was
checked against real output, not estimated.

### What I changed and why
Multiple report claims were checked and corrected against actual code output
rather than accepted as written, including: the diversification-benefit paragraph
(an early draft claimed combined funds beat equity-only in two of three methods;
checking the real numbers showed this was wrong — only max-Sharpe showed a clear
benefit), the weights-over-time "Other" percentage at different dates (an early
draft had the direction backwards for one period), and the exact drawdown
date/value (verified as -26.28% on 2022-09-30, not just "around September 2022").

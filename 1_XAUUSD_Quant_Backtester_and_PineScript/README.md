# XAUUSD Algorithmic Strategy & Quantitative Backtesting Engine

Built by **Vishal Pandey** (B.Tech CSE)

## Overview
A rule-based algorithmic trading indicator (TradingView Pine Script v5) and Python quantitative backtesting suite designed for **XAUUSD (Gold)**. Combines **3-minute Fair Value Gaps (FVG)**, institutional Order Blocks, and **London/New York session filters** with strict **1% fixed account risk** (1:1.5R to 1:2.5R payoff).

## Files Included
* `xauusd_fvg_session_strategy.pine` — TradingView Pine Script v5 indicator for live FVG & session filtering.
* `quant_backtester.py` — Python quantitative engine calculating Win-Rate Expectancy, Equity Curve, Maximum Drawdown, and Annualized Sharpe Ratio.
* `xauusd_backtest_log.csv` — 120-trade historical simulation dataset on a $10,000 prop-firm evaluation account.

## Quick Start
```bash
pip install pandas numpy
python quant_backtester.py
```

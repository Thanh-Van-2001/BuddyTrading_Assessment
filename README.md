# BuddyTrading - Quant Trader Assessment

## Strategy: Trend Following with Bollinger Bands + EMA Filter

**Instrument:** BTC/USD Spot | **Exchange:** Coinbase | **Timeframe:** 60-min | **Direction:** Long-only

### Overview

A systematic trend-following strategy that enters long when BTC breaks above the upper Bollinger Band while remaining above a long-term EMA. Designed to capture strong directional moves while avoiding mean-reversion traps.

### Key Results (Jan 2023 - Nov 2025)

| Metric | Value |
|--------|-------|
| Total Return | +160.0% |
| Annualized Return | +39.4% |
| Sharpe Ratio | 1.78 |
| Max Drawdown | -12.3% |
| Win Rate | 44.8% |
| Profit Factor | 2.36 |
| Total Trades | 67 |
| Trading Fees | 0.16% RT |

### Files

| File | Description |
|------|-------------|
| `BuddyTrading_Assessment_TrendFollowingBBEMA.pdf` | Main assessment document (3 pages) |
| `backtest_engine.py` | Core backtesting engine with all strategy classes |
| `performance_analysis.py` | Trade & portfolio performance analysis |
| `optuna_optimization.py` | Optuna parameter optimization (150 trials) |
| `walk_forward_analysis.py` | Monthly walk-forward out-of-sample test |
| `long_short_test.py` | Long vs Short vs Long/Short comparison |
| `export_charts.py` | Equity curve, drawdown & monthly returns charts |
| `generate_pdf.py` | PDF report generator |
| `general/` | Helper modules (date, finance, plotting utilities) |
| `results/` | Backtest outputs: CSVs, charts, optimization results |

### How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib ta optuna fpdf2 pandas_market_calendars

# Run backtest (requires BTC/USD 60m candle data CSV)
python backtest_engine.py

# Run performance analysis
python performance_analysis.py

# Run Optuna optimization
python optuna_optimization.py

# Run walk-forward analysis
python walk_forward_analysis.py

# Generate charts
python export_charts.py

# Generate PDF report
python generate_pdf.py
```

### Data

Backtest uses Coinbase BTC/USD spot OHLCV candle data (60-minute bars). Data not included in this repository due to size constraints.

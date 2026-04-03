# BuddyTrading - Quant Trader Assessment

## Strategy: Trend Following with Bollinger Bands + EMA Filter

**Instrument:** BTC/USD Perpetual Futures | **Exchange:** Coinbase | **Timeframe:** 60-min | **Direction:** Long-only | **Leverage:** 2x

### Overview

A systematic trend-following strategy that enters long when BTC breaks above the upper Bollinger Band while remaining above a long-term EMA. Designed to capture strong directional moves while avoiding mean-reversion traps.

**Entry:** Close > BB_Upper(75, 3.0) AND Close > EMA(1000)
**Exit:** Close < BB_Middle(75) OR Close < EMA(1000)

### Key Results (Jan 2023 - Nov 2025, 2x Leverage)

| Metric | Value |
|--------|-------|
| Total Return | +422.0% |
| Annualized Return | +77.7% |
| Sharpe Ratio | 1.82 |
| Max Drawdown | -24.6% |
| Win Rate | 44.8% |
| Profit Factor | 2.12 |
| Total Trades | 67 |
| Avg Win / Avg Loss | 3.37x |
| Trading Fees | 0.16% RT |
| Funding Rate | 0.01% / 8h |

### Walk-Forward Out-of-Sample (Jan 2024 - Nov 2025)

| Metric | Value |
|--------|-------|
| Total OOS Return | +38.2% |
| Sharpe | 0.56 |
| Max DD | -31.9% |

### Files

| File | Description |
|------|-------------|
| `BuddyTrading_Assessment_TrendFollowingBBEMA.pdf` | Main assessment document (3 pages) |
| `backtest_engine.py` | Core backtesting engine + TrendFollowingBBEMA strategy |
| `run_2x_leverage.py` | Full 2x leveraged backtest with charts and regime analysis |
| `optuna_optimization.py` | Optuna parameter optimization (150 trials) |
| `walk_forward_analysis.py` | Monthly walk-forward out-of-sample test |
| `leverage_test.py` | Leverage sensitivity analysis (1x - 10x) |
| `general/` | Helper modules (date, finance, plotting utilities) |
| `data/` | Coinbase BTC/USD 60-min candle data |
| `results/` | Backtest outputs: charts, trade/portfolio CSVs, optimization results |

### How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib ta optuna pandas_market_calendars

# Run main backtest (2x leverage)
python run_2x_leverage.py

# Run base backtest (1x, no leverage)
python backtest_engine.py

# Run Optuna optimization
python optuna_optimization.py

# Run walk-forward analysis
python walk_forward_analysis.py

# Run leverage comparison (1x - 10x)
python leverage_test.py
```

### Data

Coinbase BTC/USD OHLCV candle data (60-minute bars), included in `data/btcusd_60m.csv`.

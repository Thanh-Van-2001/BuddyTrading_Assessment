# -*- coding: utf-8 -*-
"""Export equity curve + drawdown chart for TrendFollowingBBEMA BTC 60m"""
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest_engine import TrendFollowingBBEMA

PRICE_DATA_DIR = 'D:/coinlion_data/coinlion/candle data/'

params = {
    'start_date': '2023-01-01', 'asset': 'BTC', 'timeframe': '60m',
    'datafile_path': PRICE_DATA_DIR + 'btcusd_60m.csv',
    'datafile_extra_path': None, 'max_no_of_trades': 1, 'initial_nav': 100,
    'tp_pct': 1.0, 'sl_pct': 0.15, 'max_holding': 1000000, 'tcost': 0.0016,
    'ema_period': 1000, 'bb_window': 75, 'bb_dev': 3,
}

strat = TrendFollowingBBEMA(params)
strat.run_backtest_fast()

# Daily NAV
nav = strat.port_history['nav'].copy()
nav.index = pd.to_datetime(nav.index)
nav_daily = nav.groupby(nav.index.date).last()
nav_daily = pd.Series(nav_daily.values, index=pd.to_datetime(nav_daily.index))

# Drawdown
cummax = nav_daily.cummax()
dd_pct = (nav_daily - cummax) / cummax * 100

# --- Figure 1: Equity Curve ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

axes[0].plot(nav_daily.index, nav_daily.values, color='#2196F3', linewidth=1.5)
axes[0].fill_between(nav_daily.index, 100, nav_daily.values, alpha=0.1, color='#2196F3')
axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('TrendFollowingBBEMA — BTC/USDT 60m — Equity Curve\n(tcost=0.16%, SL=15%, no TP cap)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('NAV ($)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Annotate final NAV
final_nav = nav_daily.iloc[-1]
axes[0].annotate(f'Final: ${final_nav:.1f}', xy=(nav_daily.index[-1], final_nav),
                 fontsize=10, fontweight='bold', color='#2196F3',
                 xytext=(-80, 15), textcoords='offset points')

# --- Figure 2: Drawdown ---
axes[1].fill_between(dd_pct.index, 0, dd_pct.values, color='#F44336', alpha=0.4)
axes[1].plot(dd_pct.index, dd_pct.values, color='#F44336', linewidth=0.8)
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
axes[1].set_ylabel('Drawdown (%)', fontsize=11)
axes[1].set_xlabel('Date', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Annotate max DD
max_dd_val = dd_pct.min()
max_dd_date = dd_pct.idxmin()
axes[1].annotate(f'Max DD: {max_dd_val:.1f}%', xy=(max_dd_date, max_dd_val),
                 fontsize=10, fontweight='bold', color='#F44336',
                 xytext=(30, -15), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='#F44336'))

plt.tight_layout()
fig.savefig('backtest_results_coinlion_data/BBEMA_BTC_60m_equity_drawdown.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: backtest_results_coinlion_data/BBEMA_BTC_60m_equity_drawdown.png")

# --- Figure 2: Monthly Returns Heatmap ---
nav_monthly = nav_daily.resample('M').last()
monthly_ret = nav_monthly.pct_change().dropna() * 100

# Build heatmap data
years = sorted(monthly_ret.index.year.unique())
months = list(range(1, 13))
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

heatmap_data = pd.DataFrame(index=years, columns=months, dtype=float)
for dt, ret in monthly_ret.items():
    heatmap_data.loc[dt.year, dt.month] = ret

fig2, ax2 = plt.subplots(figsize=(14, 4))
data = heatmap_data.values.astype(float)
im = ax2.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=30)

ax2.set_xticks(range(12))
ax2.set_xticklabels(month_labels, fontsize=10)
ax2.set_yticks(range(len(years)))
ax2.set_yticklabels(years, fontsize=10)

for i in range(len(years)):
    for j in range(12):
        val = data[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 15 else 'black'
            ax2.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=9, color=color)

ax2.set_title('TrendFollowingBBEMA — BTC/USDT 60m — Monthly Returns (%)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Return %', shrink=0.8)
plt.tight_layout()
fig2.savefig('backtest_results_coinlion_data/BBEMA_BTC_60m_monthly_returns.png', dpi=200, bbox_inches='tight')
plt.close(fig2)
print("Saved: backtest_results_coinlion_data/BBEMA_BTC_60m_monthly_returns.png")

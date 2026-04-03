# -*- coding: utf-8 -*-
"""
Full analysis for TrendFollowingBBEMA BTC/USD Perp Futures 60m, 2x leverage
Generates: charts, stats, regime analysis
"""
import warnings, math
import numpy as np
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

PRICE_DATA_DIR = 'D:/coinlion_data/coinlion/candle data/'
OUT = 'backtest_results_coinlion_data/'

LEVERAGE = 2
TCOST = 0.0016
SL_PCT = 0.15
FUNDING_PER_8H = 0.0001
EMA_P, BB_W, BB_D = 1000, 75, 3.0


def load_data():
    raw = pd.read_csv(PRICE_DATA_DIR + 'btcusd_60m.csv')
    raw['time_utc'] = pd.to_datetime(raw['time_utc'])
    raw['time_est'] = pd.to_datetime(raw['time_est'])
    raw = raw.set_index('time_utc').sort_index()
    raw = raw[['o','h','l','c','v','time_est']]
    raw[['o','h','l','c','v']] = raw[['o','h','l','c','v']].shift(1)
    raw = raw.iloc[1:]
    raw.columns = ['open','high','low','close','volume','time_est']
    return raw.loc['2023-01-01':]


def run_backtest(df, leverage=2, initial_nav=100):
    df = df.copy()
    df['ema'] = EMAIndicator(close=df['close'], window=EMA_P).ema_indicator()
    bb = BollingerBands(close=df['close'], window=BB_W, window_dev=BB_D)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()

    long_cond = (df['close'] > df['bb_upper']) & (df['close'] > df['ema'])
    exit_cond = (df['close'] < df['bb_middle']) | (df['close'] < df['ema'])
    df['signal'] = np.nan
    df.loc[long_cond, 'signal'] = 1
    df.loc[exit_cond, 'signal'] = 0
    df['signal'] = df['signal'].ffill().fillna(0)

    close = df['close'].values
    signal = df['signal'].values
    n = len(close)

    nav = np.zeros(n)
    nav[0] = initial_nav
    cur_nav = initial_nav
    in_pos = False
    entry_px = 0.0
    qty = 0.0
    bars_in = 0
    trades = []

    for i in range(1, n):
        px = close[i]
        prev_px = close[i - 1]

        if in_pos:
            pnl = (px - prev_px) * qty
            bars_in += 1
            funding = abs(qty) * px * FUNDING_PER_8H if bars_in % 8 == 0 else 0.0
            cur_nav += pnl - funding

            pos_ret = (px - entry_px) / entry_px
            if pos_ret <= -SL_PCT or signal[i] == 0:
                tc = abs(qty) * px * TCOST
                cur_nav -= tc
                dur_h = bars_in
                trades.append({
                    'entry_date': df.index[i - bars_in], 'exit_date': df.index[i],
                    'entry_px': entry_px, 'exit_px': px,
                    'qty': qty, 'direction': 1.0,
                    'pnl_raw': (px - entry_px) * qty,
                    'tcost': tc + (abs(qty) * entry_px * TCOST),
                    'pnl_tcost': (px - entry_px) * qty - tc - (abs(qty) * entry_px * TCOST),
                    'duration': dur_h,
                    'exit_type': 'SL' if pos_ret <= -SL_PCT else 'SIGNAL',
                })
                in_pos = False
                qty = 0.0
        else:
            if signal[i] == 1 and signal[i-1] == 0 and cur_nav > 1:
                entry_px = px
                notional = cur_nav * leverage
                qty = notional / px
                tc = abs(qty) * px * TCOST
                cur_nav -= tc
                in_pos = True
                bars_in = 0

        nav[i] = cur_nav

    nav_s = pd.Series(nav, index=df.index)
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        trades_df['ret'] = trades_df['pnl_tcost'] / (trades_df['qty'] / leverage * trades_df['entry_px'])
    return nav_s, trades_df, df


if __name__ == '__main__':
    df_raw = load_data()
    nav, trades, df_sig = run_backtest(df_raw, leverage=LEVERAGE)

    # ---- Stats ----
    nav.index = pd.to_datetime(nav.index)
    nav_daily = nav.groupby(nav.index.date).last()
    nav_daily = pd.Series(nav_daily.values, index=pd.to_datetime(nav_daily.index))
    final = nav_daily.iloc[-1]
    total_ret = (final / 100 - 1) * 100
    daily_ret = nav_daily.pct_change().dropna()
    years = (nav_daily.index[-1] - nav_daily.index[0]).days / 365.25
    ann_ret = ((final / 100) ** (1 / years) - 1) * 100
    vol = daily_ret.std() * np.sqrt(365) * 100
    sharpe = ann_ret / vol
    cummax = nav_daily.cummax()
    dd_pct = ((nav_daily - cummax) / cummax) * 100
    max_dd = dd_pct.min()

    n_trades = len(trades)
    win_rate = (trades['pnl_tcost'] >= 0).mean() * 100
    profit_factor = trades.loc[trades['pnl_tcost'] >= 0, 'pnl_tcost'].sum() / abs(trades.loc[trades['pnl_tcost'] < 0, 'pnl_tcost'].sum())
    avg_win = trades.loc[trades['pnl_tcost'] >= 0, 'ret'].mean() * 100
    avg_loss = trades.loc[trades['pnl_tcost'] < 0, 'ret'].mean() * 100
    avg_ret_trade = trades['ret'].mean() * 100
    avg_dur = trades['duration'].mean()
    total_fees = trades['tcost'].sum()
    gross_pnl = trades['pnl_raw'].sum()

    # Time in market
    time_in = (df_sig['signal'] > 0).sum() / len(df_sig) * 100

    # BTC B&H
    btc_start = df_raw['close'].iloc[0]
    btc_end = df_raw['close'].iloc[-1]
    bnh_ret = (btc_end / btc_start - 1) * 100
    btc_daily = df_raw['close'].copy()
    btc_daily.index = pd.to_datetime(btc_daily.index)
    btc_d = btc_daily.groupby(btc_daily.index.date).last()
    btc_d = pd.Series(btc_d.values, index=pd.to_datetime(btc_d.index))
    btc_vol = btc_d.pct_change().dropna().std() * np.sqrt(365) * 100
    btc_ann = ((btc_end / btc_start) ** (1 / years) - 1) * 100
    btc_sharpe = btc_ann / btc_vol
    btc_dd = ((btc_d - btc_d.cummax()) / btc_d.cummax()).min() * 100

    print("=" * 80)
    print(f"TRENDFOLLOWINGBBEMA - BTC/USD PERP FUTURES 60m - 2x LEVERAGE")
    print("=" * 80)
    print(f"Total Return:      +{total_ret:.1f}%")
    print(f"Ann. Return:       +{ann_ret:.1f}%")
    print(f"Volatility:        {vol:.1f}%")
    print(f"Sharpe:            {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd:.1f}%")
    print(f"Return / DD:       {abs(ann_ret/max_dd):.2f}x")
    print(f"Trades:            {n_trades}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Avg Win:           +{avg_win:.2f}%")
    print(f"Avg Loss:          {avg_loss:.2f}%")
    print(f"Avg Win/Avg Loss:  {abs(avg_win/avg_loss):.2f}x")
    print(f"Avg Trade Return:  +{avg_ret_trade:.2f}%")
    print(f"Avg Duration:      {avg_dur:.0f} hours ({avg_dur/24:.1f} days)")
    print(f"Total Fees:        ${total_fees:.2f}")
    print(f"Gross PnL:         ${gross_pnl:.2f}")
    print(f"Fee Drag:          {total_fees/gross_pnl*100:.1f}%")
    print(f"Time in Market:    {time_in:.1f}%")
    print(f"Final NAV:         ${final:.1f}")
    print()
    print(f"BTC B&H:           +{bnh_ret:.1f}% (Sharpe={btc_sharpe:.2f}, DD={btc_dd:.1f}%)")

    # Annual
    trades['year'] = pd.to_datetime(trades['entry_date']).dt.year
    print("\n--- ANNUAL BREAKDOWN ---")
    for yr in sorted(trades['year'].unique()):
        sub = trades[trades['year'] == yr]
        yr_pnl = sub['pnl_tcost'].sum()
        yr_wr = (sub['pnl_tcost'] >= 0).mean() * 100
        print(f"  {yr}: {len(sub)} trades, PnL=${yr_pnl:.2f}, WR={yr_wr:.0f}%")

    # Regime
    print("\n--- REGIME PERFORMANCE ---")
    regimes = [
        ("Bull", "2023-09-27", "2024-03-13"),
        ("Bull", "2024-09-06", "2024-12-06"),
        ("Bull", "2025-04-08", "2025-10-06"),
        ("Bear", "2024-06-05", "2024-08-05"),
        ("Bear", "2025-01-21", "2025-03-10"),
        ("Sideway", "2023-02-13", "2023-09-15"),
    ]
    for reg, s, e in regimes:
        period = nav.loc[s:e]
        if len(period) > 0:
            ret = (period.iloc[-1] / period.iloc[0] - 1) * 100
            print(f"  {reg:8s} {s} to {e}: {ret:+.1f}%")

    # Trade distribution
    print("\n--- TRADE RETURN DISTRIBUTION ---")
    rets = trades['ret'] * 100
    for p in [0, 5, 25, 50, 75, 95, 100]:
        val = np.percentile(rets, p)
        print(f"  {p:3d}th: {val:+.2f}%")

    # ---- CHARTS ----

    # 1. Equity + Drawdown
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    axes[0].plot(nav_daily.index, nav_daily.values, color='#2196F3', linewidth=1.5)
    axes[0].fill_between(nav_daily.index, 100, nav_daily.values, alpha=0.1, color='#2196F3')
    axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('TrendFollowingBBEMA - BTC/USD Perp Futures 60m - 2x Leverage\n'
                      '(tcost=0.16%, SL=15%, funding=0.01%/8h)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('NAV ($)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate(f'Final: ${final:.0f}', xy=(nav_daily.index[-1], final),
                     fontsize=10, fontweight='bold', color='#2196F3',
                     xytext=(-80, 15), textcoords='offset points')

    axes[1].fill_between(dd_pct.index, 0, dd_pct.values, color='#F44336', alpha=0.4)
    axes[1].plot(dd_pct.index, dd_pct.values, color='#F44336', linewidth=0.8)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_ylabel('Drawdown (%)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    max_dd_date = dd_pct.idxmin()
    axes[1].annotate(f'Max DD: {max_dd:.1f}%', xy=(max_dd_date, max_dd),
                     fontsize=10, fontweight='bold', color='#F44336',
                     xytext=(30, -15), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#F44336'))
    plt.tight_layout()
    fig.savefig(OUT + 'BBEMA_2x_equity_drawdown.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 2. Monthly returns heatmap
    nav_monthly = nav_daily.resample('M').last()
    monthly_ret = nav_monthly.pct_change().dropna() * 100
    yrs = sorted(monthly_ret.index.year.unique())
    hm = pd.DataFrame(index=yrs, columns=range(1, 13), dtype=float)
    for dt, ret in monthly_ret.items():
        hm.loc[dt.year, dt.month] = ret

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    data = hm.values.astype(float)
    im = ax2.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=50)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax2.set_yticks(range(len(yrs)))
    ax2.set_yticklabels(yrs)
    for i in range(len(yrs)):
        for j in range(12):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 20 else 'black'
                ax2.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=9, color=color)
    ax2.set_title('TrendFollowingBBEMA BTC Perp 60m 2x - Monthly Returns (%)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Return %', shrink=0.8)
    plt.tight_layout()
    fig2.savefig(OUT + 'BBEMA_2x_monthly_returns.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)

    print(f"\nSaved: {OUT}BBEMA_2x_equity_drawdown.png")
    print(f"Saved: {OUT}BBEMA_2x_monthly_returns.png")

    # Save trades & portfolio
    trades.to_csv(OUT + 'BBEMA_2x_trades.csv', index=False)
    pd.DataFrame({'nav': nav.values}, index=nav.index).to_csv(OUT + 'BBEMA_2x_portfolio.csv')
    print(f"Saved: {OUT}BBEMA_2x_trades.csv")
    print(f"Saved: {OUT}BBEMA_2x_portfolio.csv")

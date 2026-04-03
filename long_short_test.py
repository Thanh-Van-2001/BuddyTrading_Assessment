# -*- coding: utf-8 -*-
"""
Test Long/Short Balanced TrendFollowingBBEMA on BTC 60m
- Long: close > bb_upper AND close > ema  (existing)
- Short: close < bb_lower AND close < ema (mirror)
- Compare: long-only vs long-short vs short-only
"""
import os, time, math, warnings
import numpy as np
import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest_engine import (
    BackTestSA, _bt_kernel_singlepos_stepwise, _reconstruct_trades_from_series
)

PRICE_DATA_DIR = 'D:/coinlion_data/coinlion/candle data/'
OUT_DIR = 'backtest_results_coinlion_data/'

# ---- Load data ----
def load_data(asset='BTC', timeframe='60m', start_date='2023-01-01'):
    path = PRICE_DATA_DIR + asset.lower() + 'usd_' + timeframe + '.csv'
    raw = pd.read_csv(path)
    raw['time_utc'] = pd.to_datetime(raw['time_utc'])
    raw['time_est'] = pd.to_datetime(raw['time_est'])
    raw = raw.set_index('time_utc').sort_index()
    raw = raw[['o', 'h', 'l', 'c', 'v', 'time_est']]
    raw[['o', 'h', 'l', 'c', 'v']] = raw[['o', 'h', 'l', 'c', 'v']].shift(1)
    raw = raw.iloc[1:]
    raw.columns = ['open', 'high', 'low', 'close', 'volume', 'time_est']
    raw = raw.loc[start_date:]
    return raw


def generate_signals_longshort(df, ema_period=1000, bb_window=75, bb_dev=3.0, mode='longshort'):
    """Generate signals: 1=long, -1=short, 0=flat"""
    df = df.copy()
    df['ema'] = EMAIndicator(close=df['close'], window=ema_period).ema_indicator()
    bb = BollingerBands(close=df['close'], window=bb_window, window_dev=bb_dev)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()

    if mode == 'long_only':
        long_cond = (df['close'] > df['bb_upper']) & (df['close'] > df['ema'])
        exit_cond = (df['close'] < df['bb_middle']) | (df['close'] < df['ema'])
        df['signal'] = np.nan
        df.loc[long_cond, 'signal'] = 1
        df.loc[exit_cond, 'signal'] = 0

    elif mode == 'short_only':
        short_cond = (df['close'] < df['bb_lower']) & (df['close'] < df['ema'])
        exit_cond = (df['close'] > df['bb_middle']) | (df['close'] > df['ema'])
        df['signal'] = np.nan
        df.loc[short_cond, 'signal'] = -1
        df.loc[exit_cond, 'signal'] = 0

    elif mode == 'longshort':
        # Long entry
        long_cond = (df['close'] > df['bb_upper']) & (df['close'] > df['ema'])
        long_exit = (df['close'] < df['bb_middle']) | (df['close'] < df['ema'])

        # Short entry
        short_cond = (df['close'] < df['bb_lower']) & (df['close'] < df['ema'])
        short_exit = (df['close'] > df['bb_middle']) | (df['close'] > df['ema'])

        df['signal'] = np.nan
        df.loc[long_cond, 'signal'] = 1
        df.loc[short_cond, 'signal'] = -1
        df.loc[long_exit & ~short_cond, 'signal'] = 0
        df.loc[short_exit & ~long_cond, 'signal'] = 0

    df['signal'] = df['signal'].ffill().fillna(0)
    return df


def run_backtest_from_signals(df, tcost=0.0016, initial_nav=100):
    """Run backtest using the signal column. Handles long (signal=1) and short (signal=-1)."""
    close = df['close'].values.astype(np.float64)
    signal = df['signal'].values.astype(np.float64)
    n = len(close)

    # Simple vectorized backtest supporting long and short
    nav = np.zeros(n)
    nav[0] = initial_nav
    position = 0.0  # +qty = long, -qty = short
    entry_px = 0.0
    cur_nav = initial_nav
    trades = []

    for i in range(1, n):
        px = close[i]
        prev_px = close[i - 1]

        # MTM on existing position
        if position != 0:
            pnl = (px - prev_px) * position
            tc = 0.0
        else:
            pnl = 0.0
            tc = 0.0

        sig = signal[i]
        prev_sig = signal[i - 1]

        # Position changes
        if prev_sig != sig:
            # Close existing position
            if position != 0:
                tc -= abs(position) * px * tcost
                exit_px = px
                trade_pnl = (exit_px - entry_px) * position
                trade_tc = (abs(entry_px) + abs(exit_px)) * abs(position) * tcost
                trades.append({
                    'direction': 1.0 if position > 0 else -1.0,
                    'entry_px': entry_px,
                    'exit_px': exit_px,
                    'pnl_raw': trade_pnl,
                    'tcost': trade_tc,
                    'pnl_tcost': trade_pnl - trade_tc,
                    'qty': abs(position),
                })
                position = 0.0

            # Open new position
            if sig != 0:
                qty = cur_nav / px if px > 0 else 0
                position = qty * sig  # positive for long, negative for short
                entry_px = px
                tc -= abs(position) * px * tcost

        cur_nav = cur_nav + pnl + tc
        nav[i] = cur_nav

    # Close final position
    if position != 0:
        exit_px = close[-1]
        trade_pnl = (exit_px - entry_px) * position
        trade_tc = (abs(entry_px) + abs(exit_px)) * abs(position) * tcost
        trades.append({
            'direction': 1.0 if position > 0 else -1.0,
            'entry_px': entry_px,
            'exit_px': exit_px,
            'pnl_raw': trade_pnl,
            'tcost': trade_tc,
            'pnl_tcost': trade_pnl - trade_tc,
            'qty': abs(position),
        })

    nav_series = pd.Series(nav, index=df.index)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=['direction', 'entry_px', 'exit_px', 'pnl_raw', 'tcost', 'pnl_tcost', 'qty'])

    return nav_series, trades_df


def compute_metrics(nav_series, trades_df, label=''):
    nav = nav_series.copy()
    nav.index = pd.to_datetime(nav.index)
    nav_daily = nav.groupby(nav.index.date).last()
    nav_daily = pd.Series(nav_daily.values, index=pd.to_datetime(nav_daily.index))

    final_nav = nav_daily.iloc[-1]
    total_ret = (final_nav / 100 - 1) * 100
    daily_ret = nav_daily.pct_change().dropna()
    years = (nav_daily.index[-1] - nav_daily.index[0]).days / 365.25
    ann_ret = ((final_nav / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    vol = daily_ret.std() * np.sqrt(365) * 100
    sharpe = ann_ret / vol if vol > 0 else 0

    cummax = nav_daily.cummax()
    max_dd = ((nav_daily - cummax) / cummax).min() * 100

    n_trades = len(trades_df)
    if n_trades > 0 and 'pnl_tcost' in trades_df.columns:
        win_rate = (trades_df['pnl_tcost'] >= 0).mean() * 100
        avg_ret_trade = (trades_df['pnl_tcost'] / (trades_df['qty'] * trades_df['entry_px'])).mean() * 100
        n_long = (trades_df['direction'] == 1).sum()
        n_short = (trades_df['direction'] == -1).sum()
    else:
        win_rate = 0
        avg_ret_trade = 0
        n_long = 0
        n_short = 0

    return {
        'Config': label,
        'Total Ret%': round(total_ret, 1),
        'Ann Ret%': round(ann_ret, 1),
        'Vol%': round(vol, 1),
        'Sharpe': round(sharpe, 2),
        'Max DD%': round(max_dd, 1),
        'Trades': n_trades,
        'Long': n_long,
        'Short': n_short,
        'Win%': round(win_rate, 1),
        'Avg Trade%': round(avg_ret_trade, 2),
    }


if __name__ == '__main__':
    print("=" * 100)
    print("LONG vs SHORT vs LONG/SHORT — TrendFollowingBBEMA BTC 60m")
    print("ema=1000, bb_window=75, bb_dev=3.0, tcost=0.16%")
    print("=" * 100)

    df_raw = load_data('BTC', '60m', '2023-01-01')
    results = []
    nav_curves = {}

    for mode in ['long_only', 'short_only', 'longshort']:
        t0 = time.time()
        df_sig = generate_signals_longshort(df_raw, ema_period=1000, bb_window=75, bb_dev=3.0, mode=mode)
        nav_s, trades = run_backtest_from_signals(df_sig, tcost=0.0016, initial_nav=100)
        elapsed = time.time() - t0

        metrics = compute_metrics(nav_s, trades, label=mode)
        metrics['Time(s)'] = round(elapsed, 1)
        results.append(metrics)
        nav_curves[mode] = nav_s

        print(f"\n[{mode}] Done in {elapsed:.1f}s")
        if len(trades) > 0 and 'direction' in trades.columns:
            long_trades = trades[trades['direction'] == 1]
            short_trades = trades[trades['direction'] == -1]
            if len(long_trades) > 0:
                long_wr = (long_trades['pnl_tcost'] >= 0).mean() * 100
                long_avg = (long_trades['pnl_tcost'] / (long_trades['qty'] * long_trades['entry_px'])).mean() * 100
                print(f"  Long trades: {len(long_trades)}, WR={long_wr:.1f}%, Avg={long_avg:.2f}%")
            if len(short_trades) > 0:
                short_wr = (short_trades['pnl_tcost'] >= 0).mean() * 100
                short_avg = (short_trades['pnl_tcost'] / (short_trades['qty'] * short_trades['entry_px'])).mean() * 100
                print(f"  Short trades: {len(short_trades)}, WR={short_wr:.1f}%, Avg={short_avg:.2f}%")

    # Summary table
    df_results = pd.DataFrame(results)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(df_results.to_string(index=False))

    # --- Plot all 3 curves ---
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {'long_only': '#2196F3', 'short_only': '#F44336', 'longshort': '#4CAF50'}
    labels = {'long_only': 'Long Only', 'short_only': 'Short Only', 'longshort': 'Long/Short'}

    for mode, nav_s in nav_curves.items():
        nav_s.index = pd.to_datetime(nav_s.index)
        nav_daily = nav_s.groupby(nav_s.index.date).last()
        nav_daily = pd.Series(nav_daily.values, index=pd.to_datetime(nav_daily.index))
        ax.plot(nav_daily.index, nav_daily.values, color=colors[mode],
                linewidth=1.5, label=labels[mode])

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('TrendFollowingBBEMA BTC 60m — Long vs Short vs Long/Short\n'
                 '(ema=1000, bb=75, dev=3.0, tcost=0.16%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('NAV ($)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'BBEMA_long_short_comparison.png'), dpi=200)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR}BBEMA_long_short_comparison.png")

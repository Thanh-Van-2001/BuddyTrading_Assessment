# -*- coding: utf-8 -*-
"""
Test different leverage levels on TrendFollowingBBEMA BTC 60m (Futures)
Leverage amplifies returns AND drawdowns. Also considers liquidation risk.
"""
import warnings, time, math
import numpy as np
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

PRICE_DATA_DIR = 'D:/coinlion_data/coinlion/candle data/'
OUT_DIR = 'backtest_results_coinlion_data/'


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


def run_leveraged_backtest(df, leverage=1.0, tcost=0.0016, sl_pct=0.15,
                           ema_period=1000, bb_window=75, bb_dev=3.0,
                           initial_nav=100, funding_rate_per_8h=0.0001):
    """
    Simulates leveraged futures backtest.
    - leverage: notional = equity * leverage
    - liquidation: if unrealized loss >= equity (margin), position is liquidated
    - funding: charged every 8 hours while in position (perp futures cost)
    - sl_pct: stop-loss on the POSITION (not on equity), so effective equity SL = sl_pct * leverage
    """
    df = df.copy()
    df['ema'] = EMAIndicator(close=df['close'], window=ema_period).ema_indicator()
    bb = BollingerBands(close=df['close'], window=bb_window, window_dev=bb_dev)
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
    qty = 0.0  # notional qty
    bars_in_trade = 0
    trades = []
    liquidated = False

    # Funding: 1 bar = 1 hour, funding every 8 hours
    bars_per_funding = 8

    for i in range(1, n):
        px = close[i]
        prev_px = close[i - 1]

        if in_pos:
            # Mark-to-market PnL (leveraged)
            pnl = (px - prev_px) * qty
            bars_in_trade += 1

            # Funding cost (every 8 hours)
            funding = 0.0
            if bars_in_trade % bars_per_funding == 0:
                funding = abs(qty) * px * funding_rate_per_8h

            cur_nav += pnl - funding

            # Check liquidation (equity <= 0)
            if cur_nav <= 0:
                cur_nav = 0.001  # wiped out
                in_pos = False
                liquidated = True
                trades.append({
                    'entry_px': entry_px, 'exit_px': px,
                    'ret': -1.0, 'pnl': -initial_nav, 'exit_type': 'LIQUIDATED'
                })
                nav[i] = cur_nav
                continue

            # Check SL on position return
            pos_ret = (px - entry_px) / entry_px
            if pos_ret <= -sl_pct:
                # Close at SL
                tc = abs(qty) * px * tcost
                cur_nav -= tc
                trades.append({
                    'entry_px': entry_px, 'exit_px': px,
                    'ret': pos_ret, 'pnl': (px - entry_px) * qty - tc,
                    'exit_type': 'SL'
                })
                in_pos = False
                qty = 0.0

            # Check signal exit
            elif signal[i] == 0:
                tc = abs(qty) * px * tcost
                cur_nav -= tc
                trades.append({
                    'entry_px': entry_px, 'exit_px': px,
                    'ret': pos_ret, 'pnl': (px - entry_px) * qty - tc,
                    'exit_type': 'SIGNAL'
                })
                in_pos = False
                qty = 0.0

        else:
            # Check entry
            if signal[i] == 1 and signal[i-1] == 0 and cur_nav > 1 and not liquidated:
                entry_px = px
                notional = cur_nav * leverage
                qty = notional / px
                tc = abs(qty) * px * tcost
                cur_nav -= tc
                in_pos = True
                bars_in_trade = 0

        nav[i] = cur_nav
        if liquidated:
            break

    # Fill remaining if liquidated early
    if liquidated:
        for j in range(i + 1, n):
            nav[j] = cur_nav

    nav_series = pd.Series(nav, index=df.index)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return nav_series, trades_df, liquidated


def compute_metrics(nav_series, trades_df, leverage, liquidated):
    nav = nav_series.copy()
    nav.index = pd.to_datetime(nav.index)
    nav_daily = nav.groupby(nav.index.date).last()
    nav_daily = pd.Series(nav_daily.values, index=pd.to_datetime(nav_daily.index))

    final = nav_daily.iloc[-1]
    total_ret = (final / 100 - 1) * 100
    daily_ret = nav_daily.pct_change().dropna()
    years = (nav_daily.index[-1] - nav_daily.index[0]).days / 365.25
    ann_ret = ((final / 100) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else -100
    vol = daily_ret.std() * np.sqrt(365) * 100
    sharpe = ann_ret / vol if vol > 0 else 0

    cummax = nav_daily.cummax()
    max_dd = ((nav_daily - cummax) / cummax).min() * 100

    n_trades = len(trades_df)
    if n_trades > 0 and 'pnl' in trades_df.columns:
        win_rate = (trades_df['pnl'] >= 0).mean() * 100
    else:
        win_rate = 0

    return {
        'Leverage': f'{leverage}x',
        'Total Ret%': round(total_ret, 1),
        'Ann Ret%': round(ann_ret, 1),
        'Vol%': round(vol, 1),
        'Sharpe': round(sharpe, 2),
        'Max DD%': round(max_dd, 1),
        'Trades': n_trades,
        'Win%': round(win_rate, 1),
        'Final NAV': round(final, 1),
        'Liquidated': 'YES' if liquidated else 'No',
    }


if __name__ == '__main__':
    print("=" * 110)
    print("LEVERAGE TEST: TrendFollowingBBEMA BTC/USD Perp Futures 60m")
    print("ema=1000, bb=75, dev=3.0, tcost=0.16%, SL=15%, funding=0.01%/8h")
    print("=" * 110)

    df = load_data('BTC', '60m', '2023-01-01')
    results = []
    nav_curves = {}

    leverages = [1, 2, 3, 5, 7, 10]

    for lev in leverages:
        nav_s, trades, liq = run_leveraged_backtest(
            df, leverage=lev, tcost=0.0016, sl_pct=0.15,
            ema_period=1000, bb_window=75, bb_dev=3.0
        )
        metrics = compute_metrics(nav_s, trades, lev, liq)
        results.append(metrics)
        nav_curves[lev] = nav_s
        print(f"  {lev}x: Ret={metrics['Total Ret%']:+.1f}%, Sharpe={metrics['Sharpe']}, "
              f"DD={metrics['Max DD%']}%, Liq={metrics['Liquidated']}")

    df_results = pd.DataFrame(results)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print("\n" + "=" * 110)
    print(df_results.to_string(index=False))
    print("=" * 110)

    # Also test with tighter SL for higher leverage
    print("\n\n" + "=" * 110)
    print("LEVERAGE + TIGHTER SL TEST")
    print("=" * 110)

    combos = [
        (1, 0.15),
        (2, 0.10),
        (2, 0.07),
        (3, 0.07),
        (3, 0.05),
        (5, 0.05),
        (5, 0.03),
    ]
    results2 = []
    for lev, sl in combos:
        nav_s, trades, liq = run_leveraged_backtest(
            df, leverage=lev, tcost=0.0016, sl_pct=sl,
            ema_period=1000, bb_window=75, bb_dev=3.0
        )
        m = compute_metrics(nav_s, trades, lev, liq)
        m['SL%'] = f'{sl*100:.0f}%'
        m['Equity SL%'] = f'{sl*lev*100:.0f}%'
        results2.append(m)
        print(f"  {lev}x SL={sl*100:.0f}% (eq SL={sl*lev*100:.0f}%): "
              f"Ret={m['Total Ret%']:+.1f}%, Sharpe={m['Sharpe']}, DD={m['Max DD%']}%")

    df2 = pd.DataFrame(results2)
    print("\n" + "=" * 110)
    print(df2.to_string(index=False))
    print("=" * 110)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {1: '#2196F3', 2: '#4CAF50', 3: '#FF9800', 5: '#F44336', 7: '#9C27B0', 10: '#795548'}
    for lev, nav_s in nav_curves.items():
        nav_s.index = pd.to_datetime(nav_s.index)
        nav_d = nav_s.groupby(nav_s.index.date).last()
        nav_d = pd.Series(nav_d.values, index=pd.to_datetime(nav_d.index))
        ax.plot(nav_d.index, nav_d.values, color=colors.get(lev, 'gray'),
                linewidth=1.5, label=f'{lev}x leverage')

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('TrendFollowingBBEMA BTC Perp Futures 60m - Leverage Comparison\n'
                 '(ema=1000, bb=75, dev=3.0, tcost=0.16%, SL=15%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('NAV ($)')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR + 'BBEMA_leverage_comparison.png', dpi=200)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR}BBEMA_leverage_comparison.png")

# -*- coding: utf-8 -*-
"""
Walk-Forward Analysis for TrendFollowingBBEMA on BTC 60m
- Train on expanding window up to month start
- Optimize ema_period, bb_window, bb_dev via Optuna on IS
- Run OOS on each month with best IS params
- Stitch OOS NAVs together for true out-of-sample equity curve
"""
import os, time, warnings, operator
import numpy as np
import pandas as pd
import optuna
from dateutil.relativedelta import relativedelta
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

warnings.simplefilter(action='ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest_engine import TrendFollowingBBEMA, _bt_kernel_singlepos_stepwise, _reconstruct_trades_from_series

PRICE_DATA_DIR = 'D:/coinlion_data/coinlion/candle data/'
OUT_DIR = 'backtest_results_coinlion_data/walk_forward/'
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Config ----
ASSET = 'BTC'
TIMEFRAME = '60m'
TCOST = 0.0016
LEVERAGE = 2
FUNDING_PER_8H = 0.0001
SL_PCT = 0.15
START_YM = '2024-01'   # WF starts here (2023 = pure IS warm-up)
N_TRIALS = 80
BARS_PER_YEAR = 365 * 24

BASE_PARAMS = {
    'start_date': '2020-01-01',  # load all data
    'asset': ASSET,
    'timeframe': TIMEFRAME,
    'datafile_path': PRICE_DATA_DIR + ASSET.lower() + 'usd_' + TIMEFRAME + '.csv',
    'datafile_extra_path': None,
    'max_no_of_trades': 1,
    'initial_nav': 100,
    'tp_pct': 1.0,
    'sl_pct': 0.15,
    'max_holding': 1000000,
    'tcost': TCOST,
}


def generate_signals_on_df(df, ema_period, bb_window, bb_dev):
    """Compute indicators + signals on full df (needs history for warmup)."""
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
    return df


def run_fast_on_slice(df_full, ema_period, bb_window, bb_dev, tcost, initial_nav=100,
                      backtest_start=None, backtest_end=None,
                      leverage=LEVERAGE, sl_pct=SL_PCT, funding_per_8h=FUNDING_PER_8H):
    """
    Compute indicators on full df (for warmup), then leveraged backtest only the
    [backtest_start:backtest_end] slice.
    """
    df = generate_signals_on_df(df_full, ema_period, bb_window, bb_dev)

    if backtest_start is not None or backtest_end is not None:
        df = df.loc[backtest_start:backtest_end].copy()

    close = df['close'].values
    signal = df['signal'].values
    n = len(close)
    if n == 0:
        return pd.Series(dtype=float), pd.DataFrame()

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
            funding = abs(qty) * px * funding_per_8h if bars_in % 8 == 0 else 0.0
            cur_nav += pnl - funding

            if cur_nav <= 0:
                cur_nav = 0.001
                in_pos = False
                trades.append({'pnl_tcost': -initial_nav})
                nav[i] = cur_nav
                continue

            pos_ret = (px - entry_px) / entry_px
            if pos_ret <= -sl_pct or signal[i] == 0:
                tc = abs(qty) * px * tcost
                cur_nav -= tc
                trades.append({'pnl_tcost': (px - entry_px) * qty - tc})
                in_pos = False
                qty = 0.0
        else:
            if signal[i] == 1 and signal[i-1] == 0 and cur_nav > 1:
                entry_px = px
                qty = (cur_nav * leverage) / px
                tc = abs(qty) * px * tcost
                cur_nav -= tc
                in_pos = True
                bars_in = 0

        nav[i] = cur_nav

    nav_series = pd.Series(nav, index=df.index)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return nav_series, trades_df


def load_full_data():
    raw = pd.read_csv(BASE_PARAMS['datafile_path'])
    raw['time_utc'] = pd.to_datetime(raw['time_utc'])
    raw['time_est'] = pd.to_datetime(raw['time_est'])
    raw = raw.set_index('time_utc').sort_index()
    raw = raw[['o', 'h', 'l', 'c', 'v', 'time_est']]
    raw[['o', 'h', 'l', 'c', 'v']] = raw[['o', 'h', 'l', 'c', 'v']].shift(1)
    raw = raw.iloc[1:]
    raw.columns = ['open', 'high', 'low', 'close', 'volume', 'time_est']
    return raw


if __name__ == '__main__':
    print("=" * 80)
    print("WALK-FORWARD ANALYSIS: TrendFollowingBBEMA - BTC 60m")
    print("=" * 80)

    full_df = load_full_data()
    print(f"Full data: {full_df.index[0]} to {full_df.index[-1]} ({len(full_df)} bars)")

    start_dt = pd.to_datetime(START_YM + '-01')
    last_date = full_df.index[-1]

    months = []
    cur = start_dt
    while cur <= last_date:
        months.append(cur)
        cur += relativedelta(months=1)

    all_oos_navs = []
    metrics_rows = []
    cumulative_nav = 100.0

    for first_day in months:
        month_label = first_day.strftime('%Y-%m')
        train_end = first_day - pd.Timedelta(seconds=1)
        month_end = first_day + relativedelta(months=1) - pd.Timedelta(seconds=1)

        train_df = full_df.loc[:train_end].copy()
        oos_df = full_df.loc[first_day:month_end].copy()

        if len(train_df) < 1500 or len(oos_df) == 0:
            print(f"[{month_label}] Skip: train={len(train_df)}, oos={len(oos_df)}")
            continue

        print(f"\n[{month_label}] Train: {len(train_df)} bars, OOS: {len(oos_df)} bars")

        # IS backtest starts from 2023-01-01 to use sufficient but not too old data
        is_start = '2023-01-01'

        # --- Optuna on IS ---
        def objective(trial):
            ep = trial.suggest_int('ema_period', 300, 1500, step=50)
            bw = trial.suggest_int('bb_window', 30, 150, step=5)
            bd = trial.suggest_float('bb_dev', 2.0, 4.0, step=0.25)

            # Compute indicators on full train_df (for warmup), backtest from is_start
            nav_s, trades = run_fast_on_slice(train_df, ep, bw, bd, TCOST,
                                              backtest_start=is_start)
            if len(nav_s) < 2 or len(trades) < 10:
                return -1e9

            rets = nav_s.pct_change().dropna()
            if rets.std() == 0:
                return -1e9
            sharpe = rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR)
            return float(sharpe) if np.isfinite(sharpe) else -1e9

        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best = study.best_params
        print(f"  IS best: ema={best['ema_period']}, bb_win={best['bb_window']}, "
              f"bb_dev={best['bb_dev']}, Sharpe={study.best_value:.3f}")

        # --- Run OOS: compute indicators on data up to month_end, backtest only OOS window ---
        full_up_to_oos = full_df.loc[:month_end].copy()
        oos_nav, oos_trades = run_fast_on_slice(
            full_up_to_oos, best['ema_period'], best['bb_window'], best['bb_dev'],
            TCOST, initial_nav=cumulative_nav,
            backtest_start=first_day, backtest_end=month_end
        )

        if len(oos_nav) > 0:
            oos_ret = (oos_nav.iloc[-1] / oos_nav.iloc[0] - 1) * 100
            cumulative_nav = oos_nav.iloc[-1]
            all_oos_navs.append(oos_nav)

            n_trades = len(oos_trades)
            win_rate = (oos_trades['pnl_tcost'] >= 0).mean() * 100 if n_trades > 0 else 0

            metrics_rows.append({
                'month': month_label,
                'ema_period': best['ema_period'],
                'bb_window': best['bb_window'],
                'bb_dev': best['bb_dev'],
                'is_sharpe': round(study.best_value, 3),
                'oos_return_%': round(oos_ret, 2),
                'oos_trades': n_trades,
                'oos_win_rate_%': round(win_rate, 1),
                'cumulative_nav': round(cumulative_nav, 2),
            })
            print(f"  OOS: return={oos_ret:.2f}%, trades={n_trades}, "
                  f"win={win_rate:.1f}%, NAV={cumulative_nav:.2f}")

    # --- Summary ---
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(OUT_DIR, 'wf_monthly_metrics.csv'), index=False)
    print("\n" + "=" * 80)
    print("WALK-FORWARD MONTHLY RESULTS")
    print("=" * 80)
    print(df_metrics.to_string(index=False))

    total_ret = (cumulative_nav / 100 - 1) * 100
    print(f"\nCumulative OOS NAV: {cumulative_nav:.2f} (Total return: {total_ret:.1f}%)")

    # --- Stitch OOS NAV and plot ---
    if all_oos_navs:
        stitched = pd.concat(all_oos_navs)
        stitched = stitched[~stitched.index.duplicated(keep='last')]
        stitched.index = pd.to_datetime(stitched.index)
        nav_daily = stitched.groupby(stitched.index.date).last()
        nav_daily = pd.Series(nav_daily.values, index=pd.to_datetime(nav_daily.index))

        # Compute OOS stats
        daily_ret = nav_daily.pct_change().dropna()
        years = (nav_daily.index[-1] - nav_daily.index[0]).days / 365.25
        ann_ret = ((cumulative_nav / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        vol = daily_ret.std() * np.sqrt(365) * 100
        sharpe_oos = ann_ret / vol if vol > 0 else 0
        cummax = nav_daily.cummax()
        max_dd = ((nav_daily - cummax) / cummax).min() * 100

        print(f"\nOOS Aggregate Stats:")
        print(f"  Ann. Return: {ann_ret:.1f}%")
        print(f"  Volatility:  {vol:.1f}%")
        print(f"  Sharpe:      {sharpe_oos:.2f}")
        print(f"  Max DD:      {max_dd:.1f}%")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(nav_daily.index, nav_daily.values, color='#2196F3', linewidth=1.5)
        ax.fill_between(nav_daily.index, 100, nav_daily.values, alpha=0.1, color='#2196F3')
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Walk-Forward OOS Equity — BBEMA BTC 60m\n'
                     f'Ann.Ret={ann_ret:.1f}%, Sharpe={sharpe_oos:.2f}, MaxDD={max_dd:.1f}%',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('NAV ($)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'wf_oos_equity.png'), dpi=200)
        plt.close(fig)
        print(f"\nSaved: {OUT_DIR}wf_oos_equity.png")

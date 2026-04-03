# -*- coding: utf-8 -*-
"""
Optuna optimization for TrendFollowingBBEMA on BTC 60m
Search space: ema_period, bb_window, bb_dev
Objective: maximize Sharpe ratio (with tcost=0.0016)
"""
import os
import math
import time
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from coinlion_backtest_coinlion_data import TrendFollowingBBEMA

optuna.logging.set_verbosity(optuna.logging.WARNING)

PRICE_DATA_DIR = 'D:/coinlion_data/coinlion/candle data/'
OUT_DIR = 'backtest_results_coinlion_data/'
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_PARAMS = {
    'start_date': '2023-01-01',
    'asset': 'BTC',
    'timeframe': '60m',
    'datafile_path': PRICE_DATA_DIR + 'btcusd_60m.csv',
    'datafile_extra_path': None,
    'max_no_of_trades': 1,
    'initial_nav': 100,
    'tp_pct': 1.0,
    'sl_pct': 0.15,
    'max_holding': 1000000,
    'tcost': 0.0016,
}

MIN_TRADES = 20
BARS_PER_YEAR = 365 * 24  # 60m bars

def compute_metrics(strat):
    nav = strat.port_history['nav']
    nav = pd.to_numeric(nav, errors='coerce').dropna()
    if len(nav) < 2:
        return None

    initial = float(nav.iloc[0])
    final = float(nav.iloc[-1])
    total_ret = final / initial - 1.0
    years = len(nav) / BARS_PER_YEAR
    cagr = (final / initial) ** (1.0 / years) - 1.0 if years > 0 else 0

    rets = nav.pct_change().dropna()
    vol = rets.std(ddof=1) * math.sqrt(BARS_PER_YEAR) if len(rets) > 2 else 0
    sharpe = (rets.mean() / rets.std(ddof=1)) * math.sqrt(BARS_PER_YEAR) if (len(rets) > 2 and rets.std(ddof=1) > 0) else 0

    cummax = nav.cummax()
    dd_pct = ((nav - cummax) / cummax).min()

    trades = strat.trades_history
    n_trades = len(trades)
    win_rate = (trades['pnl_tcost'] >= 0).mean() * 100 if n_trades > 0 else 0

    return {
        'total_ret': total_ret,
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'max_dd': dd_pct,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'final_nav': final,
    }


def objective(trial):
    ema_period = trial.suggest_int('ema_period', 200, 2000, step=50)
    bb_window = trial.suggest_int('bb_window', 20, 200, step=5)
    bb_dev = trial.suggest_float('bb_dev', 1.5, 4.0, step=0.25)

    params = FIXED_PARAMS.copy()
    params['ema_period'] = ema_period
    params['bb_window'] = bb_window
    params['bb_dev'] = bb_dev

    try:
        strat = TrendFollowingBBEMA(params)
        strat.run_backtest_fast()
        metrics = compute_metrics(strat)

        if metrics is None or metrics['n_trades'] < MIN_TRADES:
            return -1e9

        for k, v in metrics.items():
            trial.set_user_attr(k, float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)

        return float(metrics['sharpe'])
    except Exception as e:
        trial.set_user_attr('error', str(e))
        return -1e9


if __name__ == '__main__':
    print("=" * 80)
    print("OPTUNA OPTIMIZATION: TrendFollowingBBEMA - BTC 60m")
    print("Search: ema_period[200-2000], bb_window[20-200], bb_dev[1.5-4.0]")
    print("Objective: maximize Sharpe | tcost=0.0016 | SL=15%")
    print("=" * 80)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="BBEMA_BTC_60m",
        direction="maximize",
        sampler=sampler
    )

    t0 = time.time()
    study.optimize(objective, n_trials=150, n_jobs=1, show_progress_bar=True)
    elapsed = time.time() - t0

    # Results
    print(f"\nOptimization done in {elapsed:.1f}s ({len(study.trials)} trials)")
    print(f"\nBest Sharpe: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_attrs = study.best_trial.user_attrs
    print(f"\nBest trial metrics:")
    for k, v in best_attrs.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save trials
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df_trials.to_csv(os.path.join(OUT_DIR, 'BBEMA_optuna_trials.csv'), index=False)

    # Save best params
    best = {
        'best_params': study.best_params,
        'best_sharpe': study.best_value,
        'best_metrics': {k: v for k, v in best_attrs.items() if k != 'error'},
        'baseline_params': {'ema_period': 1000, 'bb_window': 75, 'bb_dev': 3.0},
    }
    with open(os.path.join(OUT_DIR, 'BBEMA_best_params.json'), 'w') as f:
        json.dump(best, f, indent=2)

    print(f"\nSaved: {OUT_DIR}BBEMA_optuna_trials.csv")
    print(f"Saved: {OUT_DIR}BBEMA_best_params.json")

    # --- Compare with baseline ---
    print("\n" + "=" * 80)
    print("BASELINE vs OPTUNA BEST COMPARISON")
    print("=" * 80)

    baseline_params = FIXED_PARAMS.copy()
    baseline_params.update({'ema_period': 1000, 'bb_window': 75, 'bb_dev': 3.0})
    strat_base = TrendFollowingBBEMA(baseline_params)
    strat_base.run_backtest_fast()
    base_m = compute_metrics(strat_base)

    opt_params = FIXED_PARAMS.copy()
    opt_params.update(study.best_params)
    strat_opt = TrendFollowingBBEMA(opt_params)
    strat_opt.run_backtest_fast()
    opt_m = compute_metrics(strat_opt)

    comparison = pd.DataFrame({
        'Metric': ['Total Return %', 'CAGR %', 'Volatility %', 'Sharpe', 'Max DD %', 'Trades', 'Win Rate %', 'Final NAV'],
        'Baseline': [
            f"{base_m['total_ret']*100:.1f}", f"{base_m['cagr']*100:.1f}", f"{base_m['vol']*100:.1f}",
            f"{base_m['sharpe']:.2f}", f"{base_m['max_dd']*100:.1f}",
            f"{base_m['n_trades']}", f"{base_m['win_rate']:.1f}", f"{base_m['final_nav']:.1f}"
        ],
        'Optuna Best': [
            f"{opt_m['total_ret']*100:.1f}", f"{opt_m['cagr']*100:.1f}", f"{opt_m['vol']*100:.1f}",
            f"{opt_m['sharpe']:.2f}", f"{opt_m['max_dd']*100:.1f}",
            f"{opt_m['n_trades']}", f"{opt_m['win_rate']:.1f}", f"{opt_m['final_nav']:.1f}"
        ],
    })
    print(comparison.to_string(index=False))

    # Param comparison
    print(f"\nBaseline params: ema_period=1000, bb_window=75, bb_dev=3.0")
    print(f"Optuna params:   {study.best_params}")

    # Top 10 trials
    print("\n--- TOP 10 TRIALS BY SHARPE ---")
    top10 = df_trials.nlargest(10, 'value')[['number', 'value', 'params_ema_period', 'params_bb_window', 'params_bb_dev',
                                              'user_attrs_total_ret', 'user_attrs_max_dd', 'user_attrs_n_trades']]
    top10.columns = ['Trial', 'Sharpe', 'EMA', 'BB_Win', 'BB_Dev', 'TotalRet', 'MaxDD', 'Trades']
    print(top10.to_string(index=False))

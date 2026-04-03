# -*- coding: utf-8 -*-
"""
Backtest Engine for TrendFollowingBBEMA Strategy
BTC/USD Perpetual Futures | Coinbase | 60-min bars | Long-only | 2x Leverage

This module contains:
  - _bt_kernel_singlepos_stepwise: Numba-accelerated backtest kernel (single position)
  - _reconstruct_trades_from_series: Extracts trade-level records from position vectors
  - BackTestSA: Base backtesting class (data loading, portfolio tracking, trade management)
  - TrendFollowingBBEMA: Strategy using Bollinger Band breakout + EMA trend filter

Entry: Close > BB_Upper(75, 3.0) AND Close > EMA(1000)
Exit:  Close < BB_Middle(75)     OR  Close < EMA(1000)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import time
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

PRICE_DATA_DIR = 'D:/coinlion_data/coinlion/candle data/'
BACKTEST_DATA_DIR = "backtest_results_coinlion_data/"
os.makedirs(BACKTEST_DATA_DIR, exist_ok=True)

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco


# ======================== FAST BACKTEST KERNEL ========================

@njit
def _bt_kernel_singlepos_stepwise(close, signal,
                                  tp_mult, sl_mult, max_holding_limit, tcost_rate,
                                  initial_nav, end_force_close_idx):
    """
    Numba-accelerated single-position backtest kernel.
    Processes bar-by-bar with step-wise mark-to-market.

    Parameters:
        close:    array of close prices
        signal:   array of signals (1=long, 0=exit)
        tp_mult:  take-profit multiplier (e.g. 2.0 = +100%)
        sl_mult:  stop-loss multiplier (e.g. 0.85 = -15%)
        max_holding_limit: max bars to hold (large = no limit)
        tcost_rate: transaction cost per unit traded
        initial_nav: starting NAV
        end_force_close_idx: bar index to force-close (last bar)

    Returns:
        qty, trade, pnl_raw, tcost_arr, pnl_tcost, nav (all arrays length n)
    """
    n = close.shape[0]
    qty        = np.zeros(n)
    trade      = np.zeros(n)
    pnl_raw    = np.zeros(n)
    tcost_arr  = np.zeros(n)
    pnl_tcost  = np.zeros(n)
    nav        = np.zeros(n)

    prev_qty = 0.0
    cur_nav  = initial_nav
    in_pos   = 0
    hold     = 0
    freeze   = False
    entry_px = 0.0

    for i in range(n):
        px = close[i]
        do_trade = 0.0

        if in_pos == 0:
            if (signal[i] > 0.0) and (i != end_force_close_idx) and (not freeze):
                do_trade = cur_nav / px if px > 0.0 else 0.0
                prev_qty = prev_qty + do_trade
                in_pos   = 1
                hold     = 0
                entry_px = px
        else:
            hold += 1
            if signal[i] == 0.0:
                do_trade = -prev_qty
                prev_qty = 0.0
                in_pos   = 0
                freeze   = False
            else:
                tp_hit = px >= entry_px * tp_mult
                sl_hit = px <= entry_px * sl_mult
                mh_hit = (hold == max_holding_limit)
                force_close = (i == end_force_close_idx)
                if tp_hit or sl_hit or mh_hit or force_close:
                    do_trade = -prev_qty
                    prev_qty = 0.0
                    in_pos   = 0
                    freeze   = True

        # Step-wise mark-to-market
        if i == 0:
            prev_px = px
            pnl = 0.0
            prev_nav = cur_nav
        else:
            prev_px = close[i-1]
            prev_nav = nav[i-1]
            pnl = (px - prev_px) * (qty[i-1])

        tcost = -abs(do_trade) * px * tcost_rate

        trade[i]     = do_trade
        qty[i]       = (qty[i-1] if i > 0 else 0.0) + do_trade
        pnl_raw[i]   = pnl
        tcost_arr[i] = tcost
        pnl_tcost[i] = pnl + tcost
        nav[i]       = prev_nav + pnl_tcost[i]
        cur_nav      = nav[i]

        if (in_pos == 0) and (signal[i] == 0.0):
            freeze = False

    return qty, trade, pnl_raw, tcost_arr, pnl_tcost, nav


def _reconstruct_trades_from_series(df, trade_col, close_col, tcost_rate, tp_mult, sl_mult):
    """Extract individual trade records from position change vectors."""
    trades = []
    open_idx = None
    qty_open = 0.0
    entry_px = 0.0
    time_col = 'time_est' if 'time_est' in df.columns else None

    for t, trd in trade_col.items():
        if trd > 0 and open_idx is None:
            open_idx = t
            qty_open = float(trd)
            entry_px = float(close_col.loc[t])
        elif trd < 0 and open_idx is not None:
            exit_idx = t
            exit_px  = float(close_col.loc[t])
            pnl_raw  = (exit_px - entry_px) * qty_open
            tcost    = (entry_px + exit_px) * qty_open * tcost_rate
            pnl_t    = pnl_raw - tcost
            trades.append({
                'open': 0,
                'entry_date': df.loc[open_idx, time_col] if time_col else open_idx,
                'qty': qty_open,
                'direction': 1.0,
                'entry_px': entry_px,
                'tp_px': entry_px * tp_mult,
                'sl_px': entry_px * sl_mult,
                'duration': df.index.get_loc(exit_idx) - df.index.get_loc(open_idx),
                'exit_date': df.loc[exit_idx, time_col] if time_col else exit_idx,
                'exit_px': exit_px,
                'pnl_raw': pnl_raw,
                'tcost': tcost,
                'pnl_tcost': pnl_t
            })
            open_idx = None
            qty_open = 0.0
            entry_px = 0.0

    cols = ['open','entry_date','qty','direction','entry_px','tp_px','sl_px',
            'duration','exit_date','exit_px','pnl_raw','tcost','pnl_tcost']
    return pd.DataFrame(trades, columns=cols)


# ======================== BASE BACKTEST CLASS ========================

class BackTestSA:
    """
    Base backtesting class for single-asset strategies.
    Handles data loading, portfolio tracking, trade management, and performance output.
    Child classes must implement generate_signals() to populate self.df['signal'].
    """

    def __init__(self, parameters):
        self.start_date = pd.to_datetime(parameters['start_date'])
        self.asset = parameters['asset']
        self.timeframe = parameters['timeframe']
        self.cur_nav = parameters['initial_nav']
        self.max_no_of_trades = parameters['max_no_of_trades']
        self.in_trade = False
        self.tp_mult = 1 + parameters['tp_pct']
        self.sl_mult = 1 - parameters['sl_pct']
        self.max_holding_limit = parameters['max_holding']
        self.freeze = False
        self.tcost = parameters['tcost']

        self.load_data(parameters['datafile_path'])
        self.load_extra_data(parameters['datafile_extra_path'])

        self.end_date = self.df.index.values[-1]

        self.trades_history = pd.DataFrame(columns=[
            'open','entry_date','qty','direction','entry_px','tp_px','sl_px',
            'duration','exit_date','exit_px','pnl_raw','tcost','pnl_tcost'
        ])

        if parameters['datafile_extra_path'] is None:
            self.port_history = pd.DataFrame(
                0.0, index=self.df.index,
                columns=['qty','trade','pnl_raw','tcost','pnl_tcost','nav']
            )
        else:
            self.port_history = (
                pd.DataFrame(index=pd.concat([self.df,self.df_extra],axis=1).index,
                             columns=['qty','trade','pnl_raw','tcost','pnl_tcost','nav'])
                .fillna(0)
            )

    def load_data(self, csv_path):
        """Load and prepare OHLCV data from CSV."""
        raw_data = pd.read_csv(csv_path)
        raw_data['time_utc'] = pd.to_datetime(raw_data['time_utc'])
        raw_data['time_est'] = pd.to_datetime(raw_data['time_est'])
        raw_data = raw_data.set_index('time_utc').sort_index()
        raw_data = raw_data[['o','h','l','c','v','time_est']]
        raw_data[['o','h','l','c','v']] = raw_data[['o','h','l','c','v']].shift(1)
        raw_data = raw_data.iloc[1:]
        raw_data.columns = ['open','high','low','close','volume','time_est']
        raw_data = raw_data.loc[self.start_date:]
        self.df = raw_data.copy()

    def load_extra_data(self, csv_path):
        """Load extra timeframe data (optional)."""
        if csv_path is not None:
            raw_data = pd.read_csv(csv_path)
            raw_data['time_utc'] = pd.to_datetime(raw_data['time_utc'])
            raw_data['time_est'] = pd.to_datetime(raw_data['time_est'])
            raw_data = raw_data.set_index('time_utc').sort_index()
            raw_data = raw_data[['o','h','l','c','v','time_est']]
            raw_data[['o','h','l','c','v']] = raw_data[['o','h','l','c','v']].shift(1)
            raw_data = raw_data.iloc[1:]
            raw_data.columns = ['open','high','low','close','volume','time_est']
            raw_data = raw_data.loc[self.start_date:]
            self.df_extra = raw_data.copy()

    def generate_signals(self):
        """Override in child class. Must create self.df['signal'] column."""
        if 'signal' not in self.df.columns:
            raise Exception('You have not created signals yet')

    def run_backtest_fast(self):
        """Run vectorized backtest using the numba kernel (single position only)."""
        self.generate_signals()
        df = self.df
        if 'signal' not in df.columns:
            raise Exception('You have not created signals yet')

        close  = df['close'].to_numpy(dtype=np.float64)
        signal = df['signal'].to_numpy(dtype=np.float64)
        n = close.shape[0]
        if n == 0:
            raise ValueError("Empty data")

        initial_nav   = float(self.cur_nav)
        tp_mult       = float(self.tp_mult)
        sl_mult       = float(self.sl_mult)
        max_holding   = int(self.max_holding_limit)
        tcost_rate    = float(self.tcost)
        end_force_idx = n - 1

        if int(self.max_no_of_trades) != 1:
            print("[FAST] max_no_of_trades > 1 not supported in fast mode.")
            return

        qty, trade, pnl_raw, tcost_arr, pnl_tcost, nav = _bt_kernel_singlepos_stepwise(
            close, signal, tp_mult, sl_mult, max_holding, tcost_rate,
            initial_nav, end_force_idx
        )

        idx = df.index
        ph = self.port_history
        if not isinstance(ph, pd.DataFrame) or not ph.index.equals(idx):
            ph = pd.DataFrame(index=idx, columns=['qty','trade','pnl_raw','tcost','pnl_tcost','nav']).fillna(0.0)
        ph['qty']       = qty
        ph['trade']     = trade
        ph['pnl_raw']   = pnl_raw
        ph['tcost']     = tcost_arr
        ph['pnl_tcost'] = pnl_tcost
        ph['nav']       = nav
        self.port_history = ph
        self.cur_nav = float(nav[-1])

        # Reconstruct trade records
        try:
            df_out = self.df.copy()
            df_out['trade'] = self.port_history['trade']
            df_out['close'] = self.df['close']
            self.trades_history = _reconstruct_trades_from_series(
                df_out, df_out['trade'], df_out['close'],
                tcost_rate=tcost_rate, tp_mult=tp_mult, sl_mult=sl_mult
            )
        except Exception as e:
            print(f"[FAST] reconstruct trades failed: {e}")

    def save_backtest_fast(self, out_dir=None):
        """Save trade and portfolio history to CSV."""
        out_dir = out_dir or BACKTEST_DATA_DIR
        os.makedirs(out_dir, exist_ok=True)
        strat_name = self.__class__.__name__
        asset_name = self.asset
        time_frame = self.timeframe

        trades_file = os.path.join(out_dir, f"{strat_name}_{asset_name}_{time_frame}_trades_history.csv")
        port_file   = os.path.join(out_dir, f"{strat_name}_{asset_name}_{time_frame}_portfolio_history.csv")

        if isinstance(self.trades_history, pd.DataFrame) and len(self.trades_history):
            self.trades_history.to_csv(trades_file, index=False)
        else:
            pd.DataFrame(columns=['open','entry_date','qty','direction','entry_px','tp_px','sl_px',
                                  'duration','exit_date','exit_px','pnl_raw','tcost','pnl_tcost']).to_csv(trades_file, index=False)
        self.port_history.to_csv(port_file)

    def show_performace(self):
        """Plot equity curve."""
        plt.style.use('ggplot')
        self.port_history['nav'].plot()
        plt.title("Strategy results")
        plt.show()


# ======================== STRATEGY: TrendFollowingBBEMA ========================

class TrendFollowingBBEMA(BackTestSA):
    """
    Trend-following strategy using Bollinger Bands breakout + EMA trend filter.

    Entry (Long): Close > BB_Upper(bb_window, bb_dev) AND Close > EMA(ema_period)
    Exit:         Close < BB_Middle(bb_window)         OR  Close < EMA(ema_period)

    Default parameters (optimized via Optuna, 150 trials):
        ema_period = 1000  (~42 days on hourly bars)
        bb_window  = 75    (~3 days)
        bb_dev     = 3.0   (3 standard deviations)
    """

    def __init__(self, parameters=None):
        super().__init__(parameters or {})
        self.parameters = parameters or {}

        self.ema_period = self.parameters.get("ema_period", 1000)
        self.bb_window  = self.parameters.get("bb_window", 75)
        self.bb_dev     = self.parameters.get("bb_dev", 3.0)

    def generate_signals(self):
        df = self.df.copy()

        # Long-term trend filter
        df["ema"] = EMAIndicator(close=df["close"], window=self.ema_period).ema_indicator()

        # Volatility breakout
        bb = BollingerBands(close=df["close"], window=self.bb_window, window_dev=self.bb_dev)
        df["bb_upper"]  = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()

        # Entry: breakout above upper BB while above EMA (momentum + trend alignment)
        long_condition = (df["close"] > df["bb_upper"]) & (df["close"] > df["ema"])

        # Exit: price returns below BB midline OR drops below EMA
        exit_condition = (df["close"] < df["bb_middle"]) | (df["close"] < df["ema"])

        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df


# ======================== MAIN ========================

if __name__ == '__main__':
    start_date = '2023-01-01'
    asset = 'BTC'
    timeframe = '60m'
    filename = PRICE_DATA_DIR + asset.lower() + 'usd_' + timeframe + '.csv'

    parameters = {
        'start_date': start_date,
        'asset': asset,
        'timeframe': timeframe,
        'datafile_path': filename,
        'datafile_extra_path': None,
        'max_no_of_trades': 1,
        'initial_nav': 100,
        'tp_pct': 1.0,
        'sl_pct': 0.15,
        'max_holding': 1000000,
        'tcost': 0.0016,
        'ema_period': 1000,
        'bb_window': 75,
        'bb_dev': 3,
    }

    start_time = time.time()
    strat = TrendFollowingBBEMA(parameters)
    strat.run_backtest_fast()
    strat.save_backtest_fast()
    strat.show_performace()
    print(f"Elapsed time: {time.time() - start_time:.2f}s")

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:48:46 2024

@author: wayne
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import time
from ta.momentum import AwesomeOscillatorIndicator, StochasticOscillator, RSIIndicator, ROCIndicator, WilliamsRIndicator
from ta.trend import ADXIndicator, CCIIndicator, MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
import warnings

# Tắt tất cả các cảnh báo FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
PRICE_DATA_DIR = 'D:/coinlion_data/coinlion/candle data/'
ORDERBOOK_DIR  = 'D:/coinlion_data/coinlion/orderbook data/'
from multiprocessing import Pool
import os
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

@njit
def _bt_kernel_singlepos_stepwise(close, signal,
                                  tp_mult, sl_mult, max_holding_limit, tcost_rate,
                                  initial_nav, end_force_close_idx):
    """
    Mimics update_port_history (step-wise MTM) and run_backtest()'s input/output logic.
    - Charge per bar when trade occurs: -abs(trade)*close[i]*tcost
    - freeze: True after TP/SL/MaxHolding; reset only when bar signal==0 is encountered
    - enter only when signal>0, not at last bar
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
        do_trade = 0.0  # mặc định không giao dịch bar này

        # ====== QUYẾT ĐỊNH TRADE THEO LOGIC GỐC ======
        if in_pos == 0:
            # mở lệnh: signal>0, không ở bar cuối, và không bị freeze
            if (signal[i] > 0.0) and (i != end_force_close_idx) and (not freeze):
                # qty = NAV_prev / px  (giống add_new_trade)
                do_trade = cur_nav / px if px > 0.0 else 0.0
                prev_qty = prev_qty + do_trade
                in_pos   = 1
                hold     = 0
                entry_px = px
        else:
            # đang có vị thế
            hold += 1
            if signal[i] == 0.0:
                # exit theo signal == 0 (gốc làm trước TP/SL)
                do_trade = -prev_qty
                prev_qty = 0.0
                in_pos   = 0
                freeze   = False  # gốc reset freeze ở nhánh này
            else:
                # kiểm tra TP/SL/MaxHolding
                tp_hit = px >= entry_px * tp_mult
                sl_hit = px <= entry_px * sl_mult
                mh_hit = (hold == max_holding_limit)  # gốc: đóng khi duration == limit
                force_close = (i == end_force_close_idx)
                if tp_hit or sl_hit or mh_hit or force_close:
                    do_trade = -prev_qty
                    prev_qty = 0.0
                    in_pos   = 0
                    # gốc: monitor_open_positions đặt freeze=True khi TP/SL/MaxHolding/last
                    freeze   = True

        # ====== STEP-WISE MTM & NAV GIỐNG update_port_history ======
        # PnL mark-to-market trên vị thế T-1
        if i == 0:
            prev_px = px
            pnl = 0.0
            prev_nav = cur_nav
        else:
            prev_px = close[i-1]
            prev_nav = nav[i-1]
            pnl = (px - prev_px) * (qty[i-1])  # qty T-1

        # phí giao dịch của bar này (nếu có)
        tcost = -abs(do_trade) * px * tcost_rate

        # cập nhật mảng
        trade[i]     = do_trade
        qty[i]       = (qty[i-1] if i > 0 else 0.0) + do_trade
        pnl_raw[i]   = pnl
        tcost_arr[i] = tcost
        pnl_tcost[i] = pnl + tcost
        nav[i]       = prev_nav + pnl_tcost[i]
        cur_nav      = nav[i]

        # quy ước reset freeze: chỉ reset ở nhánh signal==0 (đã làm ở trên)
        # nếu muốn reset khi signal về 0 kể cả không có vị thế, giữ nguyên như gốc:
        if (in_pos == 0) and (signal[i] == 0.0):
            freeze = False

    return qty, trade, pnl_raw, tcost_arr, pnl_tcost, nav


def _reconstruct_trades_from_series(df, trade_col, close_col, tcost_rate, tp_mult, sl_mult):
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


class BackTestSA:
    def __init__(self, parameters):
        # --- GÁN THUỘC TÍNH TRƯỚC ---
        # start_date nên là Timestamp để lọc dễ
        self.start_date = pd.to_datetime(parameters['start_date'])

        # asset / timeframe / NAV / ...
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

        # --- SAU ĐÓ MỚI LOAD DỮ LIỆU ---
        self.load_data(parameters['datafile_path'])
        self.load_extra_data(parameters['datafile_extra_path'])

        # special case of vertical barrier
        self.end_date = self.df.index.values[-1]

        strat_name = self.__class__.__name__
        if strat_name.startswith('Orderbook'):
            self.load_orderbook_data(parameters['orderbook_path'])

        self.trades_history = pd.DataFrame(columns=[
            'open','entry_date','qty','direction','entry_px','tp_px','sl_px',
            'duration','exit_date','exit_px','pnl_raw','tcost','pnl_tcost'
        ])

        if not strat_name.startswith('Orderbook'):
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
        raw_data = pd.read_csv(csv_path)
        raw_data['time_utc'] = pd.to_datetime(raw_data['time_utc'])
        raw_data['time_est'] = pd.to_datetime(raw_data['time_est'])
        raw_data = raw_data.set_index('time_utc').sort_index()
        raw_data = raw_data[['o','h','l','c','v','time_est']]
        raw_data[['o','h','l','c','v']] = raw_data[['o','h','l','c','v']].shift(1)
        raw_data = raw_data.iloc[1:]
        raw_data.columns = ['open','high','low','close','volume','time_est']
        # dùng lát cắt từ start_date trở đi, tránh .loc[exact] bị KeyError nếu có time component
        raw_data = raw_data.loc[self.start_date:]
        self.df = raw_data.copy()
        
        
    def load_extra_data(self, csv_path):
        '''
        load in csv extra price data file
        '''
        if csv_path is not None:
            raw_data = pd.read_csv(csv_path)
            raw_data['time_utc'] = pd.to_datetime(raw_data['time_utc'])
            raw_data['time_est'] = pd.to_datetime(raw_data['time_est'])
            # shift one to make sure close price happens on the timestamp
            raw_data = raw_data.set_index('time_utc').sort_index()
            raw_data = raw_data[['o','h','l','c','v','time_est']]
            raw_data[['o','h','l','c','v']] = raw_data[['o','h','l','c','v']].shift(1)
            raw_data = raw_data.iloc[1:]
            raw_data.columns = ['open','high','low','close','volume','time_est']
            raw_data = raw_data.loc[self.start_date:]
            self.df_extra = raw_data.copy()
    
            
    def load_orderbook_data(self, csv_path):
        '''
        load in csv order book data file
        '''
        if csv_path is not None:
            raw_data = pd.read_csv(csv_path)
            raw_data['minute'] = pd.to_datetime(raw_data['minute'])
            raw_data = raw_data.set_index('minute').sort_index()

            self.df_orderbook = raw_data.copy()
        

    def add_new_trade(self, df_row):
        '''
        add a new trade record to open_trades
        '''
        amt = self.cur_nav / self.max_no_of_trades
        price = df_row.close
        qty = amt * df_row.signal / price
        
        trade = {
            'open': 1,
            'entry_date': df_row.time_est,
            'qty': qty,
            'direction': df_row.signal,
            'entry_px': price,
            'tp_px': price * self.tp_mult,
            'sl_px': price * self.sl_mult,
            'duration': 0,
            'exit_date': None,
            'exit_px': None,
            'pnl_raw': 0,
            'tcost': 0,
            'pnl_tcost': 0
            }
        self.trades_history.loc[len(self.trades_history)] = trade
        
        # update trade column in port_history
        self.update_trade(qty, df_row)
        
        
    def close_positions(self, trades_list, df_row):
        '''
        close current positions
        '''
        for idx in trades_list:
            exit_date = df_row.time_est
            exit_px = df_row.close
            qty = self.trades_history.loc[idx,'qty']
            entry_px = self.trades_history.loc[idx,'entry_px']
            
            self.trades_history.loc[idx, 'open'] = 0
            self.trades_history.loc[idx, 'exit_date'] = exit_date
            self.trades_history.loc[idx, 'exit_px'] = exit_px
            
            # compute realized pnl
            pnl_raw = (exit_px - entry_px) * qty
            tcost = (entry_px + exit_px) * qty * self.tcost
            pnl_tcost = pnl_raw - tcost
            self.trades_history.loc[idx, 'pnl_raw'] = pnl_raw
            self.trades_history.loc[idx, 'tcost'] = tcost
            self.trades_history.loc[idx, 'pnl_tcost'] = pnl_tcost
            
            # update trade column in port_history
            self.update_trade(-qty, df_row)
            
    
    def update_trade(self, qty, df_row):
        '''
        update trade and qty columns in port_history
        '''
        self.port_history.loc[df_row.Index, 'trade'] += qty
        
        
    def update_port_history(self, df_row):
        '''
        update port_history
        '''
        loc = self.df.index.get_loc(df_row.Index)
        trade = self.port_history.trade.iloc[loc]
        curr_px = self.df.close.iloc[loc]
        
        if loc == 0: # enter on first timestamp
            prev_qty = 0
            pnl_raw = 0
            prev_nav = self.cur_nav
        else:
            prev_qty = self.port_history.qty.iloc[loc - 1]
            prev_px = self.df.close.iloc[loc - 1]
            prev_nav = self.port_history.nav.iloc[loc - 1]
            pnl_raw = (curr_px - prev_px) * prev_qty
        
        tcost = -abs(trade) * curr_px * self.tcost
        pnl_tcost = pnl_raw + tcost
        nav = prev_nav + pnl_tcost
        
        self.port_history.loc[df_row.Index, 'qty'] = prev_qty + trade
        self.port_history.loc[df_row.Index, 'pnl_raw'] = pnl_raw
        self.port_history.loc[df_row.Index, 'tcost'] = tcost
        self.port_history.loc[df_row.Index, 'pnl_tcost'] = pnl_tcost
        self.port_history.loc[df_row.Index, 'nav'] = nav
        self.cur_nav = nav

        
    def count_open_pos(self):
        '''
        count number of current open positions
        '''
        return (self.trades_history['open'] == 1).sum()
    
    
    def get_open_pos_list(self):
        '''
        get index of open trades
        '''
        return self.trades_history.index[self.trades_history.open == 1].tolist()
    
    
    def get_open_pos(self):
        '''
        get open trades
        '''
        return self.trades_history.loc[self.trades_history['open'] == 1]
        
    
    def generate_signals(self):
        '''
        use this function to make sure generate signals has been included in the child class
        '''
        if 'signal' not in self.df.columns:
            raise Exception('You have not created signals yet')


    def monitor_open_positions(self, df_row):
        '''
        monitor open positions
        '''
        open_trades = self.get_open_pos()
        for trade in open_trades.itertuples():
            # check take profit and stop loss
            if df_row.close >= trade.tp_px or df_row.close <= trade.sl_px:
                self.close_positions([trade.Index], df_row)
                self.freeze = True
            # check if reach max holding
            elif trade.duration == self.max_holding_limit:
                self.close_positions([trade.Index], df_row)
                self.freeze = True
        
        # close all positions on last timestamp    
        if df_row.Index == self.end_date:
            self.close_positions(self.get_open_pos_list(), df_row)
            self.freeze = True
                
                
    def run_backtest(self):
        # signals generated from child class
        self.generate_signals()

        # loop over dataframe
        for row in self.df.itertuples():
             
            print(row.time_est)
            
            # no open position
            if self.in_trade is False:
                # open a long
                if row.signal > 0:
                    if row.Index != self.end_date:
                        self.add_new_trade(row)
                        self.in_trade = True
                # no signal
                else:
                    self.port_history.loc[row.Index, 'nav'] = self.cur_nav        
            # in trade
            else:
                # add duration
                self.trades_history.loc[self.trades_history['open'] == 1,'duration'] += 1
                
                # close trades
                if row.signal == 0:
                    self.close_positions(self.get_open_pos_list(), row)
                    self.in_trade = False
                    self.freeze = False
                else:
                    # check TP, SL and Max Holding
                    self.monitor_open_positions(row)
                    # additional signal
                    if row.signal > 0:
                        # check if we can add new trade
                        if not self.freeze and self.count_open_pos() < self.max_no_of_trades:
                            self.add_new_trade(row)         
            
            # update port history
            self.update_port_history(row)
                        
    def run_backtest_fast(self):
        # 1) build signal
        self.generate_signals()
        df = self.df
        if 'signal' not in df.columns:
            raise Exception('You have not created signals yet')

        close  = df['close'].to_numpy(dtype=np.float64)
        signal = df['signal'].to_numpy(dtype=np.float64)
        n = close.shape[0]
        if n == 0:
            raise ValueError("Empty data")

        # 2) params
        initial_nav   = float(self.cur_nav)
        tp_mult       = float(self.tp_mult)
        sl_mult       = float(self.sl_mult)
        max_holding   = int(self.max_holding_limit)
        tcost_rate    = float(self.tcost)
        end_force_idx = n - 1

        if int(self.max_no_of_trades) != 1:
            print("[FAST] max_no_of_trades > 1 → fallback run_backtest() cũ.")
            return self.run_backtest()

        # 3) kernel step-wise
        qty, trade, pnl_raw, tcost_arr, pnl_tcost, nav = _bt_kernel_singlepos_stepwise(
            close, signal, tp_mult, sl_mult, max_holding, tcost_rate,
            initial_nav, end_force_idx
        )

        # 4) port_history
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

        # 5)  trades_history from vector trade 
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


    def save_backtest_fast(self, out_dir: str = None):
        out_dir = out_dir or BACKTEST_DATA_DIR
        os.makedirs(out_dir, exist_ok=True)

        strat_name = self.__class__.__name__
        asset_name = self.asset
        time_frame = self.timeframe

        trades_history_file = os.path.join(out_dir, f"{strat_name}_{asset_name}_{time_frame}_trades_history.csv")
        port_history_file   = os.path.join(out_dir, f"{strat_name}_{asset_name}_{time_frame}_portfolio_history.csv")

        if isinstance(self.trades_history, pd.DataFrame) and len(self.trades_history):
            self.trades_history.to_csv(trades_history_file, index=False)
        else:
            pd.DataFrame(columns=['open','entry_date','qty','direction','entry_px','tp_px','sl_px',
                                  'duration','exit_date','exit_px','pnl_raw','tcost','pnl_tcost']).to_csv(trades_history_file, index=False)

        self.port_history.to_csv(port_history_file)                                                  
    def show_performace(self):
        
        plt.style.use('ggplot')
        self.port_history['nav'].plot()
        plt.title("Strategy results")
        plt.show()

    def save_backtest(self):
        
        strat_name = self.__class__.__name__
        asset_name = self.asset
        time_frame = self.timeframe
        
        self.trades_history.to_csv(f"backtest_results_coinlion_data/{strat_name}_{asset_name}_{time_frame}_trades_history.csv", index=False)
        self.port_history.to_csv(f"backtest_results_coinlion_data/{strat_name}_{asset_name}_{time_frame}_portfolio_history.csv")

    def save_backtest_fast_regime(self, out_dir: str = None):
        out_dir = out_dir or BACKTEST_DATA_DIR
        os.makedirs(out_dir, exist_ok=True)

        strat_name = self.__class__.__name__
        asset_name = self.asset
        time_frame = self.timeframe

        # Save trades history file
        trades_history_file = os.path.join(out_dir, f"{strat_name}_{asset_name}_{time_frame}_trades_history.csv")
        if isinstance(self.trades_history, pd.DataFrame) and len(self.trades_history):
            self.trades_history.to_csv(trades_history_file, index=False)
        else:
            pd.DataFrame(columns=['open','entry_date','qty','direction','entry_px','tp_px','sl_px',
                                'duration','exit_date','exit_px','pnl_raw','tcost','pnl_tcost']).to_csv(trades_history_file, index=False)

        # Save portfolio history file
        port_history_file = os.path.join(out_dir, f"{strat_name}_{asset_name}_{time_frame}_portfolio_history.csv")
        self.port_history.to_csv(port_history_file)

        # Process portfolio history for annual returns
        df1 = self.port_history
        # Ensure index is datetime
        df1.index = pd.to_datetime(df1.index)
        annual = df1['nav'].resample('Y').agg(['first', 'last'])
        annual['return_%'] = (annual['last'] / annual['first'] - 1) * 100
        annual.index = annual.index.year
        annual.to_csv(f"{out_dir}/short_{strat_name}_{asset_name}_{time_frame}_annual_returns.csv")

        # Define market regimes (Example regimes, you can adjust as needed)
        regimes = pd.DataFrame({
            "Regime": ["Bull", "Bull", "Bull", "Bear", "Bear", "Bear", "Sideway"],
            "Start": [
                "2023-09-27", "2024-09-06", "2025-04-08",
                "2024-06-05", "2025-01-21", "2025-10-06",
                "2023-02-13"
            ],
            "End": [
                "2024-03-13", "2024-12-06", "2025-10-06",
                "2024-08-05", "2025-03-10", "2025-11-21",
                "2023-09-15"
            ],
        })

        # Convert to datetime
        regimes["Start"] = pd.to_datetime(regimes["Start"])
        regimes["End"] = pd.to_datetime(regimes["End"])

        # Path to the coin regime return file
        coin_filename = f"D:/Coinlion_backtest/coin_regimes/{asset_name.lower()}usd_1d__{asset_name}USD_regime_returns.csv"
        
        # Kiểm tra nếu file tồn tại
        if os.path.exists(coin_filename):
            # Đọc dữ liệu từ file
            coin_regime_returns = pd.read_csv(coin_filename)
            
            # Thêm cột 'coin return' từ cột 'Return_%' trong file
            regimes["Coin Return"] = coin_regime_returns["Return_%"]
        else:
            print(f"Warning: File for {asset_name} coin return not found. File: {coin_filename}")

        # Tính toán "Strategy Return" (dựa trên NAV trong giai đoạn Start-End)
        strategy_returns = []
        for _, row in regimes.iterrows():
            start, end = row["Start"], row["End"]
            
            # Lọc NAV trong giai đoạn Start-End cho chiến lược
            period = df1.loc[start:end]

            # Tính toán chiến lược return
            if len(period) == 0:
                strat_ret = None
            else:
                first_nav = period["nav"].iloc[0]
                last_nav = period["nav"].iloc[-1]
                strat_ret = (last_nav / first_nav - 1) * 100  # Lợi nhuận chiến lược

            strategy_returns.append(strat_ret)

        regimes["Strategy Return"] = strategy_returns

        # In bảng regimes returns ra console
        print("\n=== Regimes Returns ===")
        print(regimes)

        # Lưu vào file CSV
        regimes.to_csv(f"{out_dir}/short_{strat_name}_{asset_name}_{time_frame}_regime_returns.csv", index=False)

class TrendFollowingEMAADX(BackTestSA):

    def __init__(self, parameters):
        super().__init__(parameters)

        # Dù parameters có gì đi nữa, vẫn ép giá trị cố định
        self.ema_short_period = 150
        self.ema_long_period = 500
        self.adx_period = 7


    def generate_signals(self):
        df = self.df.copy()

        # Tính EMA
        df['ema_short'] = EMAIndicator(close=df['close'], window=self.ema_short_period).ema_indicator()
        df['ema_long'] = EMAIndicator(close=df['close'], window=self.ema_long_period).ema_indicator()

        # Tính ADX
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=self.adx_period)
        df['adx'] = adx.adx()

        # Logic trend following
        trend_up = (df['ema_short'] > df['ema_long']) & (df['adx'] > 30)
     
        # Vào và thoát lệnh
        long_condition = trend_up
        exit_condition = (df['adx'] < 20) | (df['ema_short'] < df['ema_long']) 

        # Tạo cột signal
        df['signal'] = np.nan
        df.loc[long_condition, 'signal'] = 1
        df.loc[exit_condition, 'signal'] = 0

        # Duy trì trạng thái lệnh
        df['signal'] = df['signal'].ffill().fillna(0)

        self.df = df
    
class BuyandHold(BackTestSA):

    def __init__(self, parameters):
        super().__init__(parameters)

        # Dù parameters có gì đi nữa, vẫn ép giá trị cố định
       

    def generate_signals(self):
        df = self.df.copy()

        # Tính EMA
        df['signal'] = 1            

        # Duy trì trạng thái lệnh

        self.df = df

class MeanReversionRSIEMA(BackTestSA):

    def __init__(self, parameters):
        super().__init__(parameters)

        # Có thể gán từ parameters nếu cần, nhưng ở đây ép cố định như ADX version
        self.rsi_period = 14
        self.ema_short_period = 1000
        self.ema_long_period = 4700

    def generate_signals(self):
        df = self.df.copy()

        # Tính RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=self.rsi_period).rsi()

        # Tính EMA
        df['ema_short'] = EMAIndicator(close=df['close'], window=self.ema_short_period).ema_indicator()
        df['ema_long'] = EMAIndicator(close=df['close'], window=self.ema_long_period).ema_indicator()

        # Điều kiện vào lệnh (Mean Reversion + xác nhận EMA tăng)
        long_condition = (df['rsi'] < 30) & (df['ema_short'] > df['ema_long'])

        # Điều kiện thoát lệnh
        exit_condition =  (df['ema_short'] < df['ema_long']) | (df['rsi'] > 55)

        # Tạo cột tín hiệu
        df['signal'] = np.nan
        df.loc[long_condition, 'signal'] = 1
        df.loc[exit_condition, 'signal'] = 0

        # Duy trì trạng thái lệnh
        df['signal'] = df['signal'].ffill().fillna(0)

        self.df = df

class MeanReversionStochEMAV2(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Tham số mặc định (có thể được tối ưu bằng Optuna)
        self.ema_short_period = 50
        self.ema_long_period = 300
        self.stoch_k_period = 10
        self.stoch_d_period = 80
        self.stoch_buy_th = 10
        self.stoch_sell_th = 85

    def generate_signals(self):
        df = self.df.copy()

        # --- Tính EMA ---
        df["ema_short"] = EMAIndicator(close=df["close"], window=self.ema_short_period).ema_indicator()
        df["ema_long"] = EMAIndicator(close=df["close"], window=self.ema_long_period).ema_indicator()

        # --- Tính Stochastic ---
        stoch = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.stoch_k_period,
            smooth_window=self.stoch_d_period
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # --- Điều kiện vào/ra lệnh ---
        long_condition = (df["stoch_k"] < self.stoch_buy_th) & (df["ema_short"] > df["ema_long"])
        exit_condition = (df["stoch_k"] > self.stoch_sell_th) | (df["ema_short"] < df["ema_long"])

        # --- Tạo tín hiệu ---
        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0

        # --- Duy trì trạng thái lệnh ---
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class LongOnlyRangeRSIStochADX(BackTestSA):
   
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # --- Tham số mặc định ---
        self.rsi_period = 15
        self.rsi_entry_threshold = 40
        self.rsi_exit_threshold = 70
        self.stoch_k_period = 185
        self.stoch_d_period = 52
        self.stoch_k_entry_threshold = 25
        self.stoch_k_exit_threshold = 65
        self.adx_period = 65
        self.adx_threshold = 35
        self.support_resist_window = 40


        if parameters:
            for key, value in parameters.items():
                setattr(self, key, value)

    def generate_signals(self):

        df = self.df.copy()

        # --- Indicators ---
        df["rsi"] = RSIIndicator(close=df["close"], window=self.rsi_period).rsi()

        stoch = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.stoch_k_period,
            smooth_window=self.stoch_d_period
        )
        df["stoch_k"] = stoch.stoch()

        adx = ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.adx_period
        )
        df["adx"] = adx.adx()

        # Support / Resistance (không dùng trực tiếp trong logic, nhưng giữ lại để bạn có thể mở rộng)
        df["support"] = df["low"].rolling(window=self.support_resist_window).min()
        df["resistance"] = df["high"].rolling(window=self.support_resist_window).max()

        # Điền giá trị thiếu
        df.fillna(method="bfill", inplace=True)

        long_condition = (
            (df["rsi"] < self.rsi_entry_threshold) &
            (df["stoch_k"] < self.stoch_k_entry_threshold)
        ) | (df["adx"] < self.adx_threshold)

        exit_condition = (
            (df["rsi"] > self.rsi_exit_threshold) &
            (df["stoch_k"] > self.stoch_k_exit_threshold)
        )

        # --- Sinh tín hiệu ---
        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1

        # Chỉ thoát nếu đang nắm vị thế Long
        df.loc[(df["signal"].ffill().shift(1) == 1) & exit_condition, "signal"] = 0

        # Điền nốt các giá trị NaN còn lại
        df["signal"] = df["signal"].ffill().fillna(0)

        # Cập nhật lại self.df
        self.df = df

class LongOnlyTripleEMAMACDADX(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.ema_short_period = 60
        self.ema_medium_period = 200  
        self.ema_long_period = 2300 
        self.adx_period = 10      
        self.adx_threshold = 20    
        self.macd_fast = 120        
        self.macd_slow = 260        
        self.macd_signal = 90       

        # Ghi đè các tham số mặc định nếu chúng được truyền qua dictionary parameters
        if parameters:
            for key, value in parameters.items():
                setattr(self, key, value)

    def generate_signals(self):
        # Sử dụng self.df là DataFrame của tài sản hiện tại từ BackTestSA
        df = self.df.copy()

        # --- EMA ---
        df["ema_short"] = EMAIndicator(close=df["close"], window=self.ema_short_period).ema_indicator()
        df["ema_medium"] = EMAIndicator(close=df["close"], window=self.ema_medium_period).ema_indicator() # Thêm lại EMA medium
        df["ema_long"] = EMAIndicator(close=df["close"], window=self.ema_long_period).ema_indicator()

        # --- ADX ---
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=self.adx_period)
        df['adx'] = adx.adx()

        # --- MACD ---
        macd = MACD(close=df["close"],window_fast=self.macd_fast,window_slow=self.macd_slow,window_sign=self.macd_signal)
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        long_condition = ((df["ema_short"] > df["ema_medium"]) & (df["ema_medium"] > df["ema_long"]) &  
            (df["adx"] > self.adx_threshold) & (df["macd_line"] > df["macd_signal"]))

        exit_condition = (df["ema_short"] < df["ema_medium"]) | (df["adx"] < self.adx_threshold)        

        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[(df["signal"].ffill().shift(1) == 1) & exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0) # Điền các giá trị NaN và các giá trị đầu tiên

        self.df = df # Cập nhật lại self.df với cột tín hiệu

class LongOnlyEMAMACDADX(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.ema_short_period = 100
        self.ema_long_period  = 4100
        self.adx_period       = 14
        self.adx_threshold    = 25
        self.macd_fast        = 12
        self.macd_slow        = 26
        self.macd_signal      = 9
   
    def generate_signals(self):
        df = self.df.copy()

        df["ema_short"] = EMAIndicator(close=df["close"], window=self.ema_short_period).ema_indicator()
        df["ema_long"]  = EMAIndicator(close=df["close"], window=self.ema_long_period).ema_indicator()

        macd = MACD(close=df["close"],window_fast=self.macd_fast,window_slow=self.macd_slow,window_sign=self.macd_signal)
        df["macd_line"]   = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=self.adx_period)
        df['adx'] = adx.adx()

        # --- Logic long-only ---
        long_condition = (df["macd_line"] > df["macd_signal"])

        exit_condition = (df["macd_line"] < df["macd_signal"]) 

        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class LongOnlyAOROCSMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # 🧱 Tham số cố định (hardcoded) - Sẽ được ghi đè nếu 'parameters' được cung cấp
        self.sma_long_period = 400
        self.ao_short_period = 10
        self.ao_long_period = 55
        self.roc_period = 50
        self.roc_threshold = 0

    def generate_signals(self):
        df = self.df.copy()

        # --- SMA ---
        df["sma_long"] = SMAIndicator(close=df["close"], window=self.sma_long_period).sma_indicator()

        # --- Awesome Oscillator ---
        ao = AwesomeOscillatorIndicator(
            high=df["high"], low=df["low"],
            window1=self.ao_short_period,
            window2=self.ao_long_period
        )
        df["ao"] = ao.awesome_oscillator()

        # --- Rate of Change ---
        roc = ROCIndicator(close=df["close"], window=self.roc_period)
        df["roc"] = roc.roc()

        # --- Logic long-only ---
        long_condition = (df["close"] > df["sma_long"]) & \
                         (df["ao"] > 0) & \
                         (df["roc"] > 0)


        exit_condition = (df["close"] < df["sma_long"]) | \
                         (df["ao"] < 0) | \
                         (df["roc"] < 0)

        # --- Tín hiệu giao dịch ---
        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1  # Mở lệnh Long
        df.loc[exit_condition, "signal"] = 0  # Đóng lệnh

        # Điền các giá trị NaN theo chiều forward (giữ nguyên vị thế)
        # và fillna(0) cho các giá trị ban đầu chưa có tín hiệu
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class MeanReversionBBEMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Tham số tối ưu từ Optuna
        self.ema_period = 400
        self.bb_window = 20
        self.bb_dev = 2.5

    def generate_signals(self):
        df = self.df.copy()

        # Tính EMA dài
        df['ema'] = EMAIndicator(close=df['close'], window=self.ema_period).ema_indicator()

        # Tính Bollinger Bands
        bb = BollingerBands(close=df['close'], window=self.bb_window, window_dev=self.bb_dev)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()

        long_condition = (df["close"] < df["bb_high"]) & (df["close"] > df["ema"])

        exit_condition = (df["close"] > df["bb_low"]) |  (df["close"] < df["ema"])
        # & (df["close"] < df["ema"])

        # Tạo tín hiệu
        df['signal'] = np.nan
        df.loc[long_condition, 'signal'] = 1
        df.loc[exit_condition, 'signal'] = 0

        # Duy trì trạng thái lệnh
        df['signal'] = df['signal'].ffill().fillna(0)

        self.df = df

class ShortOnlyRSIEMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.ema_period = 160
        self.rsi_period = 8

    def generate_signals(self):
        df = self.df.copy()

        # --- Chỉ báo kỹ thuật ---
        df["ema"] = EMAIndicator(close=df["close"],
                                 window=self.ema_period).ema_indicator()
        df["rsi"] = RSIIndicator(close=df["close"],
                                 window=self.rsi_period).rsi()

        # --- Logic short ---
        short_condition = (df["close"] < df["ema"]) & (df["rsi"] > 75)
        exit_condition = (df["rsi"] < 60) | (df["close"] > df["ema"])

        # --- Tín hiệu giao dịch ---
        df["signal"] = np.nan
        df.loc[short_condition, "signal"] = -1
        df.loc[exit_condition, "signal"] = 0

        # Duy trì vị thế
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class RSIEMAMACD(BackTestSA):


    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Fix cứng các tham số (best parameters)
        self.ema_short_period = 80
        self.ema_long_period  = 400
        self.rsi_period       = 7
        self.rsi_buy_th       = 25
        self.rsi_sell_th      = 55
        self.macd_fast        = 18
        self.macd_slow        = 41
        self.macd_signal      = 15
        self.stop_loss_pct    = 0.05

    def generate_signals(self):
        df = self.df.copy()

        # Indicators
        df["ema_short"] = EMAIndicator(close=df["close"], window=self.ema_short_period).ema_indicator()
        df["ema_long"]  = EMAIndicator(close=df["close"], window=self.ema_long_period).ema_indicator()
        df["rsi"]       = RSIIndicator(close=df["close"], window=self.rsi_period).rsi()

        macd = MACD(
            close=df["close"],
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        # Entry / Exit conditions
        long_condition = (
            (df["ema_short"] > df["ema_long"]) &
            ((df["macd"] > df["macd_signal"]) | (df["rsi"] < self.rsi_buy_th))
        )
        exit_condition = (
            (df["ema_short"] < df["ema_long"]) &
            (df["macd"] < df["macd_signal"]) |
            (df["rsi"] > self.rsi_sell_th)
        )

        df['signal'] = np.nan
        entry_price = None

        for t in df.index:
            if long_condition.loc[t]:
                if df['signal'].ffill().iloc[-1] != 0.5:  # mở vị thế 0.5
                    entry_price = df.loc[t, 'close']
                df.loc[t, 'signal'] = 0.5
            elif exit_condition.loc[t]:
                df.loc[t, 'signal'] = 0
                entry_price = None
            elif df['signal'].ffill().iloc[-1] == 0.5 and entry_price is not None:
                # Check stop loss
                if df.loc[t, 'close'] <= entry_price * (1 - self.stop_loss_pct):
                    df.loc[t, 'signal'] = 0
                    entry_price = None

        # Fill forward signals
        df['signal'] = df['signal'].ffill().fillna(0)

        self.df = df

    def export_weights_to_csv(self, filepath):
        """
        Xuất cột 'signal' (trọng số) ra file CSV.
        """
        if 'signal' not in self.df.columns:
            raise ValueError("❌ Chưa có cột 'signal'. Hãy gọi generate_signals() trước.")

        weights = self.df[['signal']].copy()
        weights.index.name = 'time'
        weights.to_csv(filepath)
        print(f"📄 Đã lưu file trọng số weights tại: {filepath}")

class LongOnlyEMAMACDROC(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # 🧱 Tham số cố định (hardcoded)    
        self.macd_fast        = 24
        self.macd_slow        = 52
        self.macd_signal      = 18
        self.roc_period       = 30
        self.ema_short_period = 350
        self.ema_long_period  = 1900
    def generate_signals(self):
        df = self.df.copy()

        # --- EMA ---
        df["ema_short"] = EMAIndicator(close=df["close"], window=self.ema_short_period).ema_indicator()
        df["ema_long"]  = EMAIndicator(close=df["close"], window=self.ema_long_period).ema_indicator()

        macd_ind = MACD(
            close=df["close"],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        df["macd_hist"] = macd_ind.macd_diff()

        # --- ROC ---
        df["roc"] = ROCIndicator(close=df["close"], window=self.roc_period).roc()

        long_condition = (df["macd_hist"] > 0) | (df["ema_short"] > df["ema_long"]) 

        exit_condition = (df["ema_short"] < df["ema_long"]) & (df["macd_hist"] < 0) 

        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class MeanReversionRSIBBEMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters or {})

        # Gán tham số (có thể được tối ưu qua Optuna)
        self.bb_window   = parameters.get('bb_window', 10)
        self.bb_dev      = parameters.get('bb_dev', 2.5)
        self.rsi_period  = parameters.get('rsi_period', 28)
        self.rsi_buy_th  = parameters.get('rsi_buy_th', 35)
        self.rsi_sell_th = parameters.get('rsi_sell_th', 60)
        self.ema_period  = parameters.get('ema_period', 200)

    def generate_signals(self):
        df = self.df.copy()

        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=self.bb_window, window_dev=self.bb_dev)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low']  = bb.bollinger_lband()

        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=self.rsi_period).rsi()

        # EMA dài
        df['ema'] = EMAIndicator(close=df['close'], window=self.ema_period).ema_indicator()

        # --- Điều kiện vào lệnh ---
        long_condition = (df['close'] < df['bb_low']) & (df['rsi'] < self.rsi_buy_th)

        # --- Điều kiện thoát lệnh ---
        exit_condition = (df['rsi'] > self.rsi_sell_th) | (df["close"] > df["bb_high"])


        # Tạo cột tín hiệu
        df['signal'] = np.nan
        df.loc[long_condition, 'signal'] = 1
        df.loc[exit_condition, 'signal'] = 0

        # Duy trì trạng thái
        df['signal'] = df['signal'].ffill().fillna(0)
        self.df = df

class MeanReversionWilliamsEMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters or {})

        # Các tham số có thể tối ưu hoặc gán mặc định
        self.ema_period    = parameters.get('ema_period', 180)
        self.williams_period = parameters.get('williams_period', 13)
        self.will_buy_th   = parameters.get('will_buy_th', -65)
        self.will_sell_th  = parameters.get('will_sell_th', -20)

    def generate_signals(self):
        df = self.df.copy()

        # Tính EMA dài
        df['ema'] = EMAIndicator(close=df['close'], window=self.ema_period).ema_indicator()

        # Tính Williams %R
        df['williams_r'] = WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=self.williams_period
        ).williams_r()

        # --- Điều kiện vào lệnh ---
        long_condition = (df['williams_r'] < self.will_buy_th) & (df['close'] > df['ema'])

        # --- Điều kiện thoát lệnh ---
        exit_condition = (df['williams_r'] > self.will_sell_th) | (df['close'] < df['ema'])

        # Tạo tín hiệu
        df['signal'] = np.nan
        df.loc[long_condition, 'signal'] = 1
        df.loc[exit_condition, 'signal'] = 0

        # Duy trì trạng thái lệnh
        df['signal'] = df['signal'].ffill().fillna(0)

        self.df = df

class TrendFollowingEMAADXV2(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters or {})

        # Các tham số chiến lược
        self.ema_short  = parameters.get('ema_short', 10)
        self.ema_long   = parameters.get('ema_long', 481)
        self.adx_period = parameters.get('adx_period', 11)
        self.adx_th     = parameters.get('adx_th', 25)

    def generate_signals(self):
        df = self.df.copy()

        # EMA ngắn và dài
        df['ema_short'] = EMAIndicator(close=df['close'], window=self.ema_short).ema_indicator()
        df['ema_long']  = EMAIndicator(close=df['close'], window=self.ema_long).ema_indicator()

        # ADX
        df['adx'] = ADXIndicator(
            high=df['high'], low=df['low'], close=df['close'],
            window=self.adx_period
        ).adx()

        # --- Điều kiện vào lệnh ---
        long_condition = (df['ema_short'] > df['ema_long']) & (df['adx'] > self.adx_th)

        # --- Điều kiện thoát lệnh ---
        exit_condition = (df['ema_short'] < df['ema_long']) | (df['adx'] < self.adx_th)

        # Tạo tín hiệu
        df['signal'] = np.nan
        df.loc[long_condition, 'signal'] = 1
        df.loc[exit_condition, 'signal'] = 0

        # Duy trì trạng thái
        df['signal'] = df['signal'].ffill().fillna(0)

        self.df = df

class TrendFollowingKeltnerEMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters or {})

        # Tham số mặc định hoặc từ parameters
        self.ema_filter = parameters.get('ema_filter', 530)
        self.kc_window  = parameters.get('kc_window', 12)
        self.atr_mult   = parameters.get('atr_mult', 2)

    def generate_signals(self):
        df = self.df.copy()

        # EMA dài
        df['ema_filter'] = EMAIndicator(close=df['close'], window=self.ema_filter).ema_indicator()

        # Keltner Channel
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.kc_window).average_true_range()
        df['kc_mid']   = EMAIndicator(close=df['close'], window=self.kc_window).ema_indicator()
        df['kc_upper'] = df['kc_mid'] + self.atr_mult * atr
        df['kc_lower'] = df['kc_mid'] - self.atr_mult * atr

        # Điều kiện vào lệnh
        long_condition = (df['close'] > df['kc_upper']) & (df['close'] > df['ema_filter'])

        # Điều kiện thoát lệnh
        exit_condition = (df['close'] < df['ema_filter']) | (df['close'] < df['kc_mid'])

        # Tín hiệu
        df['signal'] = np.nan
        df.loc[long_condition, 'signal'] = 1
        df.loc[exit_condition, 'signal'] = 0

        # Duy trì trạng thái lệnh
        df['signal'] = df['signal'].ffill().fillna(0)

        self.df = df

class TrendFollowingIchimokuEMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters or {})
        self.parameters = parameters or {}

        self.ema_period = self.parameters.get("ema_period", 490)
        self.conversion_window = self.parameters.get("conversion_window", 15)
        self.base_window = self.parameters.get("base_window", 28)
        self.span_b_window = self.parameters.get("span_b_window", 59)

    def ichimoku(self, df):
        high_prices = df['high']
        low_prices = df['low']

        tenkan_sen = (high_prices.rolling(window=self.conversion_window).max() +
                      low_prices.rolling(window=self.conversion_window).min()) / 2
        kijun_sen = (high_prices.rolling(window=self.base_window).max() +
                     low_prices.rolling(window=self.base_window).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.base_window)
        senkou_span_b = ((high_prices.rolling(window=self.span_b_window).max() +
                          low_prices.rolling(window=self.span_b_window).min()) / 2).shift(self.base_window)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

    def generate_signals(self):
        df = self.df.copy()
        df["ema"] = EMAIndicator(close=df["close"], window=self.ema_period).ema_indicator()
        df["tenkan"], df["kijun"], df["senkou_a"], df["senkou_b"] = self.ichimoku(df)
        df["kumo_high"] = df[["senkou_a", "senkou_b"]].max(axis=1)
        df["kumo_low"]  = df[["senkou_a", "senkou_b"]].min(axis=1)

        long_condition = (
            (df["close"] > df["kumo_high"]) &
            (df["tenkan"] > df["kijun"]) &
            (df["close"] > df["ema"])
        )
        exit_condition = (df["close"] < df["ema"])

        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class TrendFollowingTEMAADX(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters or {})
        self.parameters = parameters or {}

        # Gán tham số mặc định hoặc từ parameters
        self.tema_period = self.parameters.get("tema_period", 470)
        self.adx_period  = self.parameters.get("adx_period", 10)
        self.adx_th      = self.parameters.get("adx_th", 29)

    @staticmethod
    def compute_tema(series, period):
        """Tính Triple EMA (TEMA)"""
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    def generate_signals(self):
        df = self.df.copy()

        # Tính TEMA
        df["tema"] = self.compute_tema(df["close"], self.tema_period)

        # Tính ADX
        adx_indicator = ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.adx_period
        )
        df["adx"] = adx_indicator.adx()

        # --- Điều kiện vào lệnh ---
        long_condition = (df["close"] > df["tema"]) & (df["adx"] > self.adx_th)

        # --- Điều kiện thoát lệnh ---
        exit_condition = (df["adx"] < self.adx_th) | (df["close"] < df["tema"])

        # Tín hiệu
        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class TrendFollowingBBEMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters or {})
        self.parameters = parameters or {}

        # Gán tham số mặc định hoặc lấy từ parameters
        self.ema_period = self.parameters.get("ema_period", 350)
        self.bb_window  = self.parameters.get("bb_window", 60)
        self.bb_dev     = self.parameters.get("bb_dev", 3)

    def generate_signals(self):
        df = self.df.copy()

        # Tính EMA dài
        df["ema"] = EMAIndicator(close=df["close"], window=self.ema_period).ema_indicator()

        # Tính Bollinger Bands
        bb = BollingerBands(
            close=df["close"],
            window=self.bb_window,
            window_dev=self.bb_dev
        )
        df["bb_upper"]  = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"]  = bb.bollinger_lband()

        # --- Điều kiện vào lệnh ---
        long_condition = (df["close"] > df["bb_upper"]) & (df["close"] > df["ema"])

        # --- Điều kiện thoát lệnh ---
        exit_condition = (df["close"] < df["bb_middle"]) | (df["close"] < df["ema"])

        # Tín hiệu giao dịch
        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class MeanReversionEMAZScore(BackTestSA):
    def __init__(self, parameters):
        super().__init__(parameters)

        # Gán tham số từ parameters
        self.ema_period = parameters.get("ema_period", 450)
        self.z_window   = parameters.get("z_window", 12)
        self.z_buy_th   = parameters.get("z_buy_th", -1.6)
        self.z_exit_th  = parameters.get("z_exit_th", -0.5)

    def generate_signals(self):
        df = self.df.copy()

        # EMA
        df["ema"] = EMAIndicator(close=df["close"], window=self.ema_period).ema_indicator()

        # Z-Score
        rolling_mean = df["close"].rolling(self.z_window).mean()
        rolling_std  = df["close"].rolling(self.z_window).std()
        df["zscore"] = (df["close"] - rolling_mean) / (rolling_std)

        # Điều kiện vào lệnh
        long_condition = (df["zscore"] < self.z_buy_th) & (df["close"] > df["ema"])

        # Điều kiện thoát lệnh
        exit_condition =  (df["close"] < df["ema"]) | (df["zscore"] >  self.z_exit_th)

        # Tín hiệu
        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0

        df["signal"] = df["signal"].ffill().fillna(0)
        self.df = df

class TrendFollowingOBVMFI(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.obv_ema_period = parameters.get("obv_ema", 20)
        self.mfi_window     = parameters.get("mfi_window", 10)
        self.mfi_threshold  = parameters.get("mfi_threshold", 60)

    def generate_signals(self):
        df = self.df.copy()

        # --- OBV ---
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
        df["obv"] = obv
        df["obv_ema"] = obv.ewm(span=self.obv_ema_period, adjust=False).mean()

        # --- MFI ---
        df["mfi"] = MFIIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            window=self.mfi_window
        ).money_flow_index()

        # --- Logic vào / thoát lệnh Long ---
        long_condition = (df["obv"] > df["obv_ema"]) & (df["mfi"] > self.mfi_threshold)
        exit_condition = (df["mfi"] < self.mfi_threshold) 

        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class TrendFollowingOBVEMA(BackTestSA):
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Tham số mặc định hoặc lấy từ dict
        self.price_ema_period = parameters.get("price_ema", 300)
        self.obv_ema_period   = parameters.get("obv_ema", 175)

    def generate_signals(self):
        df = self.df.copy()

        # --- EMA giá ---
        df["ema_price"] = EMAIndicator(close=df["close"], window=self.price_ema_period).ema_indicator()

        # --- OBV + EMA OBV ---
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
        df["obv"] = obv
        df["obv_ema"] = obv.ewm(span=self.obv_ema_period, adjust=False).mean()

        # --- Logic vào / thoát lệnh Long ---
        long_condition = (df["close"] > df["ema_price"]) & (df["obv"] > df["obv_ema"])
        exit_condition = (df["close"] < df["ema_price"]) | (df["obv"] < df["obv_ema"])

        df["signal"] = np.nan
        df.loc[long_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

def save_loaded_price_data(df, export_path):
    """
            Lưu lại dữ liệu giá đã load từ self.df thành CSV.
            """
    df_to_save = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df_to_save.index.name = 'time'
    df_to_save.to_csv(export_path)
    print(f"📁 Đã lưu dữ liệu giá tại: {export_path}")

def save_indicators_to_csv(df, asset, export_path):
    """
    Lưu các chỉ báo kỹ thuật (EMA ngắn/dài, ADX) vào file CSV.
    """
    indicator_df = df[['ema_short', 'ema_long', 'adx']].copy()
    indicator_df['asset'] = asset
    indicator_df.index.name = 'time'
    indicator_df.to_csv(export_path)
    print(f"📈 Đã lưu chỉ báo kỹ thuật tại: {export_path}")



class EMA_ATR_LongOnly(BackTestSA):
    """
    ENTRY: (EMA(4600) > Close) AND (ATR(65) > ATR(10))
    EXIT : (EMA(4600) < Close) AND (ATR(65) < ATR(10))
    Side : long_only
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        # Khóa tham số đúng theo idea
        self.ema_window = 4600
        self.atr_long = 65
        self.atr_short = 10

    def generate_signals(self):
        df = self.df.copy()

        # ===== Indicators =====
        df["ema"] = EMAIndicator(
            close=df["close"],
            window=self.ema_window
        ).ema_indicator()

        atr_long = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.atr_long
        )
        atr_short = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.atr_short
        )
        df["atr_long"] = atr_long.average_true_range()
        df["atr_short"] = atr_short.average_true_range()

        # ===== Conditions =====
        entry_condition = (
            (df["ema"] > df["close"]) &
            (df["atr_long"] > df["atr_short"])
        )

        exit_condition = (
            (df["ema"] < df["close"]) &
            (df["atr_long"] < df["atr_short"])
        )

        # ===== Signals (long_only) =====
        df["signal"] = np.nan
        df.loc[entry_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0

        # Duy trì trạng thái vị thế
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df

class EMA_ADX_LongOnly(BackTestSA):
    """
    ENTRY: (EMA(6000) < Close) AND (ADX(110) < 10)
    EXIT : (EMA(6000) > Close)
    Side : long_only
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        # Khóa tham số đúng theo idea
        self.ema_window = 6000
        self.adx_window = 110
        self.adx_threshold = 10

    def generate_signals(self):
        df = self.df.copy()

        # ===== Indicators =====
        df["ema"] = EMAIndicator(
            close=df["close"],
            window=self.ema_window
        ).ema_indicator()

        adx = ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.adx_window
        )
        df["adx"] = adx.adx()

        # ===== Conditions =====
        entry_condition = (
            (df["ema"] < df["close"]) &
            (df["adx"] < self.adx_threshold)
        )

        exit_condition = (
            (df["ema"] > df["close"])
        )

        # ===== Signals (long_only) =====
        df["signal"] = np.nan
        df.loc[entry_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0

        # Duy trì trạng thái vị thế
        df["signal"] = df["signal"].ffill().fillna(0)

        self.df = df
class ROC_AO_MACDSlope_ADX_LongOnly(BackTestSA):
    """
    ENTRY: (ROC(95) > -0.5) AND (AO(10/68) > 9) 
           AND (ΔMACD(240/520/180, 55 bars) > 3.55) AND (ADX(60) < 20)
    EXIT : (ROC(95) < -0.5) AND (ΔMACD(240/520/180, 55 bars) < 3.55)
    Side : long_only
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        # Khóa tham số đúng theo idea
        self.roc_period = 95
        self.roc_threshold = -0.5

        self.ao_fast = 10
        self.ao_slow = 68
        self.ao_threshold = 9.0

        self.macd_fast = 240
        self.macd_slow = 520
        self.macd_signal = 180
        self.macd_slope_threshold = 3.55
        self.macd_slope_window = 55  # số bar dùng để tính slope

        self.adx_window = 60
        self.adx_threshold = 20

    def generate_signals(self):
        df = self.df.copy()

        # ===== Indicators =====
        # ROC (Rate of Change)
        df["roc"] = df["close"].pct_change(self.roc_period) * 100

        # AO (Awesome Oscillator)
        ao_fast_ma = df["close"].rolling(self.ao_fast).mean()
        ao_slow_ma = df["close"].rolling(self.ao_slow).mean()
        df["ao"] = ao_fast_ma - ao_slow_ma

        # MACD + slope của MACD histogram
        ema_fast = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = macd_line - signal_line
        df["macd_slope"] = df["macd_hist"].diff(self.macd_slope_window)

        # ADX
        adx = ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.adx_window
        )
        df["adx"] = adx.adx()

        # ===== Conditions =====
        entry_condition = (
            (df["roc"] > self.roc_threshold) &
            (df["ao"] > self.ao_threshold) &
            (df["macd_slope"] > self.macd_slope_threshold) &
            (df["adx"] < self.adx_threshold)
        )

        exit_condition = (
            (df["roc"] < self.roc_threshold) &
            (df["macd_slope"] < self.macd_slope_threshold)
        )

        # ===== Signals (long_only) =====
        df["signal"] = np.nan
        df.loc[entry_condition, "signal"] = 1
        df.loc[exit_condition, "signal"] = 0

        # Duy trì trạng thái vị thế
        df["signal"] = df["signal"].ffill().fillna(0)
        self.df = df

def export_weights_to_csv(self, filepath):
    
    if 'signal' not in self.df.columns:
        raise ValueError("❌ Chưa có cột 'signal'. Hãy gọi generate_signals() trước.")

    weights = self.df[['signal']].copy()
    weights.index.name = 'time'  # Đặt tên index
    weights.to_csv(filepath)
    print(f"📄 Đã lưu file trọng số weights tại: {filepath}")
class DualTFStochMACDSimple(BackTestSA):
    def __init__(self, parameters: dict):
        # Kế thừa BackTestSA (đã có load_data, load_extra_data, ... )
        super().__init__(parameters)

        # tham số indicator mặc định
        self.stoch_lb    = 60
        self.entry_level = 15
        self.exit_level  = 70
        self.slow_period = 182
        self.fast_period = 84
        self.ema_period  = 63

    def generate_signals(self):
        df_1h  = self.df.copy()
        df_15m = self.df_extra.copy()

        # --- Stochastic (khung chậm) ---
        stochastic = StochasticOscillator(
            high=df_1h["high"], low=df_1h["low"], close=df_1h["close"], window=self.stoch_lb
        )
        df_1h["stoch"]   = stochastic.stoch_signal()
        df_1h["buy_1h"]  = (df_1h["stoch"] < self.entry_level).astype(float)
        df_1h["sell_1h"] = (df_1h["stoch"] > self.exit_level).astype(float)

        # --- MACD histogram (khung nhanh) ---
        macd = MACD(df_15m["close"],
                    window_slow=self.slow_period,
                    window_fast=self.fast_period,
                    window_sign=self.ema_period)
        df_15m["histogram"] = macd.macd_diff()
        df_15m["buy_15m"]   = (df_15m["histogram"] > 0).astype(float)

        # --- Merge ---
        df = pd.concat([df_15m, df_1h[["buy_1h","sell_1h"]]], axis=1).ffill()

        # --- Signal ---
        def _rule(row):
            if row["buy_15m"] == 1.0 and row["buy_1h"] == 1.0:
                return 1.0
            if row["sell_1h"] == 1.0:
                return 0.0
            return np.nan

        df["signal"] = df.apply(_rule, axis=1)

        # cập nhật lại self.df để backtest chạy
        self.df = df
        self.end_date = self.df.index.values[-1]
#%% main function       
if __name__ == '__main__':
    start_date = '2023-01-01'
    asset = 'BTC'  # Chọn tài sản cần backtest
    timeframe = '60m'
    filename = PRICE_DATA_DIR + asset.lower() + 'usd_' + timeframe + '.csv'
    
    timeframe_extra = '1h'
    if timeframe_extra in ['60m','30m','15m','5m']:
        filename_extra = PRICE_DATA_DIR + asset.lower() + 'usd_' + timeframe_extra + '.csv'
    else:
        filename_extra = None
     
    max_no_of_trades = 1
    initial_nav = 100
    tp_pct = 1
    sl_pct = 0.15
    max_holding = 1000000
    
    tcost_old = {
        'BTC': 0.0000,
        'ETH': 0.0000,
        'XRP': 0.0000,
        'BCH': 0.0000,
        'SOL': 0.0000,
        'ADA': 0.0000,
        'LTC': 0.0000,
        'LINK': 0.0000,
        'DOGE': 0.0000,
        'MATIC': 0.0000,
        'DOT': 0.0000,
        'SHIB': 0.0000,
        'AVAX': 0.0000,
        'ATOM': 0.0000,
        'UNI': 0.0000,
        'ICP': 0.0000,
        'NEAR': 0.0000,
        'IMX': 0.0000
        }
    
    tcost = {
        'BTC': 0.0016,
        'ETH': 0.0016,
        'XRP': 0.0021,
        'BCH': 0.0026,
        'SOL': 0.0021,
        'ADA': 0.0026,
        'LTC': 0.0051,
        'LINK': 0.0091
        }
    
    parameters = {
        'start_date': start_date,
        'asset': asset,
        'timeframe': timeframe,
        'datafile_path': filename,
        'datafile_extra_path': filename_extra,
        'max_no_of_trades': max_no_of_trades,
        'initial_nav': initial_nav,
        'tp_pct': tp_pct,
        'sl_pct': sl_pct,
        'max_holding': max_holding,
        'tcost': tcost[asset]
        }

    start_time = time.time()
    
    parameters['ema_period'] = 1000
    parameters['bb_window'] = 75
    parameters['bb_dev'] = 3

    ema = TrendFollowingBBEMA(parameters)
    ema.run_backtest_fast()
    ema.show_performace()
    ema.save_backtest_fast_regime()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)  
    
    # ✅ Lưu trọng số weights vào thư mục mới
    # export_path = f"backtest_results_coinlion_data/{asset}_MeanReversionRSIEMA_weights.csv"
    # ema.export_weights_to_csv(export_path)

    # price_export_path = f"backtest_results_coinlion_data/{asset}_MeanReversionRSIEMA_price_data.csv"
    # save_loaded_price_data(ema.df, price_export_path)

    # # ✅ Lưu các chỉ báo kỹ thuật EMA + ADX
    # indicator_export_path = f"backtest_results_coinlion_data/{asset}_MeanReversionRSIEMA_indicators.csv"
    # save_indicators_to_csv(ema.df, asset, indicator_export_path)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("Elapsed time: ", elapsed_time)


    
# def run_backtest(asset: str):
#     """Hàm chạy backtest cho 1 asset."""
#     start_date = '2025-01-01'
#     timeframe = '10m'
#     filename = PRICE_DATA_DIR + asset.lower() + 'usd_' + timeframe + '.csv'

#     timeframe_extra = '1h'
#     if timeframe_extra in ['60m','30m','15m','5m']:
#         filename_extra = PRICE_DATA_DIR + asset.lower() + 'usd_' + timeframe_extra + '.csv'
#     else:
#         filename_extra = None

#     max_no_of_trades = 1
#     initial_nav = 100
#     tp_pct = 100.00
#     sl_pct = 1
#     max_holding = 1000000

#     tcost_old = {
#         'BTC': 0.0000, 'ETH': 0.0000, 'XRP': 0.0000, 'BCH': 0.0000,
#         'SOL': 0.0000, 'ADA': 0.0000, 'LTC': 0.0000, 'LINK': 0.0000,
#         'DOGE': 0.0000, 'MATIC': 0.0000, 'DOT': 0.0000, 'SHIB': 0.0000,
#         'AVAX': 0.0000, 'ATOM': 0.0000, 'UNI': 0.0000, 'ICP': 0.0000,
#         'NEAR': 0.0000, 'IMX': 0.0000
#     }

#     parameters = {
#         'start_date': start_date,
#         'asset': asset,
#         'timeframe': timeframe,
#         'datafile_path': filename,
#         'datafile_extra_path': filename_extra,
#         'max_no_of_trades': max_no_of_trades,
#         'initial_nav': initial_nav,
#         'tp_pct': tp_pct,
#         'sl_pct': sl_pct,
#         'max_holding': max_holding,
#         'tcost': tcost_old[asset]
#     }

#     start_time = time.time()
#     print(f"[{asset}] bắt đầu backtest...")

#     strat = TrendFollowingBBEMA(parameters)
#     strat.run_backtest()
#     strat.show_performace()
#     strat.save_backtest()

#     elapsed = time.time() - start_time
#     print(f"[{asset}] xong, thời gian chạy: {elapsed:.2f} giây")
#     return asset, strat.port_history.tail(1)  # ví dụ trả về NAV cuối cùng


# if __name__ == '__main__':
#     assets = ["BTC", "ETH", "DOGE"]

#     # dùng Pool để chạy song song 3 asset
#     with Pool(processes=3) as pool:
#         results = pool.map(run_backtest, assets)

#     print("\n=== Tổng kết NAV cuối cùng ===")
#     for asset, nav in results:
#         print(asset, nav)
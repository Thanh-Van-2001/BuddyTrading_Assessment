import pandas as pd
import numpy as np
from general import *
from general.date import *
from general.finance import *
import warnings

# Tắt tất cả các cảnh báo FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

strategy_name = 'TrendFollowingBBEMA'
asset = 'BTC'
timeframe = '60m'

BACKTEST_DATA_DIR = 'backtest_results_coinlion_data/'

trades_history_file = BACKTEST_DATA_DIR + strategy_name + '_' + asset + '_' + timeframe + '_trades_history.csv'
port_history_file   = BACKTEST_DATA_DIR + strategy_name + '_' + asset + '_' + timeframe + '_portfolio_history.csv'

# --- Read CSV ---
trades_history_raw = pd.read_csv(trades_history_file)
port_history_raw   = pd.read_csv(port_history_file)

# --- Detect & set time index cho portfolio history ---
candidate_time_cols = ['time_utc', 'time_est', 'time_local', 'time', 'timestamp', 'datetime', 'Date', 'date']
time_col = next((c for c in candidate_time_cols if c in port_history_raw.columns), None)

if time_col is not None:
    port_history_raw[time_col] = pd.to_datetime(port_history_raw[time_col], errors='coerce')
    port_history_raw = port_history_raw.dropna(subset=[time_col])
    port_history_raw = port_history_raw.set_index(time_col)
else:
    # Trường hợp thời gian nằm ở cột index lưu thành 'Unnamed: 0' hoặc ngay index
    if 'Unnamed: 0' in port_history_raw.columns:
        port_history_raw['time'] = pd.to_datetime(port_history_raw['Unnamed: 0'], errors='coerce')
        port_history_raw = port_history_raw.dropna(subset=['time']).set_index('time')
    else:
        # cố gắng parse index
        port_history_raw.index = pd.to_datetime(port_history_raw.index, errors='coerce')
        if port_history_raw.index.isna().all():
            raise KeyError("Không tìm thấy cột thời gian trong portfolio_history CSV. Hãy kiểm tra header file.")

# --- Chuẩn hoá cột ngày trong trades_history trước khi lọc ---
for c in ['entry_date', 'exit_date']:
    if c in trades_history_raw.columns:
        trades_history_raw[c] = pd.to_datetime(trades_history_raw[c], errors='coerce')

# ====== Phần còn lại giữ nguyên (với vài phòng thủ nhỏ) ======
start_date = '2023-01-01'
pnl_column = 'pnl_tcost'
ret_column = 'ret_tcost'

port_history = port_history_raw.loc[start_date:]

trades_history = trades_history_raw.copy()
if 'entry_date' in trades_history.columns:
    trades_history = trades_history.loc[trades_history['entry_date'] >= pd.to_datetime(start_date)]
else:
    # nếu không có entry_date, bỏ lọc theo thời gian
    pass

# Tạo các cột ret nếu chưa có
if 'ret_raw' not in trades_history.columns:
    trades_history['ret_raw'] = trades_history['pnl_raw'] / (trades_history['qty'] * trades_history['entry_px'])
if 'ret_tcost' not in trades_history.columns:
    trades_history['ret_tcost'] = trades_history['pnl_tcost'] / (trades_history['qty'] * trades_history['entry_px'])

# duration (ngày)
if {'entry_date','exit_date'} <= set(trades_history.columns):
    trades_history['duration'] = (trades_history['exit_date'] - trades_history['entry_date']) / np.timedelta64(1, 'D')
else:
    trades_history['duration'] = np.nan

# ---- compute trades statistics ----
total_no = len(trades_history)
win_rate = (trades_history[pnl_column] >= 0).mean() if total_no else 0.0
loss_rate = (trades_history[pnl_column] < 0).mean() if total_no else 0.0
total_pnl = trades_history[pnl_column].sum()
avg_pnl   = trades_history[pnl_column].mean()
avg_ret   = trades_history[ret_column].mean()
avg_age   = trades_history['duration'].mean()

win_no  = (trades_history[pnl_column] >= 0).sum()
loss_no = (trades_history[pnl_column] < 0).sum()

avg_age_of_wins   = trades_history.loc[trades_history[pnl_column] >= 0, 'duration'].mean()
avg_age_of_losses = trades_history.loc[trades_history[pnl_column] < 0,  'duration'].mean()

total_win_amount = trades_history.loc[trades_history[pnl_column] >= 0, pnl_column].sum()
total_loss_amount= trades_history.loc[trades_history[pnl_column] <  0, pnl_column].sum()

avg_win_amount = trades_history.loc[trades_history[pnl_column] >= 0, pnl_column].mean()
avg_loss_amount= trades_history.loc[trades_history[pnl_column] <  0, pnl_column].mean()

avg_win_pct  = trades_history.loc[trades_history[pnl_column] >= 0, ret_column].mean()
avg_loss_pct = trades_history.loc[trades_history[pnl_column] <  0, ret_column].mean()

lowest_win   = trades_history.loc[trades_history[pnl_column] >= 0, ret_column].min()
lowest_loss  = trades_history.loc[trades_history[pnl_column] <  0, ret_column].max()
highest_win  = trades_history.loc[trades_history[pnl_column] >= 0, ret_column].max()
highest_loss = trades_history.loc[trades_history[pnl_column] <  0, ret_column].min()

trades_perf = {
    'total_trades': total_no,
    'total_pnl': "${:.2f}".format(total_pnl),
    'avg_pnl': "${:.2f}".format(avg_pnl),
    'avg_ret': "{:.2f}%".format(avg_ret*100),
    'avg_age': "{:.2f} days".format(avg_age),
    'win_rate': "{:.1f}%".format(win_rate*100),
    'loss_rate': "{:.1f}%".format(loss_rate*100),
    'no_of_wins': win_no,
    'no_of_losses': loss_no,
    'avg_age_of_wins': "{:.2f} days".format(avg_age_of_wins),
    'avg_age_of_losses': "{:.2f} days".format(avg_age_of_losses),
    'total_profits': "${:.2f}".format(total_win_amount),
    'total_losses': "${:.2f}".format(total_loss_amount),
    'avg_profit': "${:.2f}".format(avg_win_amount),
    'avg_loss': "${:.2f}".format(avg_loss_amount),
    'avg_profit_pct': "{:.2f}%".format(avg_win_pct*100),
    'avg_loss_pct': "{:.2f}%".format(avg_loss_pct*100),
    'lowest_profit_pct': "{:.2f}%".format(lowest_win*100),
    'lowest_loss_pct': "{:.2f}%".format(lowest_loss*100),
    'highest_profit_pct': "{:.2f}%".format(highest_win*100),
    'highest_loss_pct': "{:.2f}%".format(highest_loss*100),
}
df_perf = pd.Series(trades_perf)

# Equity curve & perf stats
nav_hist = port_history['nav'].rebase()
nav_daily = nav_hist.groupby(nav_hist.index.date).last()
nav_daily.index = pd.to_datetime(nav_daily.index)
nav_daily = nav_daily.to_frame(strategy_name)
perf_stats = nav_daily.idxstats()

if __name__ == "__main__":
    print("=== TRADE PERFORMANCE SUMMARY ===")
    print(df_perf)
    print("\n=== PERF STATS ===")
    print(perf_stats.round(2))
    print(nav_daily.return_table())

    import matplotlib.pyplot as plt
    nav_daily.plot(title="Equity Curve")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    # monthly_table = nav_daily.return_table()

    # monthly_table.to_clipboard(excel=True)
# Xuất sang clipboard
df1 = perf_stats.iloc[[0]]
df2 = pd.DataFrame([pd.Series(trades_perf).values])
combined = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
combined.to_clipboard(index=False, header=False)

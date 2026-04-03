# -*- coding: utf-8 -*-
"""
3-page PDF for BuddyTrading Assessment - maximally dense, no wasted space
"""
import os
from fpdf import FPDF

IMG_DIR = 'backtest_results_coinlion_data/'
OUT_PATH = 'BuddyTrading_Assessment_TrendFollowingBBEMA.pdf'


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 3, 'BuddyTrading - Quant Trader Assessment', align='R', new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(1)

    def footer(self):
        self.set_y(-8)
        self.set_font('Helvetica', 'I', 6.5)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', align='C')

    def sec(self, title):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(25, 25, 25)
        self.cell(0, 5.5, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(50, 120, 200)
        self.set_line_width(0.35)
        self.line(10, self.get_y(), 68, self.get_y())
        self.ln(1.5)

    def sub(self, title):
        self.set_font('Helvetica', 'B', 8.5)
        self.set_text_color(45, 45, 45)
        self.cell(0, 4.5, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def txt(self, text):
        self.set_font('Helvetica', '', 8)
        self.set_text_color(35, 35, 35)
        self.multi_cell(0, 3.6, text)
        self.ln(0.8)

    def blt(self, text):
        self.set_font('Helvetica', '', 8)
        self.set_text_color(35, 35, 35)
        x0 = self.get_x()
        self.cell(4, 3.6, '-')
        self.multi_cell(186, 3.6, text)
        self.set_x(x0)

    def tbl(self, headers, rows, cw=None, fs=7.5):
        if cw is None:
            cw = [190 / len(headers)] * len(headers)
        rh = 4.2
        self.set_font('Helvetica', 'B', fs)
        self.set_fill_color(50, 120, 200)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(cw[i], rh + 0.5, h, border=1, align='C', fill=True)
        self.ln()
        self.set_font('Helvetica', '', fs)
        self.set_text_color(35, 35, 35)
        for ri, row in enumerate(rows):
            self.set_fill_color(243, 247, 255) if ri % 2 == 0 else self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(cw[i], rh, str(val), border=1, align='C', fill=True)
            self.ln()
        self.ln(1)


def build():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=10)

    # ==================== PAGE 1 ====================
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(25, 25, 25)
    pdf.cell(0, 7, 'Trend Following: Bollinger Bands + EMA Filter', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 4.5, 'BTC/USD Perpetual Futures  |  Coinbase  |  60-Min  |  Long-Only  |  2x Leverage  |  2023-2025', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 1. Instrument
    pdf.sec('1. Market & Instrument')
    pdf.tbl(
        ['Exchange', 'Instrument', 'Mode', 'Direction', 'Timeframe'],
        [['Coinbase (Advanced)', 'BTC/USD Perp Futures', '2x Leverage', 'Long-only', '60 min']],
        cw=[36, 40, 40, 34, 40]
    )

    # 2. Hypothesis
    pdf.sec('2. Trading Hypothesis')
    pdf.txt(
        'This strategy exploits momentum persistence in Bitcoin. When BTC breaks above the upper '
        'Bollinger Band while remaining above a long-term EMA, it signals the onset of a strong '
        'directional move. Crypto markets exhibit pronounced trending regimes driven by herding, '
        'FOMO-driven inflows, and reflexive price-liquidity feedback loops. The combination of a '
        'volatility breakout (BB) with a slow trend filter (EMA) avoids mean-reversion traps and '
        'only enters when both short-term and long-term momentum align. The strategy is naturally '
        'selective (67 trades over 2.9 years), waiting for high-conviction setups only.'
    )

    # 3. Rules
    pdf.sec('3. Strategy Rules')
    pdf.sub('Indicators & Parameter Rationale')
    pdf.tbl(
        ['Indicator', 'Parameter', 'Rationale'],
        [
            ['EMA', '1000 bars (~42 days)', 'Captures multi-week trend; filters noise from daily volatility'],
            ['BB Window', '75 bars (~3 days)', 'Balances responsiveness vs false breakout frequency'],
            ['BB Std Dev', '3.0', 'High threshold = only enters on extreme breakouts (high conviction)'],
        ],
        cw=[30, 45, 115]
    )
    pdf.txt(
        'Parameters validated via Optuna (150 trials): top 10 trials all converged to ema=900-1000, '
        'bb_window=75, bb_dev=3.0, confirming these are in a robust optimum region, not an isolated peak.'
    )

    pdf.sub('Entry & Exit')
    pdf.tbl(
        ['', 'Condition'],
        [
            ['ENTRY (Long)', 'Close > BB Upper(75, 3.0)  AND  Close > EMA(1000)'],
            ['EXIT', 'Close < BB Middle(75-bar SMA)  OR  Close < EMA(1000)'],
        ],
        cw=[32, 158]
    )

    pdf.sub('Signal Logic (Pseudocode)')
    pdf.set_font('Courier', '', 7.5)
    pdf.set_text_color(35, 35, 35)
    pdf.set_fill_color(245, 245, 250)
    code = (
        "  ema      = EMA(close, 1000)\n"
        "  bb_upper = SMA(close, 75) + 3.0 * StdDev(close, 75)\n"
        "  bb_mid   = SMA(close, 75)\n"
        "  if close > bb_upper AND close > ema:  signal = LONG\n"
        "  if close < bb_mid   OR  close < ema:  signal = EXIT\n"
        "  signal is forward-filled (hold position until exit triggers)"
    )
    pdf.multi_cell(190, 3.5, code, fill=True)
    pdf.ln(1.5)

    pdf.sub('Position Sizing & Risk Management')
    pdf.tbl(
        ['Rule', 'Setting', 'Note'],
        [
            ['Position Size', '100% equity x 2', 'Notional = 2x equity per trade'],
            ['Stop-Loss', '15% from entry', 'On position; = 30% equity risk max'],
            ['Take-Profit', 'No cap', 'Exit via signal (best = +46.8%)'],
            ['Leverage', '2x', 'Perpetual futures, cross margin'],
            ['Funding Rate', '0.01% / 8h', 'Standard perp futures funding cost'],
        ],
        cw=[32, 33, 125]
    )

    # 4. Backtest Setup
    pdf.sec('4. Backtest Assumptions')
    pdf.tbl(
        ['Parameter', 'Value'],
        [
            ['Data Source', 'Coinbase BTC/USD OHLCV candles (60-min)'],
            ['Period', 'Jan 1, 2023 - Nov 16, 2025 (2.9 years, 25,220 bars)'],
            ['Trading Fees', '0.16% round-trip (taker fee per side)'],
            ['Funding Rate', '0.01% per 8 hours while in position'],
            ['Slippage', '0 assumed (hourly bars, BTC highly liquid)'],
            ['Execution', 'Filled at bar close; no partial fills'],
            ['Leverage', '2x (perpetual futures)'],
            ['Starting Capital', '$100'],
        ],
        cw=[35, 155]
    )

    # ==================== PAGE 2 ====================
    pdf.add_page()
    pdf.sec('5. Results & Evaluation')

    # Metrics - 2 column
    pdf.sub('Key Performance Metrics')
    pdf.tbl(
        ['Metric', 'Value', 'Metric', 'Value'],
        [
            ['Total Return', '+422.0%', 'Win Rate', '44.8% (30W / 37L)'],
            ['Ann. Return', '+77.7%', 'Profit Factor', '2.12'],
            ['Volatility', '42.7%', 'Avg Win / Avg Loss', '3.37x'],
            ['Sharpe Ratio', '1.82', 'Avg Trade Return', '+3.14%'],
            ['Max Drawdown', '-24.6%', 'Avg Trade Duration', '2.6 days'],
            ['Return / DD', '3.16x', 'Total Trades', '67'],
            ['Fee Drag', '$137 (23.2%)', 'Gross PnL', '$590.20'],
        ],
        cw=[35, 60, 40, 55]
    )

    # Annual + Regime side by side
    pdf.sub('Annual Breakdown')
    pdf.tbl(
        ['Year', 'Trades', 'Net PnL', 'Win Rate', 'Return'],
        [
            ['2023', '25', '$172.81', '40%', '+172.8%'],
            ['2024', '28', '$226.81', '54%', '+91.7%'],
            ['2025 (YTD Nov)', '14', '$53.49', '36%', '+11.8%'],
        ],
        cw=[35, 22, 40, 30, 63]
    )

    pdf.sub('Trade Return Distribution')
    pdf.tbl(
        ['', 'Min', '5th', '25th', 'Median', '75th', '95th', 'Max'],
        [['Return', '-11.4%', '-5.99%', '-3.03%', '-0.97%', '+5.13%', '+29.0%', '+46.8%']],
        cw=[22, 22, 22, 22, 26, 22, 26, 28]
    )
    pdf.txt(
        'Positive skew: median slightly negative but fat right tail (top 5 trades avg +16.7%) '
        'drives profitability. Characteristic of trend-following: cut losers fast, let winners run.'
    )

    # Equity curve
    pdf.sub('Equity Curve & Drawdown')
    img = os.path.join(IMG_DIR, 'BBEMA_2x_equity_drawdown.png')
    if os.path.exists(img):
        pdf.image(img, x=10, w=190)
    pdf.ln(0.5)

    # Monthly returns
    pdf.sub('Monthly Returns (%)')
    img2 = os.path.join(IMG_DIR, 'BBEMA_2x_monthly_returns.png')
    if os.path.exists(img2):
        pdf.image(img2, x=10, w=190)

    # ==================== PAGE 3 ====================
    pdf.add_page()

    # BTC Buy & Hold comparison
    pdf.sub('Strategy vs BTC Buy & Hold')
    pdf.tbl(
        ['', 'Strategy (2x)', 'BTC Buy & Hold'],
        [
            ['Total Return', '+422.0%', '+470.6%'],
            ['Ann. Return', '+77.7%', '+83.0%'],
            ['Volatility', '42.7%', '46.2%'],
            ['Sharpe Ratio', '1.82', '1.80'],
            ['Max Drawdown', '-24.6%', '-27.7%'],
            ['Time in Market', '16.3%', '100%'],
        ],
        cw=[38, 76, 76]
    )
    pdf.txt(
        'With 2x leverage, the strategy nearly matches B&H absolute return (+422% vs +471%) with a '
        'slightly higher Sharpe (1.82 vs 1.80), lower drawdown (-24.6% vs -27.7%), and only 16% time '
        'exposure. Capital is free 84% of the time for other strategies or yield.'
    )

    # Top trades
    pdf.sub('Top 5 Winning & Losing Trades')
    pdf.tbl(
        ['#', 'Entry Date', 'Entry $', 'Exit $', 'Return', 'Duration'],
        [
            ['W1', '2024-11-05', '$71,134', '$88,019', '+46.8%', '8.6 days'],
            ['W2', '2024-02-26', '$52,851', '$61,825', '+33.3%', '5.0 days'],
            ['W3', '2023-03-13', '$23,628', '$27,608', '+33.0%', '7.7 days'],
            ['W4', '2023-10-19', '$29,010', '$33,767', '+32.1%', '6.6 days'],
            ['W5', '2023-12-01', '$38,820', '$43,174', '+21.8%', '6.3 days'],
            ['L1', '2024-01-11', '$48,695', '$46,068', '-11.4%', '2 hours'],
            ['L2', '2025-08-10', '$121,833', '$118,404', '-6.3%', '1.3 days'],
            ['L3', '2024-04-08', '$71,988', '$69,985', '-6.2%', '1.2 days'],
        ],
        cw=[12, 32, 32, 32, 25, 57], fs=7
    )

    # Exposure & Regime
    pdf.sub('Exposure & Regime Performance')
    pdf.tbl(
        ['Regime', 'Period', 'Return', 'Stat', 'Value'],
        [
            ['Bull', 'Sep 23 - Mar 24', '+136.6%', 'Time in market', '16.3%'],
            ['Bull', 'Sep 24 - Dec 24', '+35.8%', 'Time flat', '83.7%'],
            ['Bull', 'Apr 25 - Oct 25', '+16.9%', 'Max win streak', '4'],
            ['Bear', 'Jun - Aug 2024', '-0.6%', 'Max loss streak', '9'],
            ['Bear', 'Jan - Mar 2025', '0.0%', 'DD recovery', '34 days'],
            ['Sideway', 'Feb - Sep 2023', '+63.9%', 'Expectancy', '$6.77/trade'],
        ],
        cw=[24, 40, 28, 40, 58]
    )

    # Robustness
    pdf.sec('6. Robustness')

    pdf.sub('Parameter Stability (Optuna) & Walk-Forward OOS')
    pdf.txt(
        'Optuna (150 trials, max Sharpe): best params ema=950, bb=75, dev=3.0, nearly identical to '
        'baseline (1000/75/3.0). Top 10 trials all cluster ema=900-1000. Not overfit.'
    )
    pdf.txt(
        'Walk-forward (Jan 2024 - Nov 2025): monthly re-optimization (80 trials), test on next month. '
        'Params stable (ema=900-1450, bb=75-95, dev=2.75-3.75).'
    )
    pdf.tbl(
        ['', 'Full Backtest (IS)', 'Walk-Forward (OOS)'],
        [
            ['Total Return', '+422.0%', '+38.2%'],
            ['Ann. Return', '77.7%', '18.8%'],
            ['Sharpe', '1.82', '0.56'],
            ['Max DD', '-24.6%', '-31.9%'],
        ],
        cw=[40, 75, 75]
    )
    pdf.txt(
        'IS-to-OOS Sharpe decay (1.82 -> 0.56) is typical. Strategy remains profitable OOS (+38.2% '
        'total over 23 months with 2x leverage). OOS params converge to same region as full-sample.'
    )

    # Critique
    pdf.sec('7. Critique & Limitations')

    pdf.sub('When It Works / Struggles')
    pdf.blt('Works: bull markets, post-consolidation breakouts, sideways regimes with intermittent trends.')
    pdf.blt('Struggles: choppy/whipsaw markets (frequent BB touches without follow-through), sharp V-reversals '
            '(exit signals lag), prolonged bear markets (capital sits idle).')
    pdf.blt('2025 degradation (+6.2% YTD) suggests the edge may be weakening in the current regime.')

    pdf.sub('Key Risks')
    pdf.blt('Regime change: if BTC becomes structurally mean-reverting, trend logic generates persistent losses.')
    pdf.blt('Fee + funding sensitivity: fees+funding consume 23.2% of gross PnL. Higher funding during volatile '
            'periods would erode returns further.')
    pdf.blt('Leverage risk: 2x amplifies losses; worst trade = -11.4% equity. At 5x+ leverage the strategy '
            'risks liquidation during sharp moves. 2x chosen as conservative sweet spot.')
    pdf.blt('Small sample: 67 trades over 2.9 years. Profitability driven by ~5 large winners. Removing best '
            '2-3 trades would substantially reduce returns.')
    pdf.blt('Concentration: 200% notional in single BTC position. Black swan events amplified by leverage.')
    pdf.blt('WF OOS Sharpe (0.56) below IS (1.82), suggesting part of the edge may not persist.')

    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 6.5)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 3.5, 'Tools: Python (pandas, ta, optuna, numba).', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 3.5, 'Code & data: https://github.com/Thanh-Van-2001/BuddyTrading_Assessment', new_x="LMARGIN", new_y="NEXT")

    pdf.output(OUT_PATH)
    sz = os.path.getsize(OUT_PATH)
    print(f"PDF saved: {OUT_PATH} ({pdf.page_no()} pages, {sz/1024:.0f} KB)")


if __name__ == '__main__':
    build()

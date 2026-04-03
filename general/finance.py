############################################################################## 
# IMPORT DEPENDECIES
##############################################################################


from .date import BDAY

import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta


__all__ = ['yearly_return','quarterly_return','monthly_return','return_table','rebase','rescale',\
           'Bond','calc_iboraus_notl','calc_fixaus_notl','calc_fixswe_notl',\
           'calcperf','levstats','ret2idx','idxstats',\
           'dd','max_dd','max_dd_length','max_dd_length_dates','max_days_below_highwater',\
           'plegend']
    

##############################################################################
# INPUT FUNCTIONS
##############################################################################

def yearly_return(self,mult=100):
    
    data = self.copy().rebase()
    
    if isinstance(data,pd.Series):
        data = data.to_frame(data.name)
        
    yret = data.asfreq('BA',method='ffill')
    yret.loc[data.index[-1]] = data.iloc[-1]
    yret.index = yret.index.map(lambda x: x.strftime('%Y'))
    syear = str(int(yret.index[0]) - 1)
    yret.loc[syear] = 1.
    yret = yret.sort_index()
    yret = yret.pch().mul(mult).round(1).dropna(how='all')
    
    return yret

def quarterly_return(self,mult=100):
    
    data = self.copy().rebase()
    
    if isinstance(data,pd.Series):
        data = data.to_frame(data.name)
    
    qret = data.asfreq('BQ',method='ffill')
    qret.loc[data.index[-1]] = data.iloc[-1]
    qret.index = pd.PeriodIndex(qret.index,freq='Q')
    squarter = pd.PeriodIndex([(qret.index[0].to_timestamp() - BDAY)],freq='Q')
    qret.loc[squarter[0]] = 1.
    qret = qret.sort_index()
    qret = qret.pch().mul(mult).round(1).dropna(how='all')
    
    qret.index = qret.index.map(lambda x:str(x))
    
    return qret

def monthly_return(self,mult=100):
    
    data = self.copy().rebase()
    
    if isinstance(data,pd.Series):
        data = data.to_frame(data.name)
    
    mret = data.asfreq('BM',method='ffill')
    mret.loc[data.index[-1]] = data.iloc[-1]
    smonth = mret.index[0]-relativedelta(months=1)
    mret.loc[smonth] = 1.
    mret = mret.sort_index()
    mret = mret.pch().mul(mult).round(1).dropna(how='all')
    mret.index = mret.index.map(lambda x: x.strftime('%Y-%b'))
    
    return mret

def return_table(self,mult=100):
    
    data = self.copy().rebase()
    
    if isinstance(data,pd.Series):
        data = data.to_frame(data.name)
    
    ret_table = {}
    
    for k, v in data.items():

        
        mret = v.monthly_return()[k]
        yret = v.yearly_return()[k]
        
        mret.index = pd.MultiIndex.from_arrays([mret.index.map(lambda x: x.split('-')[0]), mret.index.map(lambda x: x.split('-')[1])])
        column_nms = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        r_table     = mret.unstack().reindex(columns=column_nms)
        #r_table    = mret.unstack().reindex(columns=pd.unique(mret.index.get_level_values(1)))
        r_table['Year'] = yret
        
        ret_table[k] = r_table
        
    ret_table_df = pd.concat(ret_table)
        
    return ret_table_df
        
pd.core.generic.PandasObject.yearly_return    = yearly_return
pd.core.generic.PandasObject.quarterly_return = quarterly_return
pd.core.generic.PandasObject.monthly_return   = monthly_return
pd.core.generic.PandasObject.return_table     = return_table

def rebase(self,mode='first'):
    """Rebase the columns of a DataFrame to one using a specified method
    
    Parameters
    ----------
    mode :
      'first' (default)   - Rebase each column to one at first non-NaN observation
      'last'              - Rebase each column to one at last non-NaN observation
      'common'            - Rebase all columns to one at first observation with all columns
      '1987-07-27'        - Rebase all columns to one on specified date
    """
    data = self.copy()
    
    def _rebase(df,mode='first'):
        
        if mode == 'first':
            # Rebase each column to one at first non-NaN observation
            #idx  = (~np.isnan(self)).astype(int).cumsum(0)==1
            mask = df[df.firstobs()].ffill()
        
        elif mode == 'last':
            # Rebase each column to one at last non-NaN observation
            #idx  = ((~np.isnan(self)).astype(int).ix[::-1,:].cumsum(0)==1).ix[::-1,:]
            mask = df[df.lastobs()].bfill()
            
        elif mode == 'common':
            # Rebase all columns to one at first observation with all columns
            idx  = (~np.isnan(df)).astype(int).sum(1).argmax()
            mask = df.ix[idx]
        
        else:
            # Rebase all columns to one on specified date
            mask = df.ix[mode];
    
        return mask
    
    mask = _rebase(data,mode)
        
    return data/mask.values
    
pd.core.generic.PandasObject.rebase = rebase

def rescale(self, target_vol = 0.05):
    """Rescale time series DataFrame to have same volatility
    """
    data = self.copy()
    
    def _rescale(df,target_vol):
        scaler = target_vol * 100 / df.idxstats()['vol']
        return scaler
    
    if isinstance(data,pd.DataFrame):
        scaler = _rescale(data,target_vol)
        return (data.pch() * scaler).ret2idx()
    
    elif isinstance(data,pd.Series):
        scaler = _rescale(data,target_vol).iloc[0]
        return (data.pch() * scaler).ret2idx()

pd.core.generic.PandasObject.rescale = rescale


##############################################################################
# FIXED INCOME UTILITIES
##############################################################################
        
class Bond():
    
    def __init__(self):
        pass

    def __repr__(self):
        return "<Bond>"    
        
    @classmethod
    def from_yield(self,yld,coupon,mat,period=2):
        """ Bond constructor from Yield-to-Maturity
        
        PARAMETERS
        yld     Yield-to-maturity (per year, in percent)
        coupon  Bond coupon (per year, in percent)
        mat     Maturity (in years)
        period  Coupons per year
        """
        
        N      = mat*period
        y2k    = yld/period
        c2k    = coupon/period
        
        self.price  = c2k * ((1-(1 + y2k)**-N)/y2k) + (1+y2k)**-N
        
        self.macdur = ((1+y2k)/y2k - ((1+y2k) + N*(c2k - y2k))/(c2k*((1+y2k)**N - 1) + y2k))/period
        self.moddur = self.macdur/(1+y2k)
        self.dv01   = self.moddur * self.price
                
        return self

def calc_iboraus_notl(pi):
    """ Calculate notional of australian 90 day bank bills
    
    https://www.asx.com.au/documents/products/ird-pricing-guide.pdf
    """
    
    P = 1000000.0 * 365.0 / (365.0 + (100-pi) * 0.9)
    
    return P

def calc_fixaus_notl(pi,yrs=10.,cp=6.):
    """ Calculate notional of australian bond future
    
    http://www.asx.com.au/documents/products/asx-24-interest-rate-price-and-valuation-guide.pdf
    """
    Y = (100 - pi)
    I = Y/200 # Yield in % divided by 100
    V = 1/(1+I)
    C = cp/2 # Coupon Rate/2
    T = yrs*2 # Years * 2
    
    P = 1000*(C*(1-V.pow(T,1))/I + 100*(V.pow(T,1)))

    return P
    
def calc_fixswe_notl(yld,yrs=10.,cp=6.,d2c=360.,nam=1000000.):
    """ Calculate notional of swedish bond future
    
    yld in percentage points (e.g. 5.3)
    nam is notional amount (1M SEK)

    http://www.nasdaqomx.com/digitalAssets/100/100566_government-bond-futures.pdf 
    """
    y = yld/100    
    
    P = (((cp/y)*((1+y).pow(yrs,1)-1)+100.)/((1+y).pow((yrs-1)+(d2c/360.),1)))*(nam/100.)

    return P
    
                   
##############################################################################
# FINANCE FUNCTIONS
##############################################################################

class calcperf:
    """ Portfolio Performance Object
    Represents strategies' weights, returns, and portfolio statisitcs
    
    ATTRIBUTES:
    wt       = Stategy weights (DataFrame or [DataFrames])
    ri       = Asset returns (DataFrame)
    tc       = Transaction cost assumption (Scalar, Series, or DataFrame)
    h        = Strategy headerers (String or [Strings])
    t_rb     = Rebalance dates (Str, DateTimeIndex, [Str], [DateTimeIndex])
                'all': dates when all assets have weights
                'any': dates when any asset has a weight
                DatetimeIndex: custom date vectors
    numrolls = Number of rolls per year (Scalar, Series, DataFrame)
    
    OUTPUTS:
    portri   = Strategy portfolio-level returns (DataFrame)
    attrib   = Strategy position-level attribution (DataFrame or Panel)
    
    stats    = Portfolio summary statistics (DataFrame)
    
    porttc   = Portfolio-level transaction costs (DataFrame)
    attribtc = Position-level transaction costs (Panel)
    
    porttc_roll, attribtc_roll, porttc_trd, attribtc_trd =
        Transaction costs broken down by roll and trading costs
    """
        
    def __init__(self,wts,ri,tc=0,h=None,t_rb='any',freq=None,log=False,arithmetic=False,numrolls=0,tc_roll=0):

        # CONVERT TO LIST OF WEIGHTS AND HEADERS
        if isinstance(wts,(pd.Series,pd.DataFrame)):
            wts = [wts]
        if isinstance(h,str):
            h   = [h]
            
        # GENERATE DEFAULT HEADERS (IF NECESSARY)
        if h is None:
            h = ['strat_' + str(x) for x in range(len(wts))]
    
        # REPLICATE REBALANCE DATES (IF NECESSARY)
        if not isinstance(t_rb,list):
            t_rb = [t_rb] * len(h)
            
        # ITERATE THROUGH STRATEGIES
        port_ri          = {}
        pos_ri           = {}
        port_tc_ri       = {}
        pos_tc_ri        = {}
        port_tc_roll_ri  = {}
        pos_tc_roll_ri   = {}
        port_tc_trd_ri   = {}
        pos_tc_trd_ri    = {}        
        port_to          = {}
        pos_to           = {}
        for ii,wt in enumerate(wts):
            
            # CONVERT SERIES TO DATAFRAME
            if isinstance(wt,pd.Series):
                wt = wt.to_frame()
            
            # GET REBALANCE DATES
            if isinstance(t_rb[ii],str):
                if t_rb[ii] == 'any':
                    t_rb[ii] = wt.dropna(axis=0,how='all').index    # Rebalance on every trade date
                elif t_rb[ii] == 'all':
                    t_rb[ii] = wt.dropna(axis=0,how='any').index    # Rebalance only when all assets have weights

            # CALCULATIONS ON REBALANCE FREQUENCY
            ri_rb       = ri.reindex(index=t_rb[ii],method='ffill')
            wt_rb       = wt.reindex(index=t_rb[ii],method='ffill')
            
            # RETURN (GROSS OF COSTS)
            pos_ret_rb_gross  = (wt_rb.shift(1)*ri_rb.pct_change())
            port_ret_rb_gross = pos_ret_rb_gross.sum(1)
            
            # WTS ADJ BY MKT MOVEMENT AND CHANGE IN PORTFOLIO EQUITY VALUE
            wt_rb_postmkt = (pos_ret_rb_gross + wt_rb.shift(1)).div(1+port_ret_rb_gross,axis=0)
            trade_rb      = wt_rb - wt_rb_postmkt
            to_rb         = trade_rb.abs()
            port_to_rb    = to_rb.sum(1)
            
            # TRANSACTION COSTS (% OF POS)
            pos_tc_trd_ret_rb    = to_rb*-tc
            port_tc_trd_ret_rb   = pos_tc_trd_ret_rb.sum(1)
            pos_tc_trd_ri_rb     = pos_tc_trd_ret_rb.ret2idx()
            port_tc_trd_ri_rb    = port_tc_trd_ret_rb.ret2idx()
            
            pos_tc_roll_ret_rb   = wt_rb.abs()*-tc_roll*2*numrolls/262 # Pay full bid/ask N-times a year. Amortize costs daily.
            port_tc_roll_ret_rb  = pos_tc_roll_ret_rb.sum(1)
            pos_tc_roll_ri_rb    = pos_tc_roll_ret_rb.ret2idx()
            port_tc_roll_ri_rb   = port_tc_roll_ret_rb.ret2idx()
            
            pos_tc_ret_rb        = pos_tc_trd_ret_rb + pos_tc_roll_ret_rb
            port_tc_ret_rb       = pos_tc_ret_rb.sum(1)
            pos_tc_ri_rb         = pos_tc_ret_rb.ret2idx()
            port_tc_ri_rb        = port_tc_ret_rb.ret2idx()
            
            # RETURN (NET OF COSTS)
            pos_ret_rb  = pos_ret_rb_gross  + pos_tc_ret_rb
            port_ret_rb = port_ret_rb_gross + port_tc_ret_rb
            
            pos_ri_rb   = pos_ret_rb.ret2idx()
            port_ri_rb  = port_ret_rb.ret2idx()
            
            # INTRA-REBALANCE CALCULATIONS
            wt_irb      = wt_rb.reindex(index=ri.index,method='ffill')
            
            # INTRA-REBALANCE CHANGE IN UNDERLIER SINCE LAST TRADE
            ri_rb_lag  = ri.ffill().reindex(index=t_rb[ii])
            ri_irb_cng = (ri/ri_rb_lag.reindex(index=ri.index,method='ffill')-1).replace(np.nan,0)
            
            # MULTIPLY BY LAST TRADE WEIGHT AND SUM CROSS-SECTIONALLY
            pos_adj_irb = wt_irb.shift(1)*ri_irb_cng+1
            port_adj_irb= (pos_adj_irb-1).sum(1)+1

            # COMBINE INTRA AND INTER REBALANCE RETURNS
            port_ri[h[ii]]          = port_ri_rb.reindex(index=ri.index,method='ffill')*port_adj_irb
            pos_ri[h[ii]]           = pos_ri_rb.reindex(index=ri.index,method='ffill')*pos_adj_irb
            
            port_tc_ri[h[ii]]       = port_tc_ri_rb
            pos_tc_ri[h[ii]]        = pos_tc_ri_rb
            port_tc_roll_ri[h[ii]]  = port_tc_roll_ri_rb
            pos_tc_roll_ri[h[ii]]   = pos_tc_roll_ri_rb            
            port_tc_trd_ri[h[ii]]   = port_tc_trd_ri_rb
            pos_tc_trd_ri[h[ii]]    = pos_tc_trd_ri_rb

            port_to[h[ii]]          = port_to_rb
            pos_to[h[ii]]           = to_rb

        # BIND INPUTS
        self.wts        = wts
        self.ri         = ri
        self.tc         = tc
        self.tc_roll    = tc_roll
        self.numrolls   = numrolls
        self.h          = h
        self.t_rb       = t_rb
        
        # CONCATENATE STRATEGY RETURNS AND ATTRIBUTION
        self.portri         = pd.DataFrame(port_ri).squeeze()
        self.attrib         = pd.concat(pos_ri).squeeze()
        self.porttc         = pd.DataFrame(port_tc_ri).squeeze()
        self.attribtc       = pd.concat(pos_tc_ri).squeeze()
        self.porttc_roll    = pd.DataFrame(port_tc_roll_ri).squeeze()
        self.attribtc_roll  = pd.concat(pos_tc_roll_ri).squeeze()
        self.porttc_trd     = pd.DataFrame(port_tc_trd_ri).squeeze()
        self.attribtc_trd   = pd.concat(pos_tc_trd_ri).squeeze()
        
        self.portto         = pd.DataFrame(port_to).squeeze()
        self.posto          = pd.concat(pos_to).squeeze()
        
        # CALCULATE PERFORMANCE STATS
        self.stats      = self.idxstats(freq,log,arithmetic)
        
    def idxstats(self,freq,log,arithmetic):
        stats       = self.portri.idxstats(freq,log,arithmetic)
        avg_lev     = np.round(pd.DataFrame([levstats(wt).mean(0) for wt in self.wts],index=self.h),3)
        annual_to   = pd.DataFrame(np.round(self.portto.sum(0)/((self.portto.index[-1] - self.portto.index[0]).days/365.),2),index=self.h,columns=['to'])
        maxtc       = pd.DataFrame(np.round(100*stats.ret/annual_to.to,1),index=self.h,columns=['maxtc'])
        
        return pd.concat([stats,avg_lev,annual_to,maxtc],axis='columns')

    def __repr__(self):
        return str(self.stats)
        

def levstats(wt):
    """Calculate long, short, gross and net leverage 
    """
    if isinstance(wt,pd.Series):
        wt = wt.to_frame()
    
    longwt  =  wt[wt>0].sum(1).fillna(0)
    shortwt = (-wt[wt<0].sum(1)).fillna(0)
    grosswt = longwt + shortwt
    netwt   = longwt - shortwt
        
    return pd.DataFrame({'long':longwt,'short':shortwt,'gross':grosswt,'net':netwt})

def ret2idx(self):
    """ Convert a timeseries of percentage returns into a return index
    """
    idx    = (~np.isnan(self)).astype(int).cumsum(0).shift(-1)==1
    y      = (self+1).cumprod(0)
    y[idx] = 1
    
    return y

pd.core.generic.PandasObject.ret2idx = ret2idx

def idxstats(self,freq=None,log=False,arithmetic=False,round=True):
    """Display performance statistics for a return index
    
    PARAMETERS
    freq:   String = Frequency on which to compue risk
            Int    = Rolling N-period risk
                
    RETURNS
    out:    Series
                Table of returns, volatility, Sharpe Ratio,
                and drawdown for each asset
    """
        
    # CONVERT SERIES TO DATAFRAME
    if isinstance(self,pd.Series):
        self = self.to_frame()
    
    # RESAMPLE (IF REQUIRED)
    if freq is None:
        ri_rs   = self
        madays  = 1.
    elif isinstance(freq,str):
        ri_rs   = self.asfreq(freq,'ffill')
        madays  = 1.
    elif isinstance(freq,int):
        ri_rs   = self
        madays  = freq
         
    # CALCULATE TOTAL NUMBER OF DAYS BETWEEN SDATE TO EDATE
    def calc_delta(x):
        if x.notnull().sum() == 0:
            return np.nan
        else:
            return (self.index[x.lastobs()][0] - self.index[x.firstobs()][0]).days

    dlt         = self.apply(calc_delta).astype(float) 

    # ANNUALIZE VOLATILITY USING DAYFRAC
    est_dayfrac = (ri_rs.count()/dlt)*365/madays
    
    # CALCULATE RETURNS AND VOLATILITY
    if log:
        if arithmetic:
            ret = ri_rs.log().diff(madays).mean() * est_dayfrac
        else:
            ret = (self[self.lastobs()].sum()/self[self.firstobs()].sum()).log()*(365/dlt)
        vol = ri_rs.log().diff(madays).std_zero() * np.sqrt(est_dayfrac)
        
        rets = ri_rs.log().diff(madays)
        skew = rets.skew()
        kurt = rets.kurt() - 3.0
#        mu   = rets.mean()
#        std  = ri_rs.log().diff(madays).std_zero()
#        skew = ((rets ** 3).mean() - 3.0 * mu * std ** 2 - mu ** 3) / (std ** 3)
#        kurt = ((rets - mu) ** 4).mean() / (((rets - mu) ** 2).mean() ** 2) - 3.0
        
    else:
        if arithmetic:
            ret = ri_rs.pct_change(int(madays)).mean() * est_dayfrac
        else:
            ret = (self[self.lastobs()].sum()/self[self.firstobs()].sum())**(365/dlt)-1
            
        vol = ri_rs.pct_change(int(madays)).std_zero() * np.sqrt(est_dayfrac)
        
        rets = ri_rs.pch(int(madays))
        skew = rets.skew()
        kurt = rets.kurt() - 3.0        
    
    # COMPUTE SHARPE RATIO
    ratio = ret/vol
    
    def top1pct_dn(x):
        num_obs  = x.notnull().sum()
        pct1_obs = int(np.round(num_obs/100,0))
        pct1_obs = max(pct1_obs,5)
        return x.pch().nsmallest(pct1_obs).mean()
    
    dd          = ri_rs.max_dd()
    tail        = ri_rs.apply(lambda x: top1pct_dn(x))
    ret_to_tail = - ret / tail
    ret_to_dd   = - ret / dd
#    avg_dd      = ri_rs.dd().mean()
    
    out         = pd.concat([ret,vol,ratio,dd,tail,skew,kurt,ret_to_tail,ret_to_dd],axis=1)
    out.columns = ['ret','vol','ratio','dd','tail','skew','kurt','ret/tail','ret/dd']
    
    if round:
        out = (out*[100,100,1,10,10,0.1,0.1,0.1,0.1]).applymap(lambda x: np.round(x,3))*[1,1,1,10,10,10,10,10,10]
    
    return out
    
pd.core.generic.PandasObject.idxstats = idxstats


def dd(self):
    """ Calculate rolling drawdown
    """
    return self/self.cummax(axis=0)-1

pd.core.generic.PandasObject.dd = dd

def max_dd(self):
    """ Max drawdown
    """
    return self.dd().min()

pd.core.generic.PandasObject.max_dd = max_dd
    
def max_dd_length(self):
    """ Length (in calendar days) of maximum drawdown
    """
    edate = self.dd().idxmin() # Date of trough
    sdate = (self.dd()[:edate] == 0).cumsum().idxmax() # Date of peak before trough

    return (edate - sdate).days

pd.core.series.Series.max_dd_length = max_dd_length

def max_dd_length_df(self):
    """ Length (in days) of maximum drawdown
    """
    return self.apply(max_dd_length)

pd.core.frame.DataFrame.max_dd_length = max_dd_length_df

def max_dd_length_dates(self):
    """ Length (in calendar days) of maximum drawdown
    """
    edate = self.dd().idxmin() # Date of trough
    sdate = (self.dd()[:edate] == 0).cumsum().idxmax() # Date of peak before trough

    return (edate - sdate).days, sdate, edate

pd.core.series.Series.max_dd_length_dates = max_dd_length_dates

def max_dd_length_dates_df(self):
    """ Length (in calendar days) of maximum drawdown
    """
    return self.apply(max_dd_length_dates)

pd.core.frame.DataFrame.max_dd_length_dates = max_dd_length_dates_df

def max_days_below_highwater(self):
    """ Max number of days spent below high water mark
    """
    def days_sep(x):
        return (x.index.max() - x.index.min()).days
    
    return self.cummax().groupby(self.cummax()).apply(days_sep).max()

pd.core.series.Series.max_days_below_highwater = max_days_below_highwater

def max_days_below_highwater_df(self):
    """ Max number of days spent below high water mark
    """
    return self.apply(max_days_below_highwater)

pd.core.frame.DataFrame.max_days_below_highwater = max_days_below_highwater_df

def plegend(self,freq=None,log=False,arithmetic=False,round=True,basic=False):
    """Format headers of a Pandas object to display IDXSTATS
        
    Returns
    ----------
    out:            Pandas Object
                         Pandas Object with new headers
    
    N.B. Could make a MultiIndex. Not yet implemented for Panel.
    """
    
    stats = np.round(self.idxstats(freq,log,arithmetic,round).values.tolist(),3)
    self2 = self.copy()
    
    if basic:
        if isinstance(self2,pd.Series):
            self2.name = str(self2.name) + ' (r=%s%%|v=%s%%|s=%s|d=%s%%)' %(tuple(stats[0][:4]))
            self2 = self2.to_frame()
            
        elif isinstance(self2,pd.DataFrame):
            new_col = []
            for ii,cc in enumerate(self2.columns):
                new_col.append(str(cc) + ' (r=%s%%|v=%s%%|s=%s|d=%s%%)' %(tuple(stats[ii][:4])))
            self2.columns = pd.Index(new_col)
            
    else:
        if isinstance(self2,pd.Series):
            self2.name = str(self2.name) + ' (%s%%|%s%%|s=%s|%s%%|%s%%|%s|%s|%s|%s)' %(tuple(stats[0]))
            self2 = self2.to_frame()
            
        elif isinstance(self2,pd.DataFrame):
            new_col = []
            for ii,cc in enumerate(self2.columns):
                new_col.append(str(cc) + ' (%s%%|%s%%|s=%s|%s%%|%s%%|%s|%s|%s|%s)' %(tuple(stats[ii])))
            self2.columns = pd.Index(new_col)

    return self2
    
pd.core.generic.PandasObject.plegend = plegend

    
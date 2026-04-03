############################################################################## 
# IMPORT DEPENDECIES
##############################################################################

import numpy as np
import pandas as pd

__all__ = ['tolist','firstobs','lastobs','pch','std_zero']

##############################################################################
# INPUT FUNCTIONS
##############################################################################

def tolist(x):
    """ Ensures that an object is a list. If it is not, it is converted into a list.
    """
    
    if isinstance(x,list):
        return x
    elif isinstance(x,pd.Series):
        return list(x)
    elif isinstance (x,set):
        return list(x)
    elif isinstance(x,np.ndarray):
        return list(x)
    elif isinstance(x,pd.Index):
        return list(x)
    elif isinstance(x,tuple):
        return list(x)
    else:
        return [x]
    
def firstobs(self):
    """ Return booolean mask identifying the first non-Nan observation in each column 
    """
    return pd.notnull(self.ffill()).astype(int).cumsum(0)==1

pd.core.generic.PandasObject.firstobs = firstobs
    
def lastobs(self):
    """ Return booolean mask identifying the last non-Nan observation in each column 
    """
    return (pd.notnull(self.bfill()).astype(int).iloc[::-1].cumsum(0)==1).iloc[::-1]
    
                
pd.core.generic.PandasObject.lastobs = lastobs

def pch(self,*args,**kwargs):
    """ Percent change over given number of periods
    
    Parameters
    ----------
    periods : int, default 1
        Periods to shift for forming percent change
    fill_method : str, default 'pad'
        How to handle NAs before computing percent changes
    limit : int, default None
        The number of consecutive NAs to fill before stopping
    freq : DateOffset, timedelta, or offset alias string, optional
        Increment to use from time series API (e.g. 'M' or BDay())
    
    Returns
    -------
    chg : same type as caller

    Alias for pct_change
    """
    return self.pct_change(*args,**kwargs)

pd.core.generic.PandasObject.pch = pch

def std_zero(self,*args,**kwargs):
    """
    Unbiased estimator of population zero-drift standard deviation
    
    Reference:
    1. <Volatility Trading> Euan Sinclair, Chapter 2: Volatility Measurement
    """
    
    n = self.count(*args,**kwargs)
    n = n.astype(float)
        
    k   = (n-1)/2.
    adj = np.sqrt(2/(2*k+1)) * np.sqrt(k) * (1 - 0.125/k + 0.0078125/(k**2))
    
    return np.sqrt(self.pow(2).sum(*args,**kwargs)/(n-1)) / adj

pd.core.generic.PandasObject.std_zero = std_zero



############################################################################## 
# IMPORT DEPENDECIES
##############################################################################

import numpy as np
import pandas as pd

__all__ = ['lplot']

##############################################################################
# INPUT FUNCTIONS
##############################################################################

def lplot(self,**kwds):
    """Plot using a logrithmic Y-Axis
    ax:    matplotlib axis object
    """
    
    self = self.astype(float)
    
    #rng = np.max(np.max(self))-np.min(np.min(self))
    #mn  = np.min(np.min(self))*(1-np.clip(rng,0,1)*.02)
    #mx  = np.max(np.max(self))*(1+np.clip(rng,0,1)*.02)
    
    rng = self.max().max()-self.min().min()
    mn  = self.min().min()*(1-np.clip(rng,0,1)*.02)
    mx  = self.max().max()*(1+np.clip(rng,0,1)*.02)
    
    try:
        tx = np.arange(mn,mx,(mx-mn)/10)   
    except:
        tx = np.arange(mn-0.05,mx+0.05,0.01) 
        
    ax = self.plot(**kwds)
        
    sf_adj = np.floor(np.log10((mn+mx)/2)).astype(int)
    
    if sf_adj > 2:
        ytls = np.round(tx,4-sf_adj).astype(int)
    else:
        ytls = np.round(tx,4-sf_adj)
    
    if mn > 0:   
        ax.set_yscale('log')
    
    ax.set_ylim((mn,mx))
    ax.set_yticks(tx)
    ax.set_yticklabels(ytls)

    return ax

pd.core.generic.PandasObject.lplot = lplot
    

    
import shelve
import numpy as np
from numpy import exp, log
import scipy as sp
from utilities.base import setattrs
from src.compute.mdd.model import Model
from scipy.stats import multivariate_normal as normal

class DVModel(Model):
    def initializer( self ):  
        self.totals = 0
        self.lmtemp = 0
        self.covtemp = 0
        self['logMean'] = [ np.random.uniform(0,0.3), np.random.uniform(-0.5,0.5) ]
        self['logCov'] = np.array([[100,0],[0,100]])

    def eval_(d, v, *, logMean, logCov, **_ ):
        x = np.array([d,v]).transpose( tuple(range(1,d.ndim+1)) + (0,) )
        #print( x.shape )
        ans = normal.logpdf( np.log10(x) , logMean , logCov ) 
        #print( ans.shape )
        return ans

    def grad_(d, v, *, logMean, logCov, **_ ):
        pass

    def prep(self, d, v, weights, factor):
        logDV = np.log10([d,v])
        try:
            lmtemp = factor*np.average( logDV, weights = weights, axis = 1 )
            covtemp = factor*np.cov( logDV, aweights = weights )
        except ZeroDivisionError:
            raise(Exception("Bad wolf"))
            return
        
        if( np.isfinite(covtemp).sum() + np.isfinite(lmtemp).sum()  == 6 ):
            self.totals += factor
            self.lmtemp += lmtemp
            self.covtemp += covtemp

    def step( self ):
        if self.totals > 0:
            self['logMean'] = self.lmtemp / self.totals
            self['logCov'] = self.covtemp / self.totals
            lm = self['logMean']
            lc = self['logCov']
            self.slope = lc[0,1]/lc[0,0]
            self.iterc = lm[1] - self.slope*lm[0]
            self.line = lambda d: (10**self.iterc)*(d**self.slope)
            
        self.totals = 0
        self.lmtemp = 0
        self.covtemp = 0

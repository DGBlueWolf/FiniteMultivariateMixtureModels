import shelve
import numpy as np
from numpy import exp, log
import scipy as sp
from utilities.base import setattrs
from src.compute.mdd.model import Model
from scipy.stats import gamma
from scipy.stats import norm
from scipy.special import digamma

class DModel(Model):
    def initializer( self ):
        self['alpha'] = np.random.uniform(3,8)
        self['beta'] = np.random.uniform(1,4)
        self.deltas = { 'alpha': 0, 'beta': 0 }
        self.totals = 0

    def eval_(d, *, alpha, beta):
        return gamma.logpdf(d, alpha, 0, 1/beta )

    def grad_(d, *, alpha, beta):
        return {
            'alpha': log( beta*d ) - digamma( alpha ),
            'beta': alpha/beta - d,
        }

    def prep(self, grad, weights, factor):
        self.totals += factor * np.sum( weights )
        for k in self:
            self.deltas[k] += factor * np.sum( weights * grad[k] )

    def step(self, lr):
        for k in self:
            self.deltas[k] *= lr/self.totals
            if  np.isnan( self.deltas[k] ):
                pass
            elif self.deltas[k] + self[k] < 0:
                self[k] /= 2
            else:
                self[k] += self.deltas[k]
            self.deltas[k] = 0
        self.totals = 0

class RVModel(Model):
    def initializer( self ):
        self['r0'] = np.random.uniform(0,1)
        self['r1'] = np.random.uniform(3.5,4.5)
        self['r2'] = np.random.uniform(0.5,1.5)
        self.deltas = { 'r0': 0, 'r1': 0, 'r2': 0 }
        self.totals = 0

    def eval_(d, *, r0, r1, r2):
        return r0 + r1*(1-exp(-r2*d))

    def grad_(d, factor = 1, *, r0, r1, r2):
        return {
            'r0': factor*(1),
            'r1': factor*(1 - exp(-r2*d)),
            'r2': factor*(d*r1*exp(-r2*d)),
        }

    def prep(self, grad, weights, factor):
        self.totals += factor * np.sum( weights )
        for k in self:
            self.deltas[k] += factor * np.sum( weights * grad[k] )

    def step(self, lr):
        for k in self:
            self.deltas[k] *= lr/self.totals
            if  np.isnan( self.deltas[k] ):
                pass
            elif self.deltas[k] + self[k] < 0:
                self[k] /= 2
            else:
                self[k] += self.deltas[k]
            self.deltas[k] = 0
        self.totals = 0

class BVModel(Model):
    def initializer( self ):
        self['b0'] = np.random.uniform(2,3.5)
        self['b1'] = np.random.uniform(1,3)
        self['b2'] = np.random.uniform(0.5,1.5)
        self.deltas = { 'b0': 0, 'b1': 0, 'b2': 0 }
        self.totals = 0

    def eval_(d, *, b0, b1, b2):
        return b0 + b1*(1-exp(-b2*d))

    def grad_(d, factor = 1, *, b0, b1, b2):
        return {
            'b0': factor*(1),
            'b1': factor*(1 - exp(-b2*d)),
            'b2': factor*(d*b1*exp(-b2*d)),
        }

    def prep(self, grad, weights, factor):
        self.totals += factor * np.sum( weights )
        for k in self:
            self.deltas[k] += factor * np.sum( weights * grad[k] )

    def step(self, lr):
        for k in self:
            self.deltas[k] *= lr/self.totals
            if  np.isnan( self.deltas[k] ):
                pass
            elif self.deltas[k] + self[k] < 0:
                self[k] /= 2
            else:
                self[k] += self.deltas[k]
            self.deltas[k] = 0
        self.totals = 0

class VModel(Model):
    def initializer( self ):
        self['alpha'] = AVModel()
        self['beta'] = BVModel()

    def eval_( d, v, *, alpha, beta ):
        return gamma( alpha(d), 0, 1/beta(d) ).logpdf( v )

    def grad_( d, v, *, alpha, beta ):
        return {
            'alpha': alpha.grad( d, log( beta(d)*v ) - digamma(alpha(d)) ),
            'beta':  beta.grad( d, alpha(d) / beta(d) - v ),
        }

    def prep(self, grad, weights, factor):
        for k in self:
            self[k].prep( grad[k] , weights , factor )

    def step(self, lr):
        for k in self:
            self[k].step( lr )

class DVModel(Model):
    def initializer( self ):
        self['D'] = DModel()
        self['V'] = VModel()

    def eval_(d, v, *, D, V ):
        return D(d) + V(d, v)

    def grad_(d, v, *, D, V ):
        return {
            'D': D.grad(d),
            'V': V.grad(d, v),
        }

    def prep(self, grad, weights, factor):
        for k in self:
            self[k].prep( grad[k], weights, factor )

    def step( self, lr ):
        for k in self:
            self[k].step( lr )

'''
### INITIAlIZERS ###
# Simple Random initializer for alpha_v parameters
def init_v_alpha():
    return {
        'slope': np.random.uniform(3.5,4.5),
        'scale': np.random.uniform(0.5,1.5),
        'intercept': np.random.uniform(1,2.5),
    }

# Simple Random initializer for beta_v parameters
def init_v_beta():
    return {
        'slope': np.random.uniform(1,3),
        'scale': np.random.uniform(0.5,1.5),
        'intercept': np.random.uniform(2,3.5),
    }

### FUNCTIONS and DERIVATIVES ###
# definition of alpha_v in old model
def alpha_v( d, slope, scale, intercept):
    return  intercept + slope*(1 - np.exp(-scale*d))

# definition of the derivative of alpha_v with respect to each parameter
def der_alpha_v(d, slope, scale, intercept):
    return  {
        'slope':     1 - np.exp(-scale*d),
        'scale':     d*slope*np.exp(-scale*d),
        'intercept': 1,
    }

# definition of beta_v old model
def beta_v( d, slope, scale, intercept):
    return intercept + slope*(1 - np.exp(-scale*d))

# definition of the derivative of beta_v with respect to each parameter
def der_beta_v(d, slope, scale, intercept):
    return  {
        'slope':     1 - np.exp(-scale*d),
        'scale':     d*slope*np.exp(-scale*d),
        'intercept': 1,
    }

# derivative of the logGamma function in the original sense
def der_loggamma_alpha( d, alpha, beta ):
    tmp = beta*d
    tmp[ tmp < minlog ] = minlog
    return np.log( tmp )-digamma(alpha)

# derivative of this function with respect to beta
def der_loggamma_beta( d, alpha, beta ):
    return alpha/beta - d

# full derivative for each alpha_v parameter
def der_V_alpha( d, v, alpha, beta):
    dave = der_alpha_v(d , **alpha)
    dlog = der_loggamma_alpha(v, alpha_v(d,**alpha), beta_v(d,**beta))
    for k in dave:
        dave[k] *= dlog
    return dave

# full derivative of each beta_v parameter
def der_V_beta( d, v, alpha, beta):
    dave = der_beta_v(d , **alpha)
    dlog = der_loggamma_beta(v, alpha_v(d,**alpha), beta_v(d,**beta))
    for k in dave:
        dave[k] *= dlog
    return dave

# definition of the d/v model logarithm
def logmodel( d , v , D , V  ):
    av = alpha_v( d, **V['alpha'] )
    bv = beta_v( d, **V['beta'] )

    tmp1 = gamma.pdf( v, av, 0, 1/bv )
    if len(tmp1.shape) > 0 :
        tmp1[ tmp1 < minlog ] = minlog
    elif tmp1 < minlog:
        tmp1 = minlog

    tmp2 = gamma(D['alpha'],0,1/D['beta']).pdf( d )
    if len(tmp2.shape) > 0 :
        tmp2[ tmp2 < minlog ] = minlog
    elif tmp2 < minlog:
        tmp2 = minlog

    vd = np.log( tmp1 )
    return np.log( tmp2 ) + vd
'''

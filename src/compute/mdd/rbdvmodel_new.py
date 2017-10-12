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
    def initializer( self, *_, **__ ):
        self['alpha'] = np.random.uniform(1,2)
        self['beta'] = np.random.uniform(1,4)
        self.totals = 0
        self.sx = 0
        self.slx = 0
        self.sxlx = 0

    def eval_(d, *, alpha, beta, **_):
        return gamma.logpdf(d, alpha, 0, beta )

    def grad_(d, *, alpha, beta, **_):
        pass

    def prep(self, d, weights, factor):
        self.totals += factor * np.sum( weights )
        self.sx += factor * np.sum( weights * d )
        self.slx += factor * np.sum( weights * log(d) )
        self.sxlx += factor * np.sum( weights * d * log(d) )

    def step(self):
        self.sx /= self.totals
        self.slx /= self.totals
        self.sxlx /= self.totals
        self['alpha'] = ( self.sx / (self.sxlx - self.slx * self.sx) )
        self['beta'] = ( self.sxlx - self.slx * self.sx )
        self.totals = 0
        self.sx = 0
        self.slx = 0
        self.sxlx = 0

class RVModel(Model):
    def initializer( self, lr ):
        self.lr = lr
        self['r0'] = np.random.uniform(0,1)
        self['r1'] = np.random.uniform(1,3)
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

    def step(self):
        lr = self.lr

        #Change me
        for k in self:
            self.deltas[k] *= lr/self.totals

        self.deltas['r0'] -= self.deltas['r1'] / 2
        self.deltas['r1'] /= 2

        for k in self:
            #self.deltas[k] *= lr/self.totals
            if  np.isnan( self.deltas[k] ):
                pass
            elif self.deltas[k] + self[k] < 0:
                self[k] /= 2
            else:
                self[k] += self.deltas[k]
            self.deltas[k] = 0

        #change me
        self['r0'] = 0.0001

        self.totals = 0

class BVModel(Model):
    def initializer( self, lr ):
        self.lr = lr
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

    def step(self):
        lr = self.lr

        #Change me
        for k in self:
            self.deltas[k] *= lr/self.totals

        self.deltas['b0'] -= self.deltas['b1'] / 2
        self.deltas['b1'] /= 2

        for k in self:
            if  np.isnan( self.deltas[k] ):
                pass
            elif self.deltas[k] + self[k] < 0:
                self[k] /= 2
            else:
                self[k] += self.deltas[k]
            self.deltas[k] = 0
        self.totals = 0

class VModel(Model):
    def initializer( self, lr ):
        self['ratio'] = RVModel( lr = lr )
        self['beta'] = BVModel( lr = lr )

    def eval_( d, v, *, ratio, beta ):
        return gamma( ratio(d)*beta(d), 0, 1/beta(d) ).logpdf( v )

    def grad_( d, v, *, ratio, beta ):
        rb = ratio(d)*beta(d)
        factor = log( beta(d) ) + log(v) - digamma(rb)
        return {
            'ratio': ratio.grad( d, beta(d)*factor ),
            'beta':  beta.grad( d, ratio(d)*(1 + factor) - v ),
        }

    def prep(self, grad, weights, factor):
        for k in self:
            self[k].prep( grad[k] , weights , factor )

    def step(self):
        for k in self:
            self[k].step()

class DVModel(Model):
    def initializer( self, **kwargs ):
        self['D'] = DModel( **kwargs )
        self['V'] = VModel( **kwargs )

    def eval_(d, v, *, D, V ):
        return D(d) + V(d, v)

    def grad_(d, v, *, D, V ):
        return {
            'D': D.grad(d),
            'V': V.grad(d, v),
        }

    def prep(self, d, v, weights, factor):
        grad = self.grad(d,v)
        self['D'].prep( d , weights , factor )
        self['V'].prep( grad['V'], weights, factor )

    def step( self ):
        self['D'].step()
        self['V'].step()
        '''for k in self:
            self[k].step()
        self.a = self['V']['ratio']['r0']
        self.b = self['V']['ratio']['r1']
        self.c = self['V']['ratio']['r2']
        self.line = lambda d: self.a + self.b*( 1 - exp( -self.c * d ) )'''

import shelve
import numpy as np
from numpy import exp, log
import scipy as sp
from utilities.base import setattrs
from src.compute.mdd.model import Model
from scipy.stats import gamma
from scipy.stats import norm
from scipy.special import digamma, polygamma

class RVModel(Model):
    def initializer( self ):
        self['r0'] = np.random.uniform(0,1)
        self['r1'] = np.random.uniform(3.5,4.5)
        self['r2'] = np.random.uniform(0.5,1.5)
        self.deltas = { 'r0': 0, 'r1': 0, 'r2': 0 }
        self.totals = 0

    def eval_(d, *, r0, r1, r2):
        return r0 + r1*(1-exp(-r2*d))

    def grad_(d, dr1 = 1, dr2 = 1, *, r0, r1, r2):
        first = {
            'r0': dr1,
            'r1': dr1*(1 - exp(-r2*d)),
            'r2': dr1*(d*r1*exp(-r2*d)),
        }
        second = {
            'r0': dr2,
            'r1': dr2*first['r1']**2
            'r2': dr2*first['r2']**2 - dr1*d*first['r2']
        }
        return first, second

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
        self.firsts = { 'b0': 0, 'b1': 0, 'b2': 0 }
        self.seconds = dict(self.firsts)
        self.delta = {}
        self.Zero = {}
        self.pZero = None
        self.totals = 0

    def eval_(d, *, b0, b1, b2):
        return b0 + b1*(1-exp(-d/b2))

    def grad_(d, db1 = 1, db2 = 1, *, b0, b1, b2):
        first = {
            'b0': db1,
            'b1': db1*(1 - exp(-d/b2)),
            'b2': db1*(d*b1*exp(-d/b2)),
        }
        second = {
            'b0': db2,
            'b1': db2*( first['b1']/db1 )**2
            'b2': (db2*(first['b2']/db1) - db1*d)*(first['b2']/db1)
        }
        return first, second

    def prep(self, grad, weights, factor):
        self.totals += factor * np.sum( weights )
        for k in self:
            self.firsts[k] += factor * np.sum( weights * grad[0][k] )
            self.seconds[k] += factor * np.sum( weights * grad[1][k] )

    def step(self, lr):
        self.firsts['b0'] -= 0.5*self.firsts['b1']
        for k in self:
            self.firsts[k] /= self.totals
            self.seconds[k] /= self.totals
            self.delta[k] = self.firsts[k]*(1 - self.rate[k])/self.seconds[k]
            self.Zero[k] = self[k] - self.firsts[k]/self.seconds[k]

            if  np.isnan( self.deltas[k] ):
                pass
            elif self.deltas[k] + self[k] < 0:
                self[k] /= 2
            else:
                self[k] += self.rate[k]*self.delta[k]

            self.rate[k] *= 1 if self.pZero is None else \
                            exp( (self.gamma/self.delta[k]) * (self.Zero[k] - self.pZero[k]) )
            if self.rate[k] > (1 - lr):
                self.rate[k] = 1

        self.pZero = self.Zero
        self.Zero = {}
        self.totals = 0

class VModel(Model):
    def initializer( self ):
        self['ratio'] = RVModel()
        self['beta'] = BVModel()

    def eval_( d, v, *, ratio, beta ):
        return gamma( ratio(d)*beta(d) , 0, 1/beta(d) ).logpdf( v )

    def grad_( d, v, *, ratio, beta ):
        rb = ratio(d)*beta(d)
        factor = log( beta(d) ) + log(v) - digamma(rb)
        dr1 = beta(d)*factor
        db1 = ratio(d)*(1 + factor) - v
        dr2 = -polygamma(1,rb)*beta(d)**2
        db2 = ratio(d)/beta(d) * ( 1 - rb*polygamma(1,rb) )
        return {
            'ratio': ratio.grad(d, dr1, dr2),
            'beta': beta.grad(d, db1, db2),
        }

    def prep(self, grad, weights, factor):
        for k in self:
            self[k].prep( grad[k] , weights , factor )

    def step(self, lr):
        for k in self:
            self[k].step( lr )

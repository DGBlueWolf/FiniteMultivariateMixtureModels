import numpy as np
from scipy.optimize import fmin_slsqp
from src.compute.mdd.rbdvmodel_new import DModel, VModel, RVModel, BVModel, DVModel


class DModelCNT(DModel):
    def optimize(self, data):
        d, v, c, wt = data
        totals = np.sum(wt * c)
        if totals > 0:
            sx = np.sum(d * c * wt) / totals
            slx = np.sum(np.log(d) * c * wt) / totals
            sxlx = np.sum(d * np.log(d) * c * wt) / totals
            if sxlx - slx * sx > 0:
                self['beta'] = sxlx - slx * sx
                self['alpha'] = sx / self['beta']

class SnowRV(RVModel):
    def initializer(self):
        self.params = ['r0', 'r1', 'r2']
        self.bounds = {
            'r0': (0.001, 0.002),
            'r1': (0.5, 3),
            'r2': (0.0001, 10)
        }
        for k in self.params:
            self[k] = np.random.uniform(*self.bounds[k])


class RainRV(RVModel):
    def initializer(self):
        self.params = ['r0', 'r1', 'r2']
        self.bounds = {
            'r0': (0.001, 0.002),
            'r1': (8, 12),
            'r2': (0.3, 0.85)
        }
        for k in self.params:
            self[k] = np.random.uniform(*self.bounds[k])


class BV(BVModel):
    def initializer(self):
        self.params = ['b0', 'b1', 'b2']
        self.bounds = {
            'b0': (0.0001, 100),
            'b1': (0.0001, 100),
            'b2': (0.0001, 100)
        }
        self['b0'] = np.random.uniform(2, 3)
        self['b1'] = np.random.uniform(2, 3)
        self['b2'] = np.random.uniform(0.5, 1.5)


class RainV(VModel):
    def initializer(self):
        self.params = ['ratio', 'beta']
        self['ratio'] = RainRV()
        self['beta'] = BV()


class SnowV(VModel):
    def initializer(self):
        self.params = ['ratio', 'beta']
        self['ratio'] = SnowRV()
        self['beta'] = BV()

class DVModelCNT(DVModel):
    def optimize(self, data):
        cpy = type(self)(self)
        self['D'].optimize(data)

        def f(x, data):
            cpy.construct(x)
            d, v, c, wt = data
            res = -sum(wt * c * cpy(d, v))
            return res

        def fprime(x, data):
            cpy.construct(x)
            ans = []
            d, v, c, wt = data
            g = cpy.grad(d, v)
            for i in range(len(x)):
                ans.append(np.sum(wt * c * g[i]))
            return ans

        bounds = self.get_bounds()
        x0 = cpy.flatten()
        print(f(x0, data))

        result = fmin_slsqp(func=f, x0=x0, bounds=bounds, fprime=fprime, args=(data,), iter=10, full_output=True,
                            iprint=0)
        xs, fun, its, status, message = result
        self.construct(xs)
        return fun


class RainDV(DVModelCNT):
    def initializer(self, **kwargs):
        self.params = ['D', 'V']
        self['D'] = DModelCNT(**kwargs)
        self['V'] = RainV(**kwargs)


class SnowDV(DVModelCNT):
    def initializer(self, **kwargs):
        self.params = ['D', 'V']
        self['D'] = DModelCNT(**kwargs)
        self['V'] = SnowV(**kwargs)

import numpy as np
from numpy import exp, log
from src.compute.mdd.model import Model
from scipy.stats import gamma
from scipy.special import digamma
from scipy.optimize import fmin_slsqp


class DModel(Model):
    def initializer(self, *_, **__):
        self.params = ['alpha', 'beta']
        self['alpha'] = np.random.uniform(1, 2)
        self['beta'] = np.random.uniform(1, 4)

    def eval_(d, *, alpha, beta, **_):
        return gamma.logpdf(d, alpha, 0, beta)

    def grad_(d, *, alpha, beta, **_):
        return []

    def construct(self, x):
        return x

    def flatten(self):
        return []

    def optimize(self, data):
        totals = sum(wt * np.sum(dvsr['sr']) for dvsr, wt in data)
        if totals > 0:
            sx = sum(wt * np.sum(dvsr['sr'] * dvsr['d']) for dvsr, wt in data) / totals
            slx = sum(wt * np.sum(dvsr['sr'] * log(dvsr['d'])) for dvsr, wt in data) / totals
            sxlx = sum(wt * np.sum(dvsr['sr'] * dvsr['d'] * log(dvsr['d'])) for dvsr, wt in data) / totals
            self['alpha'] = sx / (sxlx - slx * sx)
            self['beta'] = sxlx - slx * sx


class RVModel(Model):
    def initializer(self, *_, **__):
        self.params = ['r0', 'r1', 'r2']
        self.bounds = {
            'r0': (0, 0.5),
            'r1': (0.01, 12),
            'r2': (0.001, 2)
        }
        self['r0'] = np.random.uniform(0, 1)
        self['r1'] = np.random.uniform(1, 3)
        self['r2'] = np.random.uniform(0.5, 1.5)

    def eval_(d, *, r0, r1, r2):
        return r0 + r1 * (1 - exp(-r2 * d))

    def grad_(d, factor=1, *, r0, r1, r2):
        return [factor * (1),
                factor * (1 - exp(-r2 * d)),
                factor * (d * r1 * exp(-r2 * d)),
                ]


class BVModel(Model):
    def initializer(self):
        self.params = ['b0', 'b1', 'b2']
        self.bounds = {
            'b0': (0.05, 10),
            'b1': (0.01, 20),
            'b2': (0.001, 2)
        }
        self['b0'] = np.random.uniform(2, 3.5)
        self['b1'] = np.random.uniform(1, 3)
        self['b2'] = np.random.uniform(0.5, 1.5)

    @staticmethod
    def eval_(d, *, b0, b1, b2):
        return b0 + b1 * (1 - exp(-b2 * d))

    @staticmethod
    def grad_(d, factor=1, *, b0, b1, b2):
        return [
            factor * 1,
            factor * (1 - exp(-b2 * d)),
            factor * (d * b1 * exp(-b2 * d)),
        ]


class VModel(Model):
    def initializer(self):
        self.params = ['ratio', 'beta']
        self['ratio'] = RVModel()
        self['beta'] = BVModel()

    @staticmethod
    def eval_(d, v, *, ratio, beta):
        return gamma(ratio(d) * beta(d), 0, 1 / beta(d)).logpdf(v)

    @staticmethod
    def grad_(d, v, *, ratio, beta):
        rb = ratio(d) * beta(d)
        factor = log(beta(d)) + log(v) - digamma(rb)
        return ratio.grad(d, beta(d) * factor) + \
               beta.grad(d, ratio(d) * (1 + factor) - v)

    def construct(self, x):
        for k in self.params:
            x = self[k].construct(x)
        return x


class DVModel(Model):
    def initializer(self, **kwargs):
        self.params = ['D', 'V']
        self['D'] = DModel(**kwargs)
        self['V'] = VModel(**kwargs)

    @staticmethod
    def eval_(d, v, *, D, V):
        return D(d) + V(d, v)

    @staticmethod
    def grad_(d, v, *, D, V):
        return D.grad(d) + V.grad(d, v)

    def construct(self, x):
        for k in ['D', 'V']:
            x = self[k].construct(x)
        return x

    def optimize(self, data):
        cpy = type(self)(self)
        self['D'].optimize(data)

        def f(x, data):
            cpy.construct(x)
            res = -sum(wt * np.sum(dvsr['sr'] * cpy(dvsr['d'], dvsr['v'])) for dvsr, wt in data)
            return res

        def fprime(x, data):
            cpy.construct(x)
            ans = []
            calc = [list() for _ in range(len(x))]
            for dvsr, wt in data:
                g = cpy.grad(dvsr['d'], dvsr['v'])
                for i in range(len(x)):
                    calc[i].append(wt * np.sum(dvsr['sr'] * g[i]))

            for i in range(len(x)):
                ans.append(-sum(calc[i]))
            return ans

        bounds = self.get_bounds()
        x0 = cpy.flatten()
        print(f(x0, data))

        result = fmin_slsqp(func=f, x0=x0, bounds=bounds, fprime=fprime, args=(data,), iter=10, full_output=True,
                            iprint=0)
        xs, fun, its, status, message = result

        if status not in {0, 9}:
            print(message)
            print("Val: {}, Iterations: {}".format(fun, its))
            input("Ok? ")
        else:
            self.construct(xs)

        return fun

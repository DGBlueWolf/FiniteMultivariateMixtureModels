class Model(dict):
    def initializer(self, **kwargs):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def eval_(*args, **kwargs):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def grad_(*args, **kwargs):
        raise NotImplementedError("Method not implemented for {}".format(type(self)))

    def __init__(self, init_={}, **kwargs):
        self.params = []
        self.bounds = {}
        self.initializer(**kwargs)
        self.update(init_)
        pass

    def __call__(self, *args, **kwargs):
        kwargs.update(self)
        return type(self).eval_(*args, **kwargs)

    def grad(self, *args, **kwargs):
        kwargs.update(self)
        return type(self).grad_(*args, **kwargs)

    def construct(self, x):
        L = len(self.params)
        for k, xi in zip(self.params, x[:L]):
            self[k] = xi
        return x[L:]

    def flatten(self):
        res = []
        for k in self.params:
            if isinstance(self[k], Model):
                res += self[k].flatten()
            else:
                res.append(self[k])
        return res

    def get_bounds(self):
        res = []
        for k in self.params:
            if k in self.bounds:
                res.append(self.bounds[k])
            elif isinstance(self[k], Model):
                res += self[k].get_bounds()
        return res

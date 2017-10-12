class Model(dict):
    def initializer(self):
        raise NotImplementedError("Please Implement this method")

    def eval_():
        raise NotImplementedError("Please Implement this method")

    def grad_():
        raise NotImplementedError("Method not implemented for {}".format(type(self)))

    def __init__(self, init_ = {}, **kwargs ):
        self.initializer( **kwargs )
        self.update( init_ )
        pass

    def __call__(self, *args, **kwargs):
        kwargs.update(self)
        return type(self).eval_(*args, **kwargs)

    def grad(self, *args, **kwargs):
        kwargs.update(self)
        return type(self).grad_(*args, **kwargs)

    def prep(self):
        raise NotImplementedError("Please Implement this method")

    def step(self):
        raise NotImplementedError("Please Implement this method")

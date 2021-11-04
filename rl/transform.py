import torch


class RunningStat:
    '''
    Keeps track of a running estimate of the mean and standard deviation of
    a distribution based on the observations seen so far

    Attributes
    ----------
    _M : torch.float
        estimate of the mean of the observations seen so far

    _S : torch.float
        estimate of the sum of the squared deviations from the mean of the
        observations seen so far

    n : int
        the number of observations seen so far

    Methods
    -------
    update(x)
        update the running estimates of the mean and standard deviation

    mean()
        return the estimated mean

    var()
        return the estimated variance

    std()
        return the estimated standard deviation
    '''

    def __init__(self):
        self._M = None
        self._S = None
        self.n = 0

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self._M = x.clone()
            self._S = torch.zeros_like(x)
        else:
            old_M = self._M.clone()
            self._M = old_M + (x - old_M) / self.n
            self._S = self._S + (x - old_M) * (x - self._M)

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self.n > 1:
            var = self._S / (self.n - 1)
        else:
            var = torch.pow(self.mean, 2)

        return var

    @property
    def std(self):
        return torch.sqrt(self.var)



class Transform:
    '''
    Composes several transformation and applies them sequentially

    Attributes
    ----------
    filters : list
        a list of callables

    Methods
    -------
    __call__(x)
        sequentially apply the callables in filters to the input and return the
        result
    '''

    def __init__(self, state_bound, z_filter):
        '''
        Parameters
        ----------
        filters : variatic argument list
            the sequence of transforms to be applied to the input of
            __call__
        '''

        self.state_bound = state_bound
        self.z_filter = z_filter

    def __call__(self, x, update=True):

        x = self.state_bound(x)
        x = self.z_filter(x, update)

        return x


class ZFilter:
    '''
    A z-scoring filter

    Attributes
    ----------
    running_stat : RunningStat
        an object that keeps track of an estimate of the mean and standard
        deviation of the observations seen so far

    Methods
    -------
    __call__(x)
        Update running_stat with x and return the result of z-scoring x
    '''

    def __init__(self):
        self.running_stat = RunningStat()

    def __call__(self, x, update):
        if update:
            self.running_stat.update(x)
        x = (x - self.running_stat.mean) / (self.running_stat.std + 1e-8)

        return x


class Bound:
    '''
    Implements a bounding function

    Attributes
    ----------
    low : int
        the lower bound

    high : int
        the upper bound

    Methods
    -------
    __call__(x)
        applies the specified bounds to x and returns the result
    '''

    def __init__(self, low, high):
        '''
        Parameters
        ----------
        low : int
            the lower bound

        high : int
            the upper bound
        '''
        
        self.low = low
        self.high = high

    def __call__(self, x):
        x = torch.clamp(x, self.low, self.high)

        return x

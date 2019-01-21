from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
from builtins import str
from builtins import range
from builtins import object

import numpy as np


class CygnusDistribution(object):
    '''
    the distribution looks a little like the constellation Cygnus
    '''
    def __init__(self):
        self.lower_limits = np.array([-5, -5])
        self.upper_limits = -self.lower_limits
        self.param_names = [
            'log10probzero',
            'log10probpos']
        self.nparams = self.lower_limits.shape[0]
        assert self.nparams == self.upper_limits.shape[0]
        assert len(self.param_names) == self.nparams

    def logprior(params):
        if params.shape[0] != self.nparams:
            err = 'wrong number of parameters: expected {}, got {}'.format(
                self.nparams, params.shape[0])
            raise ValueError(err)
        # for now, uniform over all parameters on the scales implied by limits
        if (np.any(params < self.lower_limits) or
                np.any(params > self.upper_limits)):
            return -np.inf
        else:
            return np.sum(np.log(self.upper_limits-self.lower_limits))

    def logprob(s, params):
        '''
        s        selection coefficient
        params   distribution parameters
        '''
        logprobzero = params[0]
        lambda_pos, lambda_neg = params[1:3]
        probpos = params[3]
        if np.abs(s) < self.zero_boundary:
            return logprobzero
        else:
            lp = (np.log(1.0-np.exp(logprobzero))
                        + np.log(np.abs(s-np.sign(s)*self.zero_boundary)))
            lp += np.log(probpos) if s > 0 else np.log(1-probpos)
            return lp

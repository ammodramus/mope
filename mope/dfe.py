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
    The distribution looks a little like the constellation Cygnus
    '''
    def __init__(self):
        self.lower_limits = np.array([-5, -5, -3])
        #self.upper_limits = -self.lower_limits
        self.upper_limits = np.array((0, 0, -self.lower_limits[-1]))
        self.param_names = [
            'log10probzero',
            'log10probpos',
            'log10ratiopostoneg',
        ]
        self.nparams = self.lower_limits.shape[0]
        assert self.nparams == self.upper_limits.shape[0]
        assert len(self.param_names) == self.nparams
        self.zero_limit = 0.2

    def logprior(params):
        if params.shape[0] != self.nparams:
            err = 'wrong number of parameters: expected {}, got {}'.format(
                self.nparams, params.shape[0])
            raise ValueError(err)
        # For now, we will assume that the prior distribution is uniform over
        # all parameters on the scales implied by limits.
        if (np.any(params < self.lower_limits) or
                np.any(params > self.upper_limits)):
            return -np.inf
        else:
            return np.sum(np.log(self.upper_limits-self.lower_limits))

    def get_loglike(self, dfe_params, focal_alpha_neg, alphas):
        '''
        dfe_params         distribution parameters (log10probzero,
                           log10probpos)
        focal_alpha_neg    mean alpha for the distribution of alphas
        alphas             population-scaled selection coefficient(s)
        '''
        log10probzero = dfe_params[0]
        log10probnonzero = np.log(1.0-np.exp(log10probzero))
        log10probpos = dfe_params[1]
        log10ratiopostoneg = dfe_params[2]
        focal_alpha_pos = 10**log10ratiopostoneg * 10**focal_alpha_neg
        logprobs = np.zeros_like(alphas)
        boundary_filt = np.abs(alphas) < self.zero_limit
        logprobs[boundary_filt] = log10probzero
        pos_filt = (~boundary_filt) & (alphas > 0)
        logprobs[pos_filt] = (log10probnonzero + np.log(focal_alpha_pos)
                              - focal_alpha_pos*alphas[pos_filt])
        neg_filt = (~boundary_filt) & (alphas < 0)
        logprobs[neg_filt] = (log10probnonzero + np.log(focal_alpha_neg)
                              - focal_alpha_neg*alphas[neg_filt])
        val = np.sum(logprobs)
        import pdb; pdb.set_trace()
        return val

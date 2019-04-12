from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
from builtins import str
from builtins import range
from builtins import object

import numpy as np
from scipy.special import logit, expit

class CygnusDistribution(object):
    '''
    This distribution looks a little like the constellation Cygnus.

    The three parameters are:
        - logitprobzero:              logit-probability that the selection
                                      is zero
        - logitprobpos                logit-probability that a non-zero selection
                                      coefficient is positive.
        - log10ratiopostoneg          log10 ratio of positive to negative mean
                                      selection coefficients
        - log10focalmeanalphaneg      log10 mean of the negative part of the DFE
                                      for the focal branch (which has relative size 1)

    Additionally, the loglike function must be specified a mean value of the
    scaled selection coefficient alpha for the negative part of the selection
    coefficient distribution.
    '''
    def __init__(self):
        self.lower_limits = np.array([-6, -6, -4, -6])
        self.upper_limits = -self.lower_limits
        self.upper_limits[-1] = 0
        self.param_names = [
            'logitprobzero',
            'logitprobpos',
            'log10ratiopostoneg',
            'log10focalmeanalphaneg',
        ]
        self.nparams = self.lower_limits.shape[0]
        assert self.nparams == self.upper_limits.shape[0]
        assert len(self.param_names) == self.nparams

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

    def get_loglike(self, dfe_params, alphas):
        '''
        dfe_params              distribution parameters
            The three parameters are:
                - logitprobzero:           The logit-probability that the selection
                                           coefficient is zero
                - logitprobpos             The logit-probability that a non-zero
                                           selection coefficient is positive.
                - log10ratiopostoneg       The log10 ratio of positive to negative
                                           mean selection coefficients
                - log10focalmeanalphaneg   log10 mean of the negative part of the DFE
                                           for the focal branch (which has
                                           relative size 1)
        alphas                  population-scaled selection coefficient(s)
        '''
        logitprobzero = dfe_params[0]
        logitprobpos = dfe_params[1]
        log10ratiopostoneg = dfe_params[2]
        log10focalmeanalphaneg = dfe_params[3]

        prob_zero = expit(logitprobzero)
        prob_nonzero = 1-prob_zero
        log_prob_zero = np.log(prob_zero)
        log_prob_nonzero = np.log(prob_nonzero)
        
        focal_mean_alpha_neg = 10**log10focalmeanalphaneg

        log_prob_pos = np.log(expit(logitprobpos))
        log_prob_neg = np.log(1-np.exp(log_prob_pos))

        focal_mean_alpha_pos = 10**log10ratiopostoneg * focal_mean_alpha_neg


        logprobs = np.zeros_like(alphas)
        is_zero = alphas == 0.0
        logprobs[is_zero] = log_prob_zero
        pos_filt = alphas > 0
        # This is where the exponential part of the "Cygnus" distribution comes
        # in. The log(alpha) + mean_alpha*alpha is the log-probability under
        # the exponential distribution.
        logprobs[pos_filt] = (log_prob_nonzero + log_prob_pos
                              + np.log(focal_mean_alpha_pos)
                              - focal_mean_alpha_pos*alphas[pos_filt])
        neg_filt = alphas < 0
        logprobs[neg_filt] = (log_prob_nonzero + log_prob_neg
                              + np.log(focal_mean_alpha_neg)
                              - focal_mean_alpha_neg*alphas[neg_filt])
        val = np.sum(logprobs)
        return val


if __name__ == '__main__':
    # Some simple tests
    import numpy.random as npr

    dfe = CygnusDistribution()
    logit_prob_zero = npr.uniform(-3, 3)
    logit_prob_pos = npr.uniform(-3, 3)
    log10_pos_to_neg = 0.5
    dfe_params = np.array([logit_prob_zero, logit_prob_pos, log10_pos_to_neg])


    num_loci = 10

    nreps = 1000
    for i in range(nreps):

        alphas = np.zeros(num_loci)
        focal_mean_alpha_neg = npr.uniform(0.001, 0.999)
        dfe_val_1 = dfe.get_loglike(dfe_params, focal_mean_alpha_neg, alphas)
        expected_dfe_val_1 = num_loci * np.log(expit(logit_prob_zero))
        assert np.isclose(dfe_val_1, expected_dfe_val_1)


        alpha = npr.uniform(0.1, 1.0)
        alphas = np.ones(num_loci) * alpha
        focal_mean_alpha_neg = npr.uniform(0.001, 0.999)
        focal_mean_alpha_pos = 10**log10_pos_to_neg * focal_mean_alpha_neg
        expected_dfe_val_2 = num_loci * (
            np.log(1-expit(logit_prob_zero)) + np.log(expit(logit_prob_pos))
            + np.log(focal_mean_alpha_pos) - focal_mean_alpha_pos * alpha)
        dfe_val_2 = dfe.get_loglike(dfe_params, focal_mean_alpha_neg, alphas)
        assert np.isclose(dfe_val_2, expected_dfe_val_2)


        alpha = npr.uniform(-1.0, -0.1)
        alphas = np.ones(num_loci) * alpha
        focal_mean_alpha_neg = npr.uniform(0.001, 0.999)
        expected_dfe_val_3 = num_loci * (
            np.log(1-expit(logit_prob_zero)) + np.log(1.0-expit(logit_prob_pos))
            + np.log(focal_mean_alpha_neg) - focal_mean_alpha_neg * alpha)
        dfe_val_3 = dfe.get_loglike(dfe_params, focal_mean_alpha_neg, alphas)
        assert np.isclose(dfe_val_3, expected_dfe_val_3)

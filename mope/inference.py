from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
import argparse
import sys
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np
import scipy.optimize as opt
import pandas as pd
from . import likelihoods as lis
from . import transition_data_mut as tdm
from . import transition_data_bottleneck as tdb
from . import params as par
from . import initialguess as igs
from functools import partial
import multiprocessing as mp
import numpy.random as npr
import emcee
import warnings

from . import newick
from . import util as ut
from . import data as da
from . import _binom
from . import _util
from .pso import pso
from .simulate import get_parameters
from . import ascertainment as asc

inf_data = None

def logl(x):
    global inf_data
    return -1.0*inf_data.inf_bound_like_obj(x)

def logp(x):
    global inf_data
    return inf_data.logprior(x)

def post_clone(x):
    global inf_data
    return inf_data.log_posterior(x)

def target(x):
    global inf_data
    val = -1.0*inf_data.penalty_bound_log_posterior(x)
    return val

def inf_bound_target(x):
    global inf_data
    val = -1.0*inf_data.log_posterior(x)
    return val


def optimize_posterior(inf_data, pool):
    '''
    maximizes posterior

    inf_data    Inference object

    returns varparams in normal space
    '''

    lower_bound = inf_data.lower+1e-10
    upper_bound = inf_data.upper-1e-10

    swarmsize = 5000
    minfunc = 1e-5
    swarm_init_weight = 0.1

    # doesn't actually matter that it's 2, just needs to be > 1
    # to signal that the pool should be used
    num_processes = 1 if pool is None else 2

    x, f = pso(target, lower_bound, upper_bound,
            swarmsize = swarmsize, minfunc = minfunc,
            init_params = inf_data.init_params,
            init_params_weight = swarm_init_weight,
            pool = pool, processes = num_processes)

    if inf_data.print_debug:
        print('! pso:', x, f)

    bounds = [[l,u] for l, u in zip(inf_data.lower, inf_data.upper)]
    epsilon = 1e-10
    x, f, d = opt.lbfgsb.fmin_l_bfgs_b(
            target,
            x,
            approx_grad = True,
            factr = 50,
            bounds = bounds,
            epsilon = epsilon)

    if inf_data.print_debug:
        print('! lbfgsb:', x, f)

    options = {'maxfev': 1000000}
    res = opt.minimize(inf_bound_target,
            x,
            method = "Nelder-Mead",
            options = options)
    x = res.x
    f = res.fun

    if inf_data.print_debug:
        print('! nelder-mead:', x, f)


    return x

def _get_valid_start_from(sf_str):
    valid_sfs = ('initguess', 'true', 'map', 'prior')
    for v in valid_sfs:
        if sf_str == v:
            return v
    warnings.warn('getting starting position from heuristic guess')
    return 'initguess'


class Inference(object):
    def __init__(self,
            data_file,
            transitions_file,
            tree_file,
            true_parameters,
            start_from,
            data_are_freqs,
            genome_size,
            ages_data_fn,
            bottleneck_file = None,
            poisson_like_penalty = 1.0,
            min_freq = 0.001,
            transition_copy = None,
            transition_buf = None,
            transition_shape = None,
            print_debug = False,
            log_unif_drift = False,
            inverse_bot_priors = False,
            post_is_prior = False,
            lower_drift_limit = 1e-3,
            upper_drift_limit = 3):

        self.asc_tree = None
        self.asc_num_loci = None
        self.asc_ages = None
        self.asc_counts = None
        self.coverage = None
        self.init_params = None
        self.true_params = None
        self.true_loglike = None
        self.lower = None
        self.upper = None
        self.transition_data = None
        self.bottleneck_data = None

        self.data_are_freqs = data_are_freqs
        self.genome_size = genome_size
        self.tree_file = tree_file
        self.transitions_file = transitions_file
        self.data_file = data_file
        self.start_from = _get_valid_start_from(start_from)
        self.init_true_params = true_parameters
        self.bottleneck_file = bottleneck_file
        self.poisson_like_penalty = poisson_like_penalty
        self.min_freq = min_freq
        self.transition_copy = transition_copy
        self.transition_buf = transition_buf
        self.transition_shape = transition_shape
        self.ages_data_fn = ages_data_fn
        self.print_debug = print_debug
        self.log_unif_drift = log_unif_drift
        self.inverse_bot_priors = inverse_bot_priors
        self.post_is_prior = post_is_prior
        self.lower_drift_limit = lower_drift_limit,
        self.upper_drift_limit = upper_drift_limit

        ############################################################
        # read in data
        dat = pd.read_csv(data_file, sep = '\t', comment = '#',
                na_values = ['NA'], header = 0)

        ############################################################
        # get a count for each unique row, since each will have its own
        # likelihood. this is useful only for called frequencies and simulated
        # data
        # frequencies and ages should be floats
        for col in list(dat.columns):
            if col != 'family':
                dat[col] = dat[col].astype(np.float64)
        self.original_data = dat
        self.data = dat.copy()

        ############################################################
        # add transition data
        if transition_copy is None:
            self.transition_data = tdm.TransitionData(self.transitions_file,
                    memory = True)
        else:
            self.transition_data = tdm.TransitionDataCopy(transition_copy,
                    transition_buf, transition_shape)
        self.freqs = self.transition_data.get_bin_frequencies()
        self.num_freqs = self.freqs.shape[0]
        self.breaks = self.transition_data.get_breaks()
        self.transition_N = self.transition_data.get_N()

        ############################################################
        # add optional bottleneck data
        if self.bottleneck_file is not None:
            self.bottleneck_data = tdb.TransitionDataBottleneck(
                    self.bottleneck_file, memory = True)
        else:
            self.bottleneck_data = None

        ############################################################
        # read in the tree file
        with open(tree_file) as fin:
            self.tree_str = fin.read().strip()
        self.plain_tree = newick.loads(
                self.tree_str,
                length_parser = ut.length_parser_str,
                look_for_multiplier = True)[0]

        ############################################################
        # a list of the leaves, each corresponding to one of the observed
        # frequency columns
        self.leaf_names = []
        # a list of all of the nodes having a length
        self.branch_names = []
        # a list of the unique multiplier names
        self.multipliernames = []
        # a list of the varnames
        self.varnames = []
        # self.multiplierdict[nodename] gives the name of the multiplier for
        # node named nodename
        self.multiplierdict = {}
        # a dict of varnames for each node
        self.varnamedict = {}
        # a dict to determine whether each varname is a bottleneck
        self.is_bottleneck = {}

        multipliernames_set = set([])
        varnames_set = set([])
        for node in self.plain_tree.walk(mode='postorder'):
            if node == self.plain_tree:
                continue
            if node.is_leaf:
                self.leaf_names.append(node.name)
            self.branch_names.append(node.name)
            if node.multipliername is not None:
                multipliernames_set.add(node.multipliername)
                self.multiplierdict[node.name] = node.multipliername
            else:
                self.multiplierdict[node.name] = None
            # check that each node.name is unique
            varnames_set.add(node.varname)
            if node.name in self.varnamedict:
                raise ValueError(
                'the name of each node in the tree must be unique')
            self.varnamedict[node.name] = node.varname
            if node.is_bottleneck:
                self.is_bottleneck[node.varname] = True
            else:
                self.is_bottleneck[node.varname] = False

        self.multipliernames = list(multipliernames_set)
        self.varnames = sorted(list(varnames_set))

        if self.data_are_freqs:
            self.coverage_names = None
        else:
            self.coverage_names = [el + '_n' for el in self.leaf_names]

        # number of branches having a length
        self.num_branches = len(self.branch_names)
        # number of leaves
        self.num_leaves = len(self.leaf_names)
        # indices of the variables with lengths
        self.branch_indices = {}
        for i, br in enumerate(self.branch_names):
            self.branch_indices[br] = i
        # indices of just the leaves
        self.leaf_indices = {}
        for i, le in enumerate(self.leaf_names):
            self.leaf_indices[le] = i

        self.var_indices = {vname: i for i, vname in enumerate(self.varnames)}
        translate_indices = []
        num_varnames = len(self.varnames)
        for br in self.branch_names:
            br_vname = self.varnamedict[br]
            vname_idx = self.var_indices[br_vname]
            translate_indices.append(vname_idx)
        for br in self.branch_names:
            br_vname = self.varnamedict[br]
            vname_idx = self.var_indices[br_vname]
            translate_indices.append(num_varnames+vname_idx)
        # for ab
        translate_indices.append(2*num_varnames)
        # for p_poly
        translate_indices.append(2*num_varnames+1)
        self.translate_indices = np.array(translate_indices,
                dtype = np.int)


        # for each branch length parameter, need a min and a max multiplier
        # this is 1 if it is not multiplied by any of the multipliers this is
        # min(mult) and max(mult) otherwise, where mult is the relevant
        # multiplier
        self.min_multipliers = np.ones(len(self.branch_names))
        self.max_multipliers = np.ones(len(self.branch_names))
        # avg_multipliers needs to be ndim in length
        self.avg_multipliers = np.ones(2*len(self.branch_names)+2)
        for i, br in enumerate(self.branch_names):
            multname = self.multiplierdict[br]
            if multname is not None:
                self.min_multipliers[i] = self.data[multname].min(skipna = True)
                self.max_multipliers[i] = self.data[multname].max(skipna = True)
                self.avg_multipliers[i] = self.data[multname].mean(skipna = True)

        #####################################################
        # removing extraneous columns
        # verifying all leaves present
        

        self.data.sort_values('family', inplace = True)

        #####################################################
        # rounding down frequencies below min_freq
        #####################################################

        # round down if allele frequencies are taken as true
        if self.data_are_freqs:
            for ln in self.leaf_names:
                needs_rounding0 = (self.data[ln] < self.min_freq)
                needs_rounding1 = (self.data[ln] > 1-self.min_freq)
                self.data.loc[needs_rounding0, ln] = 0.0
                self.data.loc[needs_rounding1, ln] = 1.0

        # don't trust allele frequencies less than 0.005. this is *not* because
        # of the lack of binomial sampling. it's because we're not really
        # modeling error. So what should be done with allele counts when the
        # MLE is less than 0.005? Could round down to zero, essentially
        # discarding those. That is a good first thing to try.

        else:   # self.data_are_freqs == False
            for ln, nn in zip(self.leaf_names, self.coverage_names):
                freqs = self.data[ln].astype(np.float64)/self.data[nn]
                needs_rounding0 = (freqs < self.min_freq)
                needs_rounding1 = (freqs > 1-self.min_freq)
                self.data.loc[needs_rounding0, ln] = 0
                self.data.loc[needs_rounding1, ln] = (
                        self.data.loc[needs_rounding1, nn])

        ##################################################### 
        # remove any fixed loci
        ##################################################### 
        fixed = da.is_fixed(self.data, self.leaf_names,
                not self.data_are_freqs)
        self.data = self.data.loc[~fixed,:]
        self.data = self.data.reset_index(drop = True)

        #####################################################
        # ages data
        #####################################################
        self.asc_ages = pd.read_csv(ages_data_fn, sep = '\t',
                comment = '#')
        counts = []
        for fam in self.asc_ages['family']:
            count = (
                self.data.loc[self.data['family'] == fam,:].shape[0])
            counts.append(count)
        self.asc_ages['count'] = counts
        self.num_asc_combs = self.asc_ages.shape[0]

        #####################################################
        # ascertainment
        #####################################################
        asc_tree = newick.loads(
                self.tree_str,
                num_freqs = self.num_freqs,
                num_loci = 2*self.num_asc_combs,
                length_parser = ut.length_parser_str,
                look_for_multiplier = True)[0]
        self.asc_tree = asc_tree

        self.num_loci = self.data.shape[0]
        self.tree = newick.loads(self.tree_str,
                num_freqs = self.num_freqs,
                num_loci = self.num_loci,
                length_parser = ut.length_parser_str,
                look_for_multiplier = True)[0]

        if 'coverage' in self.data.columns:
            # assumes that all coverages are the same, as in simulations
            self.coverage = int(self.data['coverage'][0])
        else:
            self.coverage = None

        #####################################################
        # leaf allele frequency likelihoods
        #####################################################
        if not self.data_are_freqs:
            n_names = self.coverage_names
            count_dat = self.data.loc[:,self.leaf_names].values.astype(
                    np.float64)
            ns_dat = self.data.loc[:,n_names].values.astype(
                    np.float64)
            leaf_likes = _binom.get_binom_likelihoods_cython(count_dat,
                    ns_dat, self.freqs)
        else:
            freq_dat = self.data.loc[:,self.leaf_names].values.astype(
                    np.float64)
            leaf_likes = _binom.get_nearest_likelihoods_cython(freq_dat,
                    self.freqs)

        leaf_likes_dict = {}
        for i, leaf in enumerate(self.leaf_names):
            leaf_likes_dict[leaf] = leaf_likes[:,i,:].copy('C')

        self.leaf_likes_dict = leaf_likes_dict

        #####################################################
        # likelihood objective
        #####################################################
        # min_mults and max_mults give the minimum and
        # maximum multipliers for each varname
        self.min_mults, self.max_mults = (
                self.get_min_max_mults())


        #####################################################
        # header
        #####################################################
        length_names = [el+'_l' for el in self.varnames]
        mut_names = [el+'_m' for el in self.varnames]
        header_list = (['ll'] + length_names + mut_names +
                ['log10ab', 'log10polyprob'])
        self.header_list = header_list
        self.header = '\t'.join(header_list)


        #####################################################
        # making bounds for likelihood functions
        #####################################################
        num_varnames = len(self.varnames)
        self.num_varnames = num_varnames
        ndim = 2*num_varnames + 2

        # (hard-coded bounds)
        min_allowed_len = max(self.transition_data.get_min_coal_time(),
                lower_drift_limit)
        max_allowed_len = min(self.transition_data.get_max_coal_time(),
                upper_drift_limit)
        min_allowed_bottleneck = 2
        max_allowed_bottleneck = 500
        min_mut = max(-8,
                np.log10(self.transition_data.get_min_mutation_rate()))
        max_mut = min(-1,
                np.log10(self.transition_data.get_max_mutation_rate()))
        min_ab = -9
        max_ab = 0
        min_polyprob = -9
        max_polyprob = 0

        lower_len = min_allowed_len / self.min_mults
        upper_len = max_allowed_len / self.max_mults

        is_bottleneck_arr = np.array(
                [self.is_bottleneck[vn] for vn in self.varnames], dtype = bool)
        self.is_bottleneck_arr = is_bottleneck_arr
        lower_len[is_bottleneck_arr] = min_allowed_bottleneck
        upper_len[is_bottleneck_arr] = max_allowed_bottleneck

        # convert len limits to log10... above they should be natural scale
        lower_len = np.log10(lower_len)
        upper_len = np.log10(upper_len)

        lower_mut = np.repeat(min_mut, num_varnames)
        upper_mut = np.repeat(max_mut, num_varnames)
        lower = np.concatenate((lower_len, lower_mut, (min_ab,min_polyprob)))
        upper = np.concatenate((upper_len, upper_mut, (max_ab,max_polyprob)))

        # self.lower and .upper give the bounds for prior distributions
        self.lower = lower
        self.upper = upper

        if self.print_debug:
            print(self.min_mults)
            print(self.max_mults)
            print('! self.lower:')
            for el in self.lower:
                print('! ', el)
            print('! self.upper:')
            for el in self.upper:
                print('! ', el)

        #####################################################
        # true parameters, if specified (for sim. data)
        #####################################################
        if true_parameters:
            true_bls, true_mrs, true_ab, true_ppoly = (
                    get_parameters(true_parameters, self.plain_tree))
            log10_true_ab = np.log10(true_ab)
            log10_true_ppoly = np.log10(true_ppoly)
            true_params = []
            for vn in self.varnames:
                true_params.append(np.log10(true_bls[vn]))
            for vn in self.varnames:
                true_params.append(np.log10(true_mrs[vn]))
            true_params.append(log10_true_ab)

            true_params.append(log10_true_ppoly)  # for polyprob

            self.true_params = np.array(true_params)

            self.true_loglike = self.loglike(self.true_params)

        #####################################################
        # setting initial parameters
        #####################################################
        if self.start_from == 'true':
            self.init_params = self.true_params
        else:
            self.init_params = igs.estimate_initial_parameters(self)


    def like_obj(self, varparams):
        return -self.loglike(varparams)


    def loglike(self, varparams):

        params = varparams[self.translate_indices]

        num_branches = self.num_branches
        branch_lengths = 10**params[:num_branches]
        mutation_rates = 10**params[num_branches:]
        alphabeta = 10**params[2*num_branches]
        polyprob = 10**params[2*num_branches+1]

        stat_dist = lis.get_stationary_distribution_double_beta(self.freqs,
                self.breaks, self.transition_N, alphabeta, polyprob)

        ll = lis.get_log_likelihood_somatic_newick(
            branch_lengths,
            mutation_rates,
            stat_dist,
            self)

        log_asc_probs = asc.get_locus_asc_probs(branch_lengths,
                mutation_rates, stat_dist, self, self.min_freq)
        log_asc_prob = (log_asc_probs *
                self.asc_ages['count'].values).sum()
        ll -= log_asc_prob

        if self.poisson_like_penalty > 0:
            logmeanascprob, logpoissonlike = self.poisson_log_like(
                    log_asc_probs)
            ll += logpoissonlike * self.poisson_like_penalty
        if self.print_debug:
            print('@@ {:15}'.format(str(ll)), logmeanascprob, ' ', end=' ')
            _util.print_csv_line(varparams)

        return ll


    def locusloglikes(self, varparams, use_counts = False):

        params = varparams[self.translate_indices]

        num_branches = self.num_branches
        branch_lengths = params[:num_branches]
        mutation_rates = params[num_branches:]
        alphabeta = 10**params[2*num_branches]
        polyprob = 10**params[2*num_branches+1]

        stat_dist = lis.get_stationary_distribution_double_beta(self.freqs,
                self.breaks, self.transition_N, alphabeta, polyprob)

        lls = lis.get_locus_log_likelihoods_newick(
            branch_lengths,
            mutation_rates,
            stat_dist,
            self,
            use_counts)
        lls = np.array(lls)

        log_asc_probs = asc.get_locus_asc_probs(branch_lengths,
                mutation_rates, stat_dist, self, self.min_freq)
        for i in range(lls.shape[0]):
            lls[i] -= log_asc_probs[int(self.data['family'].iloc[i])]

        logmeanascprob, logpoissonlike = self.poisson_log_like(
                log_asc_probs)
        lls += logpoissonlike * self.poisson_like_penalty

        return lls


    def inf_bound_like_obj(self, x):
        if self.print_debug and np.any(x < self.lower):
            print('inf:', x)
            return np.inf
        if self.print_debug and np.any(x > self.upper):
            print('inf:', x)
            return np.inf
        return self.like_obj(x)


    def penalty_bound_like_obj(self, x):
        good_x, penalty = self._get_bounds_penalty(x)
        return self.like_obj(good_x) + penalty


    def get_min_max_mults(self):
        '''
        for each varname, need the min and max of all the multipliers
        associated with the varname
        '''
        datp = self.asc_ages.loc[:,self.multipliernames]
        mins = {}
        maxes = {}
        for br in self.branch_names:
            vname = self.varnamedict[br]
            if vname not in mins:
                mins[vname] = set([])
                maxes[vname] = set([])
            mult = self.multiplierdict[br]
            if mult:
                mmin = np.nanmin(datp[mult].values)
                mmax = np.nanmax(datp[mult].values)
            else:
                mmin = 1
                mmax = 1
            mins[vname].add(mmin)
            maxes[vname].add(mmax)

        min_mults = []
        max_mults = []
        for vname in self.varnames:
            min_mults.append(min(mins[vname]))
            max_mults.append(max(maxes[vname]))
        min_mults = np.array(min_mults)
        max_mults = np.array(max_mults)
        return min_mults, max_mults


    def logprior(self, x):
        '''
        x is in varnames space
        '''
        num_varnames = self.num_varnames
        if np.any(x < self.lower):
            return -np.inf
        if np.any(x > self.upper):
            return -np.inf
        # for log-scale mutation parameterization
        if self.log_unif_drift:
            # first num_varnames are drift parameters
            # rather than specify drift in actual log units, just calculate
            # prior this way.
            #logp = -1.0*np.sum(np.log(x[:num_varnames])
            # this from natural-scale drift params
            # note that specifying the bottleneck size as log-uniform is the
            # same as specifying the drift as log-uniform, since
            # log D = log 2 - log B, and log B is uniform.
            # drift now in log-units
            logp = 1.0
        else:
            if self.inverse_bot_priors:
                # if D = 2/B is uniform, f_B(x) \propto x^{-2}, and 
                # log f_B(x) \propto -2log(x)
                #
                # update: if D = 2/B is uniform, the log-density of log B is
                # \propto -x
                pv = np.zeros(x.shape[0])
                pv[self.is_bottleneck_arr] = -1.0*x[self.is_bottleneck_arr]
                pv[~self.is_bottleneck_arr] = x[~self.is_bottleneck_arr]
                logp = np.sum(pv)
            else:
                # if D ~ Unif, the density of log_10 D is \propto 10^x and thus
                # the log-density of log_10 D is \propto log(10) * x.
                # log(10) = 2.3025850929940459
                logp = 2.3025850929940459*np.sum(x[:num_varnames])
        return logp


    def poisson_log_like(self, logascprobs):
        '''
        x is varparams, not branchparams

        returns tuple, harmonic mean of asc. probability and the log of the
        poisson sampling likelihood
        '''
        logmeanascprob = np.mean(logascprobs)
        loglams = logascprobs + np.log(self.genome_size)
        lams = np.exp(loglams)

        logpoissonlikes = -lams + self.asc_ages['count'].values*loglams
        logpoissonlike = logpoissonlikes.sum()
        return logmeanascprob, logpoissonlike


    def log_posterior(self, x):
        '''
        log posterior

        x      parameters in varnames space
        '''
        # wrapping parameters as their absolute values
        num_varnames = self.num_varnames
        # make x variables a copy to avoid changing state
        x = x.copy()
        # make the mutation rates negative
        x[num_varnames:2*num_varnames] = (
                -1.0*np.abs(x[num_varnames:2*num_varnames]))
        # make the last two non-positive
        x[-2:] = -np.abs(x[-2:])
        pr = self.logprior(x)
        if not self.post_is_prior:
            if not np.isfinite(pr):
                like = -np.inf
            else:
                like = -1.0*self.inf_bound_like_obj(x)
            v = pr + like
        else:
            v = pr
        if np.isnan(v):
            v = -np.inf

        if self.print_debug:
            print('@@', v, pr)

        return v

    def penalty_bound_log_posterior(self, x):
        good_x, penalty = self._get_bounds_penalty(x)
        # _get_bounds_penalty() returns a positive penalty, so we subtract it
        # from the log-posterior
        return self.log_posterior(good_x) - penalty


    def _get_bounds_penalty(self, x):
        '''
        takes in branch and mutation length parameters, bounds from
        transitions, and returns parameters that can be evaluated and a
        *positive* penalty

        x     parameters in branch name space
        '''

        lower = self.lower
        upper = self.upper

        scaling = self.upper-self.lower

        lower_penalty = ((np.maximum(self.lower - x, 0.0) / scaling)**2).sum()
        upper_penalty = ((np.maximum(x - self.upper, 0.0) / scaling)**2).sum()

        penalty = lower_penalty + upper_penalty

        good_params = np.maximum(x, self.lower)
        good_params = np.minimum(good_params, self.upper)

        return good_params, penalty

    def _get_pool(self, num_processes, mpi):

        def initializer(*args):

            global inf_data
            inf_data = Inference(*args)

        # MPI takes priority
        if mpi:
            from emcee.utils import MPIPool
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        elif num_processes > 1:
            init_args = [
                    # order important here because of *args above
                    self.data_file,
                    self.transitions_file,
                    self.tree_file,
                    self.init_true_params,
                    self.start_from,
                    self.data_are_freqs,
                    self.genome_size,
                    self.ages_data_fn,
                    self.bottleneck_file,
                    self.poisson_like_penalty,
                    self.min_freq,
                    self.transition_copy,
                    self.transition_buf,
                    self.transition_shape,
                    self.print_debug,
                    self.log_unif_drift,
                    self.inverse_bot_priors,
                    self.post_is_prior,
                    self.lower_drift_limit,
                    self.upper_drift_limit
                    ]

            pool = mp.Pool(num_processes, initializer = initializer,
                    initargs = init_args)

        else:
            pool = None

        return pool



    def _get_initial_mcmc_position(
            self, num_walkers, prev_chain, start_from, init_norm_sd, pool):

        ndim = 2*len(self.varnames) + 2

        if prev_chain is not None:
            # start from previous state
            try:
                prev_chains = pd.read_csv(prev_chain, sep = '\t', header = None, dtype = np.float64)
            except:
                prev_chains = pd.read_csv(prev_chain, sep = '\t', header = 0, dtype = np.float64, comment = '#')
            prev_chains = prev_chains.iloc[-num_walkers:,1:]
            vnames = self.varnames
            prev_chains.columns = ([el+'_l' for el in vnames] +
                    [el+'_m' for el in vnames] + ['root', 'ppoly'])
            init_pos = prev_chains.values
        elif start_from == 'map':
            # start from MAP
            if self.print_debug:
                print('! starting MAP optimization for initial MCMC params')
            init_params = optimize_posterior(self, pool)
            rx = (1+init_norm_sd*npr.randn(ndim*num_walkers))
            rx = rx.reshape((num_walkers, ndim))
            proposed_init_pos = rx*init_params
            proposed_init_pos = np.apply_along_axis(
                    func1d = lambda x: np.maximum(self.lower, x),
                    axis = 1, 
                    arr = proposed_init_pos)
            proposed_init_pos = np.apply_along_axis(
                    func1d = lambda x: np.minimum(self.upper, x),
                    axis = 1, 
                    arr = proposed_init_pos)
            init_pos = proposed_init_pos
        elif start_from == 'prior':
            # start from random points
            if self.print_debug:
                print('! starting MCMC from random point')
            nvarnames = len(self.varnames)
            nparams = self.lower.shape[0]
            rstart = np.zeros((num_walkers, nparams))
            # the mutation and the root parameters will be uniform across their
            # ranges (both are log10 scaled)
            low = np.tile(self.lower[nvarnames:], num_walkers)
            high = np.tile(self.upper[nvarnames:], num_walkers)
            rstart[:, nvarnames:] = npr.uniform(low,
                    high).reshape(num_walkers,-1)
            if self.log_unif_drift:
                # drift parameters now in log10 units
                low = np.tile(self.lower[:nvarnames], num_walkers)
                high = np.tile(self.upper[:nvarnames], num_walkers)
                rstart[:,:nvarnames] = npr.uniform(low, high).reshape(
                        num_walkers, -1)
            else:
                low = np.tile(10**self.lower[:nvarnames], num_walkers)
                high = np.tile(10**self.upper[:nvarnames], num_walkers)
                rstart[:, :nvarnames] = np.log10(npr.uniform(low,high)).reshape(
                        num_walkers,-1)
            init_pos = rstart
        else:
            # use initial guess
            rx = (1+init_norm_sd*npr.randn(ndim*num_walkers))
            rx = rx.reshape((num_walkers, ndim))
            proposed_init_pos = rx*self.init_params
            proposed_init_pos = np.apply_along_axis(
                    func1d = lambda x: np.maximum(self.lower, x),
                    axis = 1, 
                    arr = proposed_init_pos)
            proposed_init_pos = np.apply_along_axis(
                    func1d = lambda x: np.minimum(self.upper, x),
                    axis = 1, 
                    arr = proposed_init_pos)
            init_pos = proposed_init_pos

        return init_pos


    def run_mcmc(
            self, num_iter, num_walkers, num_processes = 1, mpi = False,
            prev_chain = None, start_from = 'initguess', init_norm_sd = 0.2,
            chain_alpha = 2.0, init_pos = None, pool = None):
        global inf_data
        inf_data = self


        # print calling command
        print("# " + " ".join(sys.argv))

        ndim = 2*len(self.varnames) + 2

        if pool is None and (num_processes > 1 or mpi):
            pool = self._get_pool(num_processes, mpi)

        ##########################################################################
        # get initial position: previous chains, MAP, or heuristic guess
        ##########################################################################

        if init_pos is None:
            init_pos = self._get_initial_mcmc_position(num_walkers, prev_chain,
                    start_from, init_norm_sd, pool)

        ##############################################################
        # running normal MCMC   
        ##############################################################
        print(self.header)
        sampler = emcee.EnsembleSampler(num_walkers, ndim, post_clone,
                pool = pool, a = chain_alpha)
        for ps, lnprobs, cur_seed in sampler.sample(init_pos,
                iterations = num_iter, storechain = False):
            _util.print_csv_lines(ps, lnprobs)
            self.transition_data.clear_cache()
        if mpi:
            pool.close()
        print('# mean acceptance across chains:', np.mean(sampler.acceptance_fraction))


    def run_parallel_temper_mcmc(
            self, num_iter, num_walkers, prev_chain, start_from,
            init_norm_sd, pool = None, mpi = False,
            do_evidence = False, num_processes = 1, init_pos = None,
            ntemps = None, parallel_print_all = False, chain_alpha = 2.0):

        '''
        not compatible with make_figures
        '''

        global inf_data
        inf_data = self



        ##############################################################
        # use parallel-tempering
        ##############################################################
        if do_evidence:
            ntemps = 10

        if pool is None and (num_processes > 1 or mpi):
            pool = self._get_pool(num_processes, mpi)

        if init_pos is None:
            init_pos = self._get_initial_mcmc_position(num_walkers, prev_chain,
                    start_from, init_norm_sd, pool)


        ndim = 2*len(self.varnames)+2
        max_temp = np.float('inf') if do_evidence else None
        try:
            import ptemcee
        except ImportError:
            raise ImportError("parallel-tempering MCMC requires ptemcee")
        sampler = ptemcee.Sampler(
                nwalkers = num_walkers,
                dim = ndim,
                logl = logl,
                logp = logp,
                ntemps = ntemps,
                threads = num_processes,
                pool = pool,
                a = chain_alpha,
                Tmax = max_temp)

        print('# betas:')
        for beta in sampler.betas:
            print('#', beta)

        newshape = [ntemps] + list(init_pos.shape)
        init_pos_new = np.zeros(newshape)
        for i in range(ntemps):
            init_pos_new[i,:,:] = init_pos.copy()

        if parallel_print_all:
            self.header = '\t'.join(['chain', 'lnpost'] + self.header_list)
        print(self.header)
        for p, lnprob, lnlike in sampler.sample(init_pos_new,
                iterations=num_iter, storechain = True):
            if parallel_print_all:
                _util.print_parallel_csv_lines(p, lnprob, lnlike)
            else:
                # first chain is chain with temperature 1
                _util.print_csv_lines(p[0], lnprob[0])

        if do_evidence:
            for fburnin in [0.1, 0.25, 0.4, 0.5, 0.75]:
                evidence = sampler.log_evidence_estimate(
                        fburnin=fburnin)
                print('# evidence (fburnin = {}):'.format(fburnin), evidence)

        if mpi:
            pool.close()

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
from functools import partial
import multiprocessing as mp
import numpy.random as npr
import emcee
import warnings
from itertools import izip

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
    valid_sfs = ('prior', 'true', 'map')
    for v in valid_sfs:
        if sf_str == v:
            return v
    warnings.warn('getting starting position from heuristic guess')
    return 'initguess'


class Inference(object):
    def __init__(self,
            data_files,
            tree_files,
            age_files,
            transitions_file,
            true_parameters,
            start_from,
            data_are_freqs,
            genome_size,
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
        self.init_params = None
        self.true_params = None
        self.true_loglike = None
        self.lower = None
        self.upper = None
        self.transition_data = None
        self.bottleneck_data = None

        self.data_files = data_files
        self.tree_files = tree_files
        self.age_files = age_files
        self.data_are_freqs = data_are_freqs
        self.genome_size = genome_size
        self.transitions_file = transitions_file
        self.start_from = _get_valid_start_from(start_from)
        self.init_true_params = true_parameters
        self.bottleneck_file = bottleneck_file
        self.poisson_like_penalty = poisson_like_penalty
        self.min_freq = min_freq
        self.transition_copy = transition_copy
        self.transition_buf = transition_buf
        self.transition_shape = transition_shape
        self.print_debug = print_debug
        self.log_unif_drift = log_unif_drift
        self.inverse_bot_priors = inverse_bot_priors
        self.post_is_prior = post_is_prior
        self.lower_drift_limit = lower_drift_limit,
        self.upper_drift_limit = upper_drift_limit

        ############################################################
        # for each data file, read in data, convert columns to float
        self.data = []
        self.tree_strings = []
        self.plain_trees = []
        self.trees = []
        self.ages = []
        for dat_fn, tree_fn in zip(self.data_files, self.tree_files):
            datp = pd.read_csv(dat_fn, sep = '\t', comment = '#',
                na_values = ['NA'], header = 0)
            for col in datp.columns:
                if col != 'family':
                    datp[col] = datp[col].astype(np.float64)
            datp.sort_values('family', inplace = True)
            self.data.append(datp)

            with open(tree_fn) as fin:
                tree_str = fin.read().strip()
                self.tree_strings.append(tree_str)

            self.plain_trees.append(newick.loads(
                    tree_str,
                    length_parser = ut.length_parser_str,
                    look_for_multiplier = True)[0])



        self._init_transition_data()

        ############################################################
        # a list of the leaves, each corresponding to one of the observed
        # frequency columns
        self.leaf_names = []
        # a list of all of the nodes having a length
        self.branch_names = []
        # a list of the unique multiplier names
        #self.multipliernames = []   # later
        # self.multiplierdict[treeidx][nodename] gives the name of the
        # multiplier for node named nodename in tree indexed treeidx
        self.multiplierdict = []
        # a dict of varnames for each node, for each tree
        self.varnamedict = []
        # a dict to determine whether each varname is a bottleneck
        self.is_bottleneck = {}


        ######################################################################
        # Processing trees for multipliers, var names, and branch names.
        # Going through eash tree, collect branch names, leaf names, and
        # multiplier dict for each tree.

        # Each tree will have its own branch names, leaf names, and
        # multiplierdict (indicates whether or not a node is to be multiplied
        # by some multiplier). Varnames will be added to a single set shared
        # across the different trees. Is_bottleneck must be the same across all
        # trees.

        ######################################################################
        varnames_set = set([])
        multipliernames = []
        for pt in self.plain_trees:
            self.branch_names += [[]]
            self.leaf_names += [[]]
            self.multiplierdict += [{}]
            self.varnamedict += [{}]
            multipliernames += [set([])]
            for node in pt.walk(mode='postorder'):
                if node == pt:
                    continue
                if node.is_leaf:
                    self.leaf_names[-1].append(node.name)
                self.branch_names[-1].append(node.name)
                if node.name in self.multiplierdict[-1]:
                    raise ValueError('node names must be unique in each tree')
                self.multiplierdict[-1][node.name] = node.multipliername
                if node.multipliername is not None:
                    multipliernames[-1].add(node.multipliername)
                varnames_set.add(node.varname)
                self.varnamedict[-1][node.name] = node.varname
                if node.varname in self.is_bottleneck:
                    if self.is_bottleneck[node.varname] != node.is_bottleneck:
                        raise ValueError('nodes with the same variable name must have the same status as bottlenecks in all trees')
                self.is_bottleneck[node.varname] = node.is_bottleneck

        self.multipliernames = [list(multset) for multset in multipliernames]
        self.varnames = sorted(list(varnames_set))
        if self.data_are_freqs:
            self.coverage_names = [None for ln in self.leaf_names]
        else:
            self.coverage_names = [[el + '_n' for el in ln] for ln in self.leaf_names]

        # number of branches having a length
        self.num_branches = [len(bn) for bn in self.branch_names]
        # number of leaves
        # self.num_leaves = len(self.leaf_names)   # this will break initialguess, but that's okay. start from prior
        # indices of the variables with lengths
        
        # branch indices will be different for each tree. during likelihood evaluation, will have to fill branch lengths, mutation rates, etc. independently for each tree
        self.branch_indices = [{br: i for i, br in enumerate(tbn)} for tbn in self.branch_names]
        # indices of just the leaves.... these are apparently never used.
        self.leaf_indices = [{br: i for i, br in enumerate(tln)} for tln in self.leaf_names]

        # each varname gets its own index, since it will correspond to a single branch length and mutation rate
        self.var_indices = {vname: i for i, vname in enumerate(self.varnames)}
        # translate_indices will translate parameters in varnames 'space' to parameters in branch 'space'
        translate_indices = []
        for tbn, tvnd in izip(self.branch_names, self.varnamedict):  # for each tree's branch names...
            translate_indices += [[]]
            for br in tbn:
                br_vname = tvnd[br]
                vname_idx = self.var_indices[br_vname]
                translate_indices[-1].append(vname_idx)
            num_varnames = len(self.varnames)
            for br in tbn:
                br_vname = tvnd[br]
                vname_idx = self.var_indices[br_vname]
                translate_indices[-1].append(num_varnames+vname_idx)
            # for ab
            translate_indices[-1].append(2*num_varnames)
            # for p_poly
            translate_indices[-1].append(2*num_varnames+1)
        self.translate_indices = [np.array(ti, dtype = np.int32) for ti in translate_indices]

        ##################################################### 
        # remove any fixed loci
        ##################################################### 
        fixed = [da.is_fixed(dat, ln, not self.data_are_freqs) for dat, ln in izip(self.data, self.leaf_names)]
        self.data = [dat.loc[~fi,:].reset_index(drop = True) for dat, fi in izip(self.data, fixed)]


        #####################################################
        # rounding down frequencies below min_freq
        #####################################################

        if self.data_are_freqs:
            for datp, tln in izip(self.data, self.leaf_names):
                for ln in tln:
                    if ln in datp.columns:
                        needs_rounding0 = (datp[ln] < self.min_freq)
                        needs_rounding1 = (datp[ln] > 1-self.min_freq)
                        datp.loc[needs_rounding0, ln] = 0.0
                        datp.loc[needs_rounding1, ln] = 1.0

        else:   # self.data_are_freqs == False
            for datp, tln, tcn in izip(self.data, self.leaf_names, self.coverage_names):
                for ln, nn in zip(tln, tcn):
                    freqs = datp[ln].astype(np.float64)/datp[nn]
                    needs_rounding0 = (freqs < self.min_freq)
                    needs_rounding1 = (freqs > 1-self.min_freq)
                    datp.loc[needs_rounding0, ln] = 0
                    datp.loc[needs_rounding1, ln] = (
                            datp.loc[needs_rounding1, nn])


        #####################################################
        # ages data
        #####################################################
        self.asc_ages = []
        self.num_asc_combs = []
        for age_fn, datp in izip(self.age_files, self.data):
            asc_ages = pd.read_csv(age_fn, sep = '\t', comment = '#')
            counts = []
            for fam in asc_ages['family']:
                count = (datp['family'] == fam).sum()
                counts.append(count)
            asc_ages['count'] = counts
            self.asc_ages.append(asc_ages)
            self.num_asc_combs.append(asc_ages.shape[0])

        #####################################################
        # ascertainment
        #####################################################
        self.asc_trees = []
        for tree_str, n_asc_comb in izip(self.tree_strings, self.num_asc_combs):
            asc_tree = newick.loads(
                    tree_str,
                    num_freqs = self.num_freqs,
                    num_loci = 2*n_asc_comb,
                    length_parser = ut.length_parser_str,
                    look_for_multiplier = True)[0]
            self.asc_trees.append(asc_tree)
        self.num_loci = [d.shape[0] for d in self.data]
        self.trees = []
        for tree_str, n_loc in izip(self.tree_strings, self.num_loci):
            self.trees.append(newick.loads(tree_str,
                    num_freqs = self.num_freqs,
                    num_loci = n_loc,
                    length_parser = ut.length_parser_str,
                    look_for_multiplier = True)[0])

        #####################################################
        # leaf allele frequency likelihoods
        #####################################################
        self.leaf_likes = []
        for datp, n_names, ln in izip(self.data, self.coverage_names, self.leaf_names):
            if not self.data_are_freqs:
                count_dat = datp.loc[:,ln].values.astype(
                        np.float64)
                ns_dat = datp.loc[:,n_names].values.astype(
                        np.float64)
                leaf_likes = _binom.get_binom_likelihoods_cython(count_dat,
                        ns_dat, self.freqs)
            else:
                freq_dat = datp.loc[:,leaf_names].values.astype(
                        np.float64)
                leaf_likes = _binom.get_nearest_likelihoods_cython(freq_dat,
                        self.freqs)
            leaf_likes_dict = {}
            for i, leaf in enumerate(ln):
                leaf_likes_dict[leaf] = leaf_likes[:,i,:].copy('C')
            self.leaf_likes.append(leaf_likes_dict)

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

        # (no longer setting initial parameters here)

        # end __init__


    def _init_transition_data(self):
        ############################################################
        # add transition data
        if self.transition_copy is None:
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


    def like_obj(self, varparams):
        return -self.loglike(varparams)


    def loglike(self, varparams):

        ll = 0.0
        alphabeta, polyprob = 10**varparams[-2:]

        stat_dist = lis.get_stationary_distribution_double_beta(self.freqs,
                self.breaks, self.transition_N, alphabeta, polyprob)

        for i in range(len(self.data)):
            trans_idxs = self.translate_indices[i]
            params = varparams[trans_idxs]
            num_branches = self.num_branches[i]
            branch_lengths = 10**params[:num_branches]
            mutation_rates = 10**params[num_branches:]
            asc_ages = self.asc_ages[i]

            tll = lis.get_log_likelihood_somatic_newick(
                branch_lengths,
                mutation_rates,
                stat_dist,
                self,
                i)

            log_asc_probs = asc.get_locus_asc_probs(branch_lengths,
                    mutation_rates, stat_dist, self, self.min_freq, i)
            log_asc_prob = (log_asc_probs *
                    asc_ages['count'].values).sum()
            tll -= log_asc_prob

            if self.poisson_like_penalty > 0:
                logmeanascprob, logpoissonlike = self.poisson_log_like(
                        log_asc_probs, i)
                tll += logpoissonlike * self.poisson_like_penalty

            ll += tll

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
        from collections import defaultdict
        mins = defaultdict(list)
        maxes = defaultdict(list)
        for agep, tmult, tbn, vnd, multdict in izip(self.asc_ages, self.multipliernames, self.branch_names, self.varnamedict, self.multiplierdict):
            agep = agep.loc[:,tmult]
            for br in tbn:
                vname = vnd[br]
                mult = multdict[br]
                if mult:
                    mmin = np.nanmin(agep[mult].values)
                    mmax = np.nanmax(agep[mult].values)
                else:
                    mmin = 1
                    mmax = 1
                mins[vname].append(mmin)
                maxes[vname].append(mmax)

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


    def poisson_log_like(self, logascprobs, tree_idx):
        '''
        x is varparams, not branchparams

        returns tuple, harmonic mean of asc. probability and the log of the
        poisson sampling likelihood
        '''
        logmeanascprob = np.mean(logascprobs)
        loglams = logascprobs + np.log(self.genome_size)
        lams = np.exp(loglams)

        logpoissonlikes = -lams + self.asc_ages[tree_idx]['count'].values*loglams
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
        else:
            # prior is the new default
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

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
from functools import partial
import multiprocessing as mp
import warnings
try:
    from itertools import izip
except ImportError:  # py3
    izip = zip

import numpy as np
import scipy.optimize as opt
import pandas as pd
import numpy.random as npr
import emcee
from scipy.special import gammaln

from mope import newick
from mope import util as ut
from mope import data as da
from mope import _binom
from mope import _util
from mope.pso import pso
from mope.simulate import get_parameters
from mope import ascertainment as asc
from mope.dfe import CygnusDistribution
from mope import likelihoods as li
from mope import transition_data_2d as td2d
from mope import transition_data_bottleneck as tdb
from mope import params as par

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
               swarmsize=swarmsize, minfunc=minfunc,
               init_params=inf_data.init_params,
               init_params_weight=swarm_init_weight,
               pool=pool, processes=num_processes)

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

def _get_header(inf):
    length_names = [el+'_l' for el in inf.varnames]
    if inf.selection_model:
        mutsel_names = ['el_a' for el in inf.varnames] + inf.dfe.param_names
    else:
        mutsel_names = [el+'_m' for el in inf.varnames]
    header_list = (['ll'] + length_names + mutsel_names +
            ['log10ab', 'log10polyprob'])
    if inf.selection_model:
        header_list += ['alpha_{}'.format(bp) for bp in inf.unique_positions]
    header = '\t'.join(header_list)
    return header


def _get_likelihood_limits(inf):
    num_varnames = inf.num_varnames
    ndim = 2*num_varnames + 2
    if inf.selection_model:
        ndim += inf.dfe.nparams + inf.num_unique_positions

    # (this multiplied by 0.1 for pointmass at zero)
    # this gives the log10-uniform an additional order of magnitude for
    # point mass at zero
    min_allowed_len = 0.1*max(inf.transition_data.get_min_coal_time(),
            inf.lower_drift_limit)
    max_allowed_len = min(inf.transition_data.get_max_coal_time(),
            inf.upper_drift_limit)

    min_allowed_bottleneck = 2
    max_allowed_bottleneck = 500

    min_ab = -9
    max_ab = 0
    min_polyprob = -9
    max_polyprob = 0

    lower_len = min_allowed_len / inf.min_mults
    upper_len = max_allowed_len / inf.max_mults

    is_bottleneck_arr = np.array(
            [inf.is_bottleneck[vn] for vn in inf.varnames], dtype = bool)
    inf.is_bottleneck_arr = is_bottleneck_arr
    lower_len[is_bottleneck_arr] = min_allowed_bottleneck
    upper_len[is_bottleneck_arr] = max_allowed_bottleneck

    # convert len limits to log10... above they should be natural scale
    lower_len = np.log10(lower_len)
    upper_len = np.log10(upper_len)

    if inf.selection_model:
        # This is the lower limit of the log10-mean of the exponential
        # distribution of selection coefficients for the negative selection
        # coefficients.
        lower_alpha_neg = -3
        # Don't let the mean of the (untruncated) exponential distribution
        # exceed half of the maximum value
        max_mutsel_rate = inf.transition_data.get_max_mutsel_rate()
        max_mean = 0.5*max_mutsel_rate
        upper_alpha_neg = min(-1, np.log10(max_mean))


        # For the selection model, lower_sel and upper_sel here contain the
        # lower and upper limits for the parameters of the DFEs, not the alpha
        # (2Ns) values for individual base-pair positions. The general order of
        # the parameters is:
        #   1. length parameters  (num_varnames)
        #   2. mean alphas for negative selection coefficient distributions
        #      (num_varnames)
        #   3. other DFE parameters (here, for this DFE, log10 pos/neg
        #      meanalpha ratio, log10 prob that a site has a negative vs.
        #      positive selection coefficient)
        #   4. root distribution parameters (2)
        #   5. selection coefficients for individual base-pair positions. These
        #      correspond to the values for the ontogenetic process
        #      inf.selection_focal_varname. For other processes, have to
        #      multiply the value by meanalpha(i)/meanalpha(focalprocess)
        lower_sel = np.array(([lower_alpha_neg]*num_varnames
                              + list(inf.dfe.lower_limits)))
        upper_sel = np.array([upper_alpha_neg]*num_varnames
                              + list(inf.dfe.upper_limits))

        # For the selection coefficients, the values between -5 and -3 will be
        # translated to 0 (neutral).
        # the upper and lower limits for the selection coefficients themselves
        log10_alpha_min = -5
        log10_alpha_max = np.log10(max_mutsel_rate)
        lower_alpha = np.repeat(log10_alpha_min, inf.num_unique_positions)
        upper_alpha = np.repeat(log10_alpha_max, inf.num_unique_positions)

        lower = np.concatenate((lower_len, lower_sel, (min_ab,min_polyprob),
                                lower_alpha))
        upper = np.concatenate((upper_len, upper_sel, (max_ab,max_polyprob),
                                upper_alpha))
    else:
        min_mut = max(-8,
                np.log10(inf.transition_data.get_min_mutsel_rate()))
        max_mut = min(-1,
                np.log10(inf.transition_data.get_max_mutsel_rate()))
        lower_mut = np.repeat(min_mut, num_varnames)
        upper_mut = np.repeat(max_mut, num_varnames)
        lower = np.concatenate((lower_len, lower_mut, (min_ab,min_polyprob)))
        upper = np.concatenate((upper_len, upper_mut, (max_ab,max_polyprob)))

    return lower, upper, ndim



class Inference(object):
    def __init__(
            self,
            data_files,
            tree_files,
            age_files,
            transitions_file,
            true_parameters,
            start_from,
            data_are_freqs,
            genome_size,
            bottleneck_file=None,
            poisson_like_penalty=1.0,
            min_freq=0.001,
            transition_copy=None,
            transition_buf=None,
            transition_shape=None,
            print_debug=False,
            log_unif_drift=False,
            inverse_bot_priors=False,
            post_is_prior=False,
            lower_drift_limit=1e-3,
            upper_drift_limit=3,
            min_phred_score=None,
            selection_model=False):

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
        self.lower_drift_limit = lower_drift_limit
        self.upper_drift_limit = upper_drift_limit
        self.min_phred_score = min_phred_score
        self.selection_model = selection_model

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
            if self.selection_model and 'position' not in datp.columns:
                raise ValueError('The selection model requires a "position" '
                                 'column in each dataframe. Could not find '
                                 'one in {}'.format(dat_fn))
            for col in datp.columns:
                if col not in ['family', 'position']:
                    datp[col] = datp[col].astype(np.float64)
            if self.selection_model:
                datp.sort_values(['family', 'position'], inplace = True)
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
            self.leaf_names += [[]]
            self.multiplierdict += [{}]
            self.varnamedict += [{}]
            self.branch_names += [[]]
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

        # each varname gets its own index, since it will correspond to a single
        # branch length and mutation rate
        self.var_indices = {vname: i for i, vname in enumerate(self.varnames)}
        # translate_indices will translate parameters in varnames 'space' to
        # parameters in branch 'space'
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
        self.translate_indices = [
            np.array(ti, dtype=np.int32) for ti in translate_indices]

        ##################################################### 
        # remove any fixed loci
        ##################################################### 
        fixed = [da.is_fixed(dat, ln, not self.data_are_freqs)
                 for dat, ln in izip(self.data, self.leaf_names)]
        self.data = [dat.loc[~fi, :].reset_index(
            drop=True) for dat, fi in izip(self.data, fixed)]


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
        # get unique positions if selection model
        #####################################################
        if self.selection_model:
            unique_positions = set([])
            for datp in self.data:
                unique_positions.update(
                    set(datp['position'].tolist()))
            unique_positions = sorted(unique_positions)
            self.unique_positions = unique_positions
            self.num_unique_positions = len(unique_positions)

            for datp in self.data:
                datp['position_idx'] = datp['position'].apply(
                    lambda x: unique_positions.index(x))

        #####################################################
        # ages data
        #####################################################
        self.asc_ages = []
        self.num_asc_combs = []
        for age_fn, datp, multnames in izip(self.age_files, self.data,
                                            self.multipliernames):
            asc_ages = pd.read_csv(age_fn, sep='\t', comment='#')
            counts = []
            for fam in asc_ages['family']:
                count = (datp['family'] == fam).sum()
                counts.append(count)
            asc_ages['count'] = counts
            self.asc_ages.append(asc_ages)
            self.num_asc_combs.append(asc_ages.shape[0])
            # if multipliers are not in data, add them to the data
            for mult in multnames:
                if mult not in datp:
                    multvals = []
                    for fam in datp['family']:
                        v = asc_ages.loc[
                            asc_ages['family'] == fam,mult].iloc[0]
                        assert v is not None
                        multvals.append(v)
                    datp[mult] = multvals

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



        #####################################################
        # tree structures
        #####################################################
        self.trees = []
        for tree_str, n_loc in izip(self.tree_strings, self.num_loci):
            if self.selection_model:
                # In the selection model, we are going to calculate three
                # probabilities simultaneously: (1) the usual allele frequency
                # likelihoods, (2) the probabilities that the frequencies are
                # 0 in all leaves, and (3) the probabilities that the
                # frequencies are all 1 in all leaves.
                n_loc *= 3
            self.trees.append(newick.loads(tree_str,
                    num_freqs = self.num_freqs,
                    num_loci = n_loc,
                    length_parser = ut.length_parser_str,
                    look_for_multiplier = True)[0])

        #####################################################
        # leaf allele frequency likelihoods
        #####################################################
        self.leaf_likes = []
        for datp, n_names, ln in izip(self.data, self.coverage_names,
                                      self.leaf_names):
            if not self.data_are_freqs:
                count_dat = datp.loc[:,ln].values.astype(
                        np.float64)
                ns_dat = datp.loc[:,n_names].values.astype(
                        np.float64)
                # get_binom_like...() wants -1 for missing value / assumed perfect counts
                phred_score_param = -1.0 if self.min_phred_score is None else self.min_phred_score
                leaf_likes = _binom.get_binom_likelihoods_cython(
                    count_dat, ns_dat, self.freqs, phred_score_param)
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
        # if selection, get DFE, set focal branch
        #
        #####################################################
        if self.selection_model:
            self.dfe = CygnusDistribution()
            # The focal branch is the branch that the selection coefficients
            # correspond to. The values for other branches are taken from the
            # ratios of mean-alpha parameters. We will arbitrarily choose the
            # first varname as the focal process.
            self.focal_branch = self.varnames[0]
            self.focal_branch_idx = 0


        #####################################################
        # header
        #####################################################
        self.header = _get_header(self)


        #####################################################
        # making bounds for likelihood functions
        #####################################################
        self.num_varnames = len(self.varnames)

        # For the selection model, the order of the parameters is:
        #   1. length parameters  (num_varnames)
        #   2. mean alphas for negative selection coefficient distributions
        #      (num_varnames)
        #   3. other DFE parameters (here, for this DFE, log10 pos/neg alpha
        #      ratio, log10 prob that a site has a negative vs. positive
        #      selection coefficient)
        #   4. selection coefficients for individual base-pair positions. These
        #      correspond to the values for the ontogenetic process
        #      inf.focal_branch. For other processes, have to
        #      multiply the value by meanalpha(i)/meanalpha(focalprocess)

        # ndim contains the total number of variables in the chain, so it
        # includes the selection coefficients as well, for the selection model.
        lower, upper, ndim = _get_likelihood_limits(self)
        if self.selection_model:
            self.num_nonsel_params = ndim - self.num_unique_positions
            self.max_alpha = self.transition_data.get_max_mutsel_rate()
            # set range of alphas that are intepreted as zero

            #lower = np.concatenate((lower_len, lower_sel, (min_ab,min_polyprob),
            #                        lower_alpha))


        # self.lower and .upper give the bounds for prior distributions
        self.lower = lower
        self.upper = upper
        self.ndim = ndim

        print('# self.lower:')
        for el in self.lower:
            print('# ', el)
        print('# self.upper:')
        for el in self.upper:
            print('# ', el)

        #####################################################
        # true parameters, if specified (for sim. data)
        #####################################################
        if true_parameters:
            if self.selection_model:
                raise NotImplementedError(
                    'still need to implement true-parameter evaluation '
                    'for selection model')
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

        # end __init__


    def _init_transition_data(self):
        ############################################################
        # add transition data
        if self.transition_copy is None:
            self.transition_data = td2d.TransitionData(
                self.transitions_file, memory=True,
                selection=self.selection_model)
        else:
            self.transition_data = td2d.TransitionDataCopy(
                transition_copy, transition_buf, transition_shape,
                selection=self.selection_model)
        self.freqs = self.transition_data.get_bin_frequencies()
        self.num_freqs = self.freqs.shape[0]
        self.breaks = self.transition_data.get_breaks()
        self.transition_N = self.transition_data.get_N()

        ############################################################
        # add optional bottleneck data
        if self.bottleneck_file is not None:
            self.bottleneck_data = tdb.TransitionDataBottleneck(
                    self.bottleneck_file, memory=True)
        else:
            self.bottleneck_data = None


    def like_obj(self, varparams):
        return -self.loglike(varparams)
    

    def loglike(self, orig_params):

        ll = 0.0
        bad_input = False
        if self.selection_model:
            varparams = orig_params[:-self.num_unique_positions]

            # Any input log10 selection parameter 
            log10_sel_params = orig_params[-self.num_unique_positions:]
            zero_filt = log10_sel_params < self.min_log10_sel_param
            log10_sel_params[zero_filt] = -np.inf
            sel_params = 10**log10_sel_params

            dfe_start = 2*self.num_varnames        # start of the DFE params
            dfe_params = varparams[dfe_start:dfe_start+self.dfe.nparams]

            nvn = self.num_varnames
            mutsel_rates_varpar = varparams[nvn:2*nvn]
            focal_alpha_neg = mutsel_rates_varpar[self.focal_branch_idx]
            relative_alphas = mutsel_rates_varpar / focal_alpha_neg
            # The shape of locus_alpha_values is
            # (self.num_varnames, self.num_unique_positions), so
            # locus_alpha_values[i,j] gives the selection rate for varname i
            # and unique locus j.
            locus_alpha_values = sel_params * relative_alphas[:, np.newaxis]
            if np.any(locus_alpha_values > self.max_alpha):
                ll = -np.inf
                print('@@ {:15}'.format(str(ll)), end=' ')
                _util.print_csv_line(varparams)
                return ll
        else:
            varparams = orig_params
        alphabeta, polyprob = 10**varparams[-2:]

        stat_dist = li.get_stationary_distribution_double_beta(self.freqs,
                self.breaks, self.transition_N, alphabeta, polyprob)

        for i in range(len(self.data)):
            trans_idxs = self.translate_indices[i]
            trans_idxs_len = trans_idxs[:self.num_branches[i]]
            params = varparams[trans_idxs]
            num_branches = self.num_branches[i]
            branch_lengths = 10**params[:num_branches]
            mutsel_rates = 10**params[num_branches:]
            asc_ages = self.asc_ages[i]


            if self.selection_model:
                # get alpha values for individual loci
                locus_alpha_values_p = locus_alpha_values[trans_idxs_len, :]
                import pdb; pdb.set_trace()
                # calculate log-likelihood and asc prob
                ped_ll, ped_log_asc_prob = li.get_log_l_and_asc_prob_selection(
                    branch_lengths,
                    locus_alpha_values_p,
                    stat_dist,
                    self,
                    i,
                    self.min_freq
                )
            else:
                ped_ll = li.get_log_likelihood_somatic_newick(
                    branch_lengths,
                    mutsel_rates,
                    stat_dist,
                    self,
                    i)
                log_asc_probs = asc.get_locus_asc_probs(branch_lengths,
                                                        mutsel_rates,
                                                        stat_dist, self,
                                                        self.min_freq, i)
                ped_log_asc_prob = (log_asc_probs *
                        asc_ages['count'].values).sum()

            # because everything is in log space, rather than add and subtract
            # each locus_ll and locus_log_asc_prob, we can add them all
            # (ped_ll) and subtract them all (leg_log_asc_prob) all at once.
            ped_ll -= ped_log_asc_prob
            ped_ll = ped_ll.sum()

            if self.poisson_like_penalty > 0 and not self.selection_model:
                logmeanascprob, logpoissonlike = self.poisson_log_like(
                        log_asc_probs, i)
                ped_ll += logpoissonlike * self.poisson_like_penalty

            ll += ped_ll

        if self.selection_model:
            dfe_ll = self.dfe.get_loglike(dfe_params, focal_alpha_neg,
                                          sel_params)
            ll += dfe_ll

        if self.print_debug:
            if self.selection_model:
                print('@@ {:15}'.format(str(ll)), end=' ')
                _util.print_csv_line(varparams)
            else:
                print('@@ {:15}'.format(str(ll)), logmeanascprob, ' ', end=' ')
                _util.print_csv_line(varparams)

        return ll


    def debug_loglike_components(self, varparams):

        ll = 0.0
        alphabeta, polyprob = 10**varparams[-2:]

        stat_dist = li.get_stationary_distribution_double_beta(self.freqs,
                self.breaks, self.transition_N, alphabeta, polyprob)

        comps = []
        for i in range(len(self.data)):
            trans_idxs = self.translate_indices[i]
            params = varparams[trans_idxs]
            num_branches = self.num_branches[i]
            branch_lengths = 10**params[:num_branches]
            mutation_rates = 10**params[num_branches:]
            asc_ages = self.asc_ages[i]

            tll = li.get_log_likelihood_somatic_newick(
                branch_lengths,
                mutation_rates,
                stat_dist,
                self,
                i)

            tcomps = [tll]

            log_asc_probs = asc.get_locus_asc_probs(branch_lengths,
                    mutation_rates, stat_dist, self, self.min_freq, i)
            log_asc_prob = (log_asc_probs *
                    asc_ages['count'].values).sum()
            tll -= log_asc_prob

            tcomps += [log_asc_probs, asc_ages['count'].copy().values]

            if self.poisson_like_penalty > 0:
                logmeanascprob, logpoissonlike = self.poisson_log_like(
                        log_asc_probs, i)
                tll += logpoissonlike * self.poisson_like_penalty

                tcomps += [logmeanascprob, logpoissonlike]
            

            tcomps += [tll]
            tcomps += [self.logprior(varparams)]
            ll += tll
            comps.append(tcomps)

        return comps


    def locusloglikes(self, varparams, use_counts = False):

        params = varparams[self.translate_indices]

        num_branches = self.num_branches
        branch_lengths = params[:num_branches]
        mutation_rates = params[num_branches:]
        alphabeta = 10**params[2*num_branches]
        polyprob = 10**params[2*num_branches+1]

        stat_dist = li.get_stationary_distribution_double_beta(self.freqs,
                self.breaks, self.transition_N, alphabeta, polyprob)

        lls = li.get_locus_log_likelihoods_newick(
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

            # everything is uniform on the scale of self.upper and self.lower
            return np.sum(np.log(1.0/(self.upper-self.lower)))
        else:
            u = self.upper
            l = self.lower
            drift_x = x[:num_varnames]
            mut_x = x[num_varnames:2*num_varnames]
            root_x = x[2*num_varnames:]
            #ln10 = 2.3025850929940459
            #lnln10 = 0.834032445247956 
            drift_part_arr = (drift_x*2.3025850929940459 +
                    0.834032445247956 - np.log(10**u-10**l))
            u_m = self.upper[num_varnames:2*num_varnames]
            l_m = self.lower[num_varnames:2*num_varnames]
            u_r = self.upper[2*num_varnames:]
            l_r = self.lower[2*num_varnames:]
            mut_part = np.sum(np.log(1.0/(u_m-l_m)))
            root_part = np.sum(np.log(1.0/(u_r-l_r)))

            if self.inverse_bot_priors:
                # if the drift D for bottleneck B, where D = 2/B, is
                # Uniform(2/u,2/l), where l and u are the lower and upper
                # bottleneck size limits, the prior density on B wrt z is
                #
                #    2/( (2/l-2/u) * z^2) = l*u/( (u-l) * z^2)
                #
                # and the log-density is
                #
                #    log l + log u - ln(u-l) - 2 ln z.
                #
                # The below assumes that self.is_bottleneck_arr is a boolean
                # array over only the drift components. Might be wrong about
                # this, in which case this will throw an exception.
                l_b = self.lower[self.is_bottleneck_arr]
                u_b = self.upper[self.is_bottleneck_arr]
                x_b = drift_x[self.is_bottleneck_arr]
                drift_part_arr[self.is_bottleneck_arr] = (
                        np.log(l_b) + np.log(u_b) + - np.log(u_b-l_b) -
                        2*np.log(x_b))
            drift_part = np.sum(drift_part_arr)

            return drift_part + mut_part + root_part


    def poisson_log_like(self, logascprobs, tree_idx):
        '''
        x is varparams, not branchparams

        returns tuple, harmonic mean of asc. probability and the log of the
        poisson sampling likelihood
        '''
        logmeanascprob = np.mean(logascprobs)
        loglams = logascprobs + np.log(self.genome_size)
        lams = np.exp(loglams)

        logpoissonlikes = (-lams +
                self.asc_ages[tree_idx]['count'].values*loglams -
                gammaln(self.asc_ages[tree_idx]['count'].values+1))

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
        # TODO: update the initializer for the updated Inference API

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
                self.data_files,
                self.tree_files,
                self.age_files,
                self.transitions_file,
                self.init_true_params,
                self.start_from,
                self.data_are_freqs,
                self.genome_size,
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
                self.upper_drift_limit,
                self.min_phred_score,
                self.selection_model
            ]


            pool = mp.Pool(num_processes, initializer = initializer,
                    initargs = init_args)

        else:
            pool = None

        return pool



    def _get_initial_mcmc_position(
            self, num_walkers, prev_chain, start_from, init_norm_sd, pool,
            logp, logl):

        ndim = self.ndim

        if prev_chain is not None:
            # start from previous state
            try:
                prev_chains = pd.read_csv(prev_chain, sep='\t', header=None,
                                          dtype=np.float64)
            except:
                prev_chains = pd.read_csv(prev_chain, sep='\t', header=0,
                                          dtype=np.float64, comment='#')
            prev_chains = prev_chains.iloc[-num_walkers:,1:]
            vnames = self.varnames

            mutsel_suffix = '_a' if self.selection_model else '_m'
            cols = ([el+'_l' for el in vnames] +
                    [el+mutsel_suffix for el in vnames])
            if self.selection_model:
                cols += self.dfe.param_names
            cols += ['root', 'ppoly']
            prev_chains.columns = cols

            init_pos = prev_chains.values

        else:
            while True:
                nvarnames = len(self.varnames)
                nparams = self.lower.shape[0]
                rstart = np.zeros((num_walkers, nparams))
                # the mutation and the root parameters will be uniform across
                # their ranges (both are log10 scaled)
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
                    rstart[:, :nvarnames] = np.log10(
                        npr.uniform(low,high)).reshape(num_walkers,-1)
                init_pos = rstart
                logl_val = np.array([logl(p) for p in init_pos])
                logp_val = np.array([logp(p) for p in init_pos])
                if (np.all(np.isfinite(logl_val)) and
                        np.all(np.isfinite(logp_val))):
                    break  # successfully found starting position within bounds
                else:
                    if not np.all(np.isfinite(logl_val)):
                        print(
                            '# warning: attempted start at initial position where log-likelihood is not finite')
                        print('# bad init position (logl)', file=sys.stderr)
                        print(init_pos, file=sys.stderr)
                    if not np.all(np.isfinite(logp_val)):
                        print(
                            '# warning: attempted start at initial position where prior is not finite')
                        print('# bad init position (logprior)', file=sys.stderr)
                        print(init_pos, file=sys.stderr)

        return init_pos


    def run_mcmc(
            self, num_iter, num_walkers, num_processes=1, mpi=False,
            prev_chain=None, start_from='initguess', init_norm_sd=0.2,
            chain_alpha=2.0, init_pos=None, pool=None):
        global inf_data
        inf_data = self


        # print calling command
        print("# " + " ".join(sys.argv))

        ndim = self.ndim

        if pool is None and (num_processes > 1 or mpi):
            pool = self._get_pool(num_processes, mpi)

        ##########################################################################
        # get initial position: previous chains, MAP, or heuristic guess
        ##########################################################################

        if init_pos is None:
            init_pos = self._get_initial_mcmc_position(num_walkers, prev_chain,
                    start_from, init_norm_sd, pool, logp, logl)

        ##############################################################
        # running normal MCMC   
        ##############################################################
        print(self.header)
        sampler = emcee.EnsembleSampler(num_walkers, ndim, post_clone,
                                        pool=pool, a=chain_alpha)
        for ps, lnprobs, cur_seed in sampler.sample(
                init_pos, iterations = num_iter, storechain = False):
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
                    start_from, init_norm_sd, pool, logp, logl)


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
        if not do_evidence:
            for p, lnprob, lnlike in sampler.sample(init_pos_new,
                    iterations=num_iter, storechain = True):
                if parallel_print_all:
                    _util.print_parallel_csv_lines(p, lnprob, lnlike)
                else:
                    # first chain is chain with temperature 1
                    _util.print_csv_lines(p[0], lnprob[0])

        else:
            evidence_every = 2000
            num_completed = 0
            p = init_pos_new
            while num_completed < num_iter:
                to_do = min(evidence_every, num_iter-num_completed)
                for p, lnprob, lnlike in sampler.sample(p,
                        iterations=to_do, storechain = True):
                    if parallel_print_all:
                        _util.print_parallel_csv_lines(p, lnprob, lnlike)
                    else:
                        # first chain is chain with temperature 1
                        _util.print_csv_lines(p[0], lnprob[0])
                num_completed += to_do
                for fburnin in [0.1, 0.25, 0.4, 0.5, 0.75]:
                    evidence = sampler.log_evidence_estimate(
                            fburnin=fburnin)
                    print('# evidence after {} iterations (fburnin = {}): {}'.format(num_completed, fburnin, evidence))


        if mpi:
            pool.close()

from __future__ import division
import argparse
import sys
import numpy as np
import scipy.optimize as opt
import pandas as pd
import likelihoods as lis
import transition_data_mut as tdm
import transition_data_bottleneck as tdb
import params as par
import initialguess as igs
from functools import partial
import multiprocessing as mp

import newick
import util as ut
import data as da
import _binom
import _util
from pso import pso
from simulate import get_parameters
import ascertainment as asc

'''
To include in this class:

- data
- original data size
- ascertainment (true/false)
- ascertainment data size
- ascertainment tree, if needed
- asc_ages, if needed
- asc_counts
- true parameters, if any
- data_are_counts
- fst filter
- start_from_true
- num processes
- genome size
- transitions
- freqs
- nfreqs
- breaks
- transitions_N
- tree
- tree str
- data columns
- branch_names 
- branch_indices
- num_branches
- leaf_names
- leaf_indices
- num_leaves
- coverage (int), if needed
- leaf likes dict
- likelihood objective
- initial parameters
    (either estimated or true)
- optimization parameters:
    - num threads 
    - num particles

'''

class Inference(object):
    def __init__(self,
            data_file, transitions_file, tree_file, true_parameters,
            start_from_true, data_are_freqs, fst_filter_frac,
            genome_size, num_processes, ascertainment, print_res, ages_data_fn,
            bottleneck_file = None, poisson_like_penalty = 1.0,
            min_freq = 0.001, transition_copy = None, transition_buf = None,
            transition_shape = None):

        self.asc_tree = None
        self.asc_num_loci = None
        self.asc_ages = None
        self.asc_counts = None
        self.fst_filter = None
        self.coverage = None
        self.init_params = None
        self.true_params = None
        self.true_loglike = None
        self.lower = None
        self.upper = None
        self.transition_data = None
        self.bottleneck_data = None

        self.num_processes = num_processes
        self.data_are_counts = (not data_are_freqs)
        self.genome_size = genome_size
        self.tree_file = tree_file
        self.transitions_file = transitions_file
        self.data_file = data_file
        self.ascertainment = ascertainment
        self.start_from_true = start_from_true
        self.num_processes = num_processes
        self.bottleneck_file = bottleneck_file
        self.poisson_like_penalty = poisson_like_penalty
        self.min_freq = min_freq
        self.ages_data_fn = ages_data_fn
        self.print_res = print_res

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

        if self.data_are_counts:
            self.count_names = [el + '_n' for el in self.leaf_names]
        else:
            self.count_names = None

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
        # fst filtering
        #####################################################
        self.fst_filter = fst_filter_frac
        if fst_filter_frac:
            # remove high-fst loci
            self.fst_filter = fst_filter_frac
            is_high_fst = da.find_high_fst_indices(self.data, self.leaf_names,
                                                   fst_filter_frac,
                                                   self.data_are_counts)
            self.data = self.data.loc[~is_high_fst,:]
            self.data = self.data.reset_index(drop = True)

        #####################################################
        # rounding down frequencies below min_freq
        #####################################################

        # round down if allele frequencies are taken as true
        if not self.data_are_counts:
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

        # otherwise, ... round down to zero?
        else:   # self.data_are_counts == True
            for ln, nn in zip(self.leaf_names, self.count_names):
                freqs = self.data[ln].astype(np.float64)/self.data[nn]
                needs_rounding0 = (freqs < self.min_freq)
                needs_rounding1 = (freqs > 1-self.min_freq)
                self.data.loc[needs_rounding0, ln] = 0
                self.data.loc[needs_rounding1, ln] = (
                        self.data.loc[needs_rounding1, nn])

        #####################################################
        # randomizing alleles
        # (flip the major and minor allele, randomly)
        #####################################################
        #self.data = da.randomize_alleles(self.data, self.leaf_names,
        #        self.data_are_counts)
        #self.data = self.data.reset_index(drop = True)

        #####################################################
        # ascertainment
        #####################################################
        if ascertainment:
            # remove fixed loci
            fixed = da.is_fixed(self.data, self.leaf_names,
                    self.data_are_counts)
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
        if self.data_are_counts:
            n_names = self.count_names
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
        # print header
        #####################################################

        if print_res:
            length_names = [el+'_l' for el in self.varnames]
            mut_names = [el+'_m' for el in self.varnames]
            header_list = (['ll'] + length_names + mut_names +
                    ['log10ab', 'log10polyprob'])
            print '\t'.join(header_list)


        #####################################################
        # making various bounded like_obj functions
        #####################################################

        #----------------------------------------------------
        # set bounds
        #----------------------------------------------------
        num_varnames = len(self.varnames)
        self.num_varnames = num_varnames
        ndim = 2*num_varnames + 2

        # hard-coded bounds
        #min_allowed_len = 1e-6
        min_allowed_len = max(self.transition_data.get_min_coal_time(), 1e-6)
        max_allowed_len = 3
        min_allowed_bottleneck = 2
        max_allowed_bottleneck = 500
        min_mut = -8
        max_mut = -1
        min_ab = -9
        max_ab = 0
        min_polyprob = -9
        max_polyprob = 0

        lower_len = min_allowed_len / self.min_mults
        upper_len = max_allowed_len / self.max_mults
        is_bottleneck_arr = np.array(
                [self.is_bottleneck[vn] for vn in self.varnames], dtype = bool)
        lower_len[is_bottleneck_arr] = min_allowed_bottleneck
        upper_len[is_bottleneck_arr] = max_allowed_bottleneck

        # boolean array indexed by varname indices
        self.is_bottleneck_arr = is_bottleneck_arr

        lower_mut = np.repeat(min_mut, num_varnames)
        upper_mut = np.repeat(max_mut, num_varnames)
        lower = np.concatenate((lower_len, lower_mut, (min_ab,min_polyprob)))
        upper = np.concatenate((upper_len, upper_mut, (max_ab,max_polyprob)))
        self.lower = lower
        self.upper = upper

        self.bound_lower = np.concatenate((0.5*lower_len, 2*lower_mut,
            (2*min_ab,2*min_polyprob)))
        self.bound_upper = np.concatenate((2*upper_len, upper_mut+2, (max_ab+2,
            max_polyprob+2)))

        bound = partial(ut.bound, lower = self.bound_lower,
                upper = self.bound_upper)
        unbound = partial(ut.unbound, lower = self.bound_lower,
                upper = self.bound_upper)

        self.bound = bound
        self.unbound = unbound

        #####################################################
        # true parameters
        #####################################################
        if true_parameters:
            true_bls, true_mrs, true_ab, true_ppoly = (
                    get_parameters(true_parameters, self.plain_tree))
            log10_true_ab = np.log10(true_ab)
            log10_true_ppoly = np.log10(true_ppoly)
            true_params = []
            for vn in self.varnames:
                true_params.append(true_bls[vn])
            for vn in self.varnames:
                true_params.append(np.log10(true_mrs[vn]))
            true_params.append(log10_true_ab)

            true_params.append(log10_true_ppoly)  # for polyprob

            self.true_params = np.array(true_params)

            self.true_loglike = -1.0*self.like_obj(self.true_params)

        

        #####################################################
        # setting initial parameters
        #####################################################
        if self.start_from_true:
            self.init_params = self.true_params
        else:
            self.init_params = igs.estimate_initial_parameters(self)
        self.unbound_init_params = unbound(self.init_params)


    def like_obj(self, varparams):
        return -self.loglike(varparams)

    def loglike(self, varparams):

        params = varparams[self.translate_indices]

        num_branches = self.num_branches
        branch_lengths = params[:num_branches]
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

        if self.ascertainment:
            log_asc_probs = asc.get_locus_asc_probs(branch_lengths,
                    mutation_rates, stat_dist, self, self.min_freq)
            log_asc_prob = (log_asc_probs *
                    self.asc_ages['count'].values).sum()
            ll -= log_asc_prob

            logmeanascprob, logpoissonlike = self.poisson_log_like(
                    log_asc_probs)
            ll += logpoissonlike * self.poisson_like_penalty
            print '@@ {:15}'.format(str(ll)), logmeanascprob, ' ',
        else:
            print '@@ {:15}'.format(str(ll)), ' ',
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

        if self.ascertainment:
            log_asc_probs = asc.get_locus_asc_probs(branch_lengths,
                    mutation_rates, stat_dist, self, self.min_freq)
            for i in xrange(lls.shape[0]):
                lls[i] -= log_asc_probs[int(self.data['family'].iloc[i])]

            logmeanascprob, logpoissonlike = self.poisson_log_like(
                    log_asc_probs)
            lls += logpoissonlike * self.poisson_like_penalty

        return lls

    def logit_bound_like_obj(self, ubx):
        bx = self.bound(ubx)
        return self.like_obj(bx)

    def inf_bound_like_obj(self, x):
        if np.any(x < self.lower):
            print 'inf:', x
            return np.inf
        if np.any(x > self.upper):
            print 'inf:', x
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
        mutation_rates = x[num_varnames:2*num_varnames]
        if np.any(x < self.lower):
            return -np.inf
        if np.any(x > self.upper):
            return -np.inf
        # for log-scale mutation parameterization
        logp = 1.0
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
        # make the branch lengths positive
        x[:num_varnames] = np.abs(x[:num_varnames])
        # make the mutation rates negative
        x[num_varnames:2*num_varnames] = (
                -1.0*np.abs(x[num_varnames:2*num_varnames]))
        # make the last two non-positive
        x[-2:] = -np.abs(x[-2:])
        pr = self.logprior(x)
        if not np.isfinite(pr):
            like = -np.inf
        else:
            like = -1.0*self.inf_bound_like_obj(x)
        v = pr + like
        if np.isnan(v):
            v = -np.inf

        print '@@', v, pr

        #print '@@', v, like, pr, logpoissonlike, logmeanascprob, ' ',
        #print '@@', v, like, pr, ' ',
        #_util.print_csv_line(x)

        return v

    def penalty_bound_log_posterior(self, x):
        good_x, penalty = self._get_bounds_penalty(x)
        # penalty is negative here, since the log_posterior needs to be
        # *maximized*
        return self.log_posterior(good_x) - penalty

    def logit_bound_posterior(self, ubx):
        '''
        ubx    varparams in unbound (probit-transformed) space
        '''
        x = self.bound(ubx)
        return self.log_posterior(x)


    def _get_bounds_penalty(self, x):
        '''
        takes in branch and mutation length parameters, bounds from
        transitions, and returns parameters that can be evaluated and a penalty

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


if __name__ == '__main__':
    inf = Inference(
            data_file = 'no_large_change_rounded.tsv',
            transitions_file = 'transition_matrices_mutation_gens3_symmetric.h5',
            tree_file = 'm1_study_bottleneck.newick',
            true_parameters = None,
            start_from_true = False,
            data_are_freqs = True,
            fst_filter_frac = None,
            genome_size = 20000,
            num_processes = 1,
            ascertainment = True,
            print_res = True,
            bottleneck_file = 'bottleneck_matrices.h5')

    inf.penalty_bound_like_obj(inf.lower-0.01)

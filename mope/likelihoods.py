from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
from builtins import range
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np
from scipy.stats import binom
from scipy.stats import beta as betarv
from scipy.misc import logsumexp
from . import transition_data_mut as tdm
import argparse
import pandas as pd
import sys
import scipy.optimize as opt
import scipy.linalg as scl
import scipy.stats as st
from numpy.linalg import matrix_power

from . import _likes
from ._binom import get_binom_likelihoods_cython

def discretize_beta_distn(a, b, N):
    '''
    discretize a beta distribution into bins such that bin i is delimited by 
    breaks[i] and breaks[i+1] (or breaks[len(breaks)-1] and 1).

    a, b    alpha and beta shape parameters of beta distribution
    breaks  locations of breaks as described above. must be sorted.

    breaks are [0,1/(2N)), [1/2N, 3/2N), [3/2N, 5/2N), ... [(2N-1)/2N, 2N/2N)
    '''
    assert 3/2 == 1.5
    lower_breaks = np.concatenate(([0],np.arange(1,2*N,2))) / (2*N)
    upper_breaks = np.concatenate((np.arange(1,2*N+1,2), [2*N])) / (2*N)
    p = st.beta.cdf(upper_breaks, a, b) - st.beta.cdf(lower_breaks, a,b)
    return p

def discretize_double_beta_distn(a, b, N, ppoly):
    assert 3/2 == 1.5
    distn = np.zeros(N+1)
    distn[[0,-1]] = (1.0-ppoly)/2
    newN = N - 2
    interior_distn = discretize_beta_distn(a, b, newN)
    distn[1:-1] = interior_distn * ppoly
    return distn

def get_stationary_distribution_beta(freqs, breaks, N, a, b):
    unbinned_distn = discretize_beta_distn(a, b, N)
    distn = np.add.reduceat(unbinned_distn, breaks)
    return distn

def get_stationary_distribution_double_beta(freqs, breaks, N, ab, prob_poly):
    unbinned_distn = discretize_double_beta_distn(ab, ab, N, prob_poly)
    distn = np.add.reduceat(unbinned_distn, breaks)
    return distn


def get_stationary_distribution_beta_old(freqs, breaks, N, a, b):
    '''
    given a set of breaks (calculated during matrix calculation), this gets the
    weighting of the center frequencies of the breaks to fit a beta
    distribution.

    each bin in the uniformly binned discrete distribution (corresponding to
    the distinct frequencies in the W-F model) will correspond to the
    frequencies [(2j-1)/(2N), (2j+1)/(2N)), each centered on j/N. The 0 and 1
    (frequency) categories will have zero probability.

    For distributions that aren't binned into one category for each W-F
    frequency, have to sum these terms.

    strategy for this latter case: first calculate weights for each W-F bin,
    then sum them appropriately.
    '''

    beta_breaks = np.arange(1, 2*N, 2) / (2*N)
    cdf_values = betarv.cdf(beta_breaks, a, b)
    p = np.diff(cdf_values)
    p = np.concatenate(([0], p, [0]))
    p /= p.sum()
    try:
        if breaks == 0:
            distn = p
    except ValueError:
        if breaks[0] != 0 or breaks[-1] != N:
            raise ValueError("breaks must start and end with 0 and N, resp.")
        distn = np.add.reduceat(p, breaks)
    #distn[[0,-1]] = 0
    distn /= distn.sum()
    return distn

def get_binom_likelihoods(Xs, n, freqs):
    '''
    binomial sampling likelihoods for the leaves.

    Xs     counts of the focal allele at each locus, length k, where k is
           number of loci
    n      samples size at every locus, of length k
    freqs  true allele frequencies to consider, length l

    returns a k x l array P of binomial sampling probabilities, where P[i,j]
    is the sampling probability at locus i given true frequency j
    '''
    Xs = np.array(Xs)
    ns = np.ones(Xs.shape[0]) * n
    freqs = np.array(freqs)
    params = np.vstack((Xs,ns))
    assert params.shape[0] > 0
    likes = np.transpose(np.apply_along_axis(
        lambda x: binom.pmf(x[0], x[1], freqs), 0, params))
    return likes

def get_locus_log_like_log_space(P_branches, P_leaves, locus_leaf_likes,
        stationary_distribution):
    log_conditional_probs_fob = (
            np.log(np.dot(P_leaves[0], locus_leaf_likes[0])) +
            np.log(np.dot(P_leaves[1], locus_leaf_likes[1])))
    log_conditional_probs_bra = (
            np.log(np.dot(P_branches[0], np.exp(log_conditional_probs_fob))) +
            np.log(np.dot(P_leaves[2], locus_leaf_likes[2])))

    log_conditional_probs_uro = (
            np.log(np.dot(P_leaves[3], locus_leaf_likes[3])) +
            np.log(np.dot(P_leaves[4], locus_leaf_likes[4])))

    log_conditional_probs_dig = (
            np.log(np.dot(P_leaves[8], locus_leaf_likes[8])) +
            np.log(np.dot(P_leaves[9], locus_leaf_likes[9])))

    log_conditional_probs_mus = (
            np.log(np.dot(P_leaves[7], locus_leaf_likes[7])) +
            np.log(np.dot(P_branches[3], np.exp(log_conditional_probs_dig))))
    log_conditional_probs_lat = (
            np.log(np.dot(P_leaves[6], locus_leaf_likes[6])) +
            np.log(np.dot(P_branches[4], np.exp(log_conditional_probs_mus))))

    log_conditional_probs_mes = (
            np.log(np.dot(P_branches[2], np.exp(log_conditional_probs_uro))) +
            np.log(np.dot(P_leaves[5], locus_leaf_likes[5])) +
            np.log(np.dot(P_branches[5], np.exp(log_conditional_probs_lat))))

    log_conditional_probs_end = np.log(np.dot(
        P_leaves[10], locus_leaf_likes[10]))

    # root, here
    log_conditional_probs_emb = (
            np.log(np.dot(P_branches[1], np.exp(log_conditional_probs_bra))) +
            np.log(np.dot(P_branches[6], np.exp(log_conditional_probs_mes))) +
            np.log(np.dot(P_branches[7], np.exp(log_conditional_probs_end))))

    log_stat = np.log(stationary_distribution)
    log_root_probs = log_stat + log_conditional_probs_emb
    locusloglike = logsumexp(log_root_probs)

    return locusloglike

def get_locus_log_like(P_branches, P_leaves, locus_leaf_likes,
        stationary_distribution):
    conditional_probs_fob = (
            np.dot(P_leaves[0], locus_leaf_likes[0]) *
            np.dot(P_leaves[1], locus_leaf_likes[1]))
    conditional_probs_bra = (
            np.dot(P_branches[0], conditional_probs_fob) *
            np.dot(P_leaves[2], locus_leaf_likes[2]))

    conditional_probs_uro = (
            np.dot(P_leaves[3], locus_leaf_likes[3]) *
            np.dot(P_leaves[4], locus_leaf_likes[4]))

    conditional_probs_dig = (
            np.dot(P_leaves[8], locus_leaf_likes[8]) *
            np.dot(P_leaves[9], locus_leaf_likes[9]))
    conditional_probs_mus = (
            np.dot(P_leaves[7], locus_leaf_likes[7]) *
            np.dot(P_branches[3], conditional_probs_dig))
    conditional_probs_lat = (
            np.dot(P_leaves[6], locus_leaf_likes[6]) *
            np.dot(P_branches[4], conditional_probs_mus))

    conditional_probs_mes = (
            np.dot(P_branches[2], conditional_probs_uro) *
            np.dot(P_leaves[5], locus_leaf_likes[5]) *
            np.dot(P_branches[5], conditional_probs_lat))

    conditional_probs_end = np.dot(P_leaves[10], locus_leaf_likes[10])

    # root, here
    conditional_probs_emb = (
            np.dot(P_branches[1], conditional_probs_bra) *
            np.dot(P_branches[6], conditional_probs_mes) *
            np.dot(P_branches[7], conditional_probs_end))

    locusloglike = np.log(np.dot(stationary_distribution,
        conditional_probs_emb))

    if locusloglike == -np.inf:
        return get_locus_log_like_log_space( P_branches, P_leaves,
                locus_leaf_likes, P_branches01, P_branches8leaves10,
                stationary_distribution)
    '''
    print 'locus root probs all zero:', np.all(conditional_probs_emb == 0.0)
    print 'conditional_probs_mes all zero:', np.all(conditional_probs_mes == 0.0)
    print 'conditional_probs_bra all zero:', np.all(conditional_probs_bra == 0.0)
    print
    print np.dot(P_branches01, conditional_probs_bra).sum()
    print np.dot(P_branches[3], conditional_probs_mes).sum()
    print np.dot(P_branches8leaves10, locus_leaf_likes[10]).sum()
    print
    '''

    return locusloglike

def get_log_likelihood_somatic(
        branch_lengths,
        branch_mutation_rates,
        leaf_rates,
        leaf_mutation_rates,
        data,
        transitions,
        bottleneck,
        ascertainment,
        leaf_likelihoods,
        freqs,
        stationary_distribution):
    '''
    branch_lengths            numpy array, shape (8,), containing the lengths of
                              the interior branches
    branch_mutation_rates     numpy array, shape (8,), containing the scaled
                              mutation rates for the interior branch lengths
    leaf_rates                numpy array, shape (11,), containing the per-year
                              rates of drift accumulation
    leaf_mutation_rates       numpy array, shape (11,), containing the scaled
                              mutation rates corresponding to leaf_rates 
    data                      pandas data frame containing the frequency,
                              coverage, and age data. here just used for the
                              ages
    transitions               two-dimensional (time, mutation) TransitionData
                              object containing the pre-computed transition
                              probabilities
    ascertainment             conditioning on polymorphism (heteroplasmy in at
                              least one tissue per site? {True, False} 
    leaf_likelihoods          numpy array, shape
                              (data.shape[0], 11, freqs.shape[0]) containing
                              the (non-log) leaf likelihoods for the freqs for
                              each site and each leaf. index order is
                              (site, leaf, freq)
    freqs                     numpy array containing the freqs at which
                              transition probabilities were pre-calculated
    stationary_distribution   numpy array containing the stationary
                              distribution, at the root, on freqs
    '''
        
    # leaf_lengths is a (data.shape[0] x leaf_rates.shape[0]) array, with
    # each row representing a site, containing the 11 leaf_lengths for that
    # site in its columns
    leaf_lengths = np.outer(data['age'].values, leaf_rates)

    def get_trans_probs(t, m):
        return transitions.get_transition_probabilities_time_mutation(t, m)

    def get_multiple_trans_probs(times, muts):
        assert times.shape == muts.shape
        firstP = get_trans_probs(times[0], muts[0])
        Ps = []
        Ps.append(firstP)
        for i, (time, mut) in enumerate(zip(times[1:], muts[1:])):
            Ps.append(get_trans_probs(time, mut))
        return Ps

    # get transition probabilities for each of the internal branches, which are
    # the same for each locus
    P_branches = get_multiple_trans_probs(branch_lengths,
            branch_mutation_rates)

    # setting up
    prev_age = data['age'][0]
    P_leaves = get_multiple_trans_probs(leaf_lengths[0],
        leaf_mutation_rates)
    ages = data['age'].values

    loglike = 0.0
    for locus_idx in range(data.shape[0]):
        cur_age = ages[locus_idx]
        if cur_age != prev_age:
            # this takes about 60% of execution time
            P_leaves = get_multiple_trans_probs(leaf_lengths[locus_idx],
                leaf_mutation_rates)
            prev_age = cur_age
        locus_leaf_likes = leaf_likelihoods[locus_idx]

        locusloglike = get_locus_log_like(P_branches, P_leaves,
                locus_leaf_likes, stationary_distribution)

        # conditioning on ascertainment
        ascertainment_probs = get_ascertainment_probs_somatic(P_branches,
                P_leaves, stationary_distribution)
        locusloglike -= np.log(ascertainment_probs[locus_idx])

        loglike += locusloglike

    return loglike

def get_log_likelihood_somatic_newick(
        branch_lengths,
        mutation_rates,
        stationary_distribution,
        inf):
    '''
    Calculate the log-likelihood of a given tree, with branch lengths, leaf
    rates, and mutation rates.

    mrca                       newick.Node representing the MRCA.
    branch_lengths             numpy array of branch lengths
    branch_mutation_rates      numpy array of branch mutation rates
    leaf_rates                 numpy array of drift accumulation rates at the
                               leaves
    leaf_mutation_rates        numpy array of leaf mutation rates
    branch_indices             dict, giving index (in above arrays) of each
                               branch label
    leaf_indices               dict, giving index (in above arrays) of each
                               leaf label
    ages                       numpy array of ages for each locus
    transitions                two-dimensional transition data
    ascertainment_mrca         newick.Node representing the tree, if
                               ascertainment is True
    leaf_likelihoods           dict of allele frequency likelihoods at each
                               leaf. Each value is a two-dimensional numpy
                               array A of likelihoods, where A[i,j] gives the
                               likelihood of the jth frequency at the ith locus
    stationary_distribution    numpy array giving stationary distribution of
                               allele frequencies at the root
    '''
    mrca = inf.tree
    leaf_likelihoods = inf.leaf_likes_dict
    leaf_indices = inf.leaf_indices
    branch_indices = inf.branch_indices
    multiplier_dict = inf.multiplierdict
    data = inf.data
    num_loci = inf.num_loci
    transitions = inf.transition_data
    bottlenecks = inf.bottleneck_data

    for node in mrca.walk(mode = 'postorder'):
        _likes.reset_likes_ones(node.cond_probs)

    for node in mrca.walk(mode = 'postorder'):
        if node == mrca:
            break
        name = node.name
        branch_index = branch_indices[name]
        multipliername = node.multipliername

        ancestor = node.ancestor
        mut_rate = mutation_rates[branch_index]
        ancestor_likes = ancestor.cond_probs

        if node.is_leaf:
            node_likes = leaf_likelihoods[name].copy()
        else:
            node_likes = node.cond_probs

        if multipliername is not None:
            if node.is_mut:
                # assume that the associated multipliername is an age
                ages = data[multipliername].values
                # minimum age at which mutation begins to have an effect
                mut_time = branch_lengths[branch_index]
                _likes.compute_mutation_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        ages,
                        mut_time,
                        mut_rate,
                        transitions)
            else:
                node_lengths = (data[multipliername].values *
                        branch_lengths[branch_index])
                _likes.compute_leaf_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        node_lengths,
                        mut_rate,
                        transitions)

        else:  # multipliername is None
            if node.is_bottleneck:
                if bottlenecks is None:
                    raise ValueError('pre-computed bottleneck data needed')
                bottleneck_size = branch_lengths[branch_index]
                _likes.compute_bottleneck_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        bottleneck_size,
                        mut_rate,
                        bottlenecks)

            else:  # not a bottleneck, not a just-mut branch
                node_length = branch_lengths[branch_index]
                _likes.compute_branch_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        node_length,
                        mut_rate,
                        transitions)

    loglike = _likes.compute_root_log_like(
            mrca.cond_probs,
            stationary_distribution)

    #if loglike == -np.inf:
    #    print >> sys.stderr, 'bad params:'
    #    for bl in branch_lengths:
    #        print >> sys.stderr, bl
    #    for mr in mutation_rates:
    #        print >> sys.stderr, mr
    #    raise ValueError('bad params')

    return loglike


# adding function for debugging individual likelihoods
def get_locus_log_likelihoods_newick(
        branch_lengths,
        mutation_rates,
        stationary_distribution,
        inf,
        use_counts = False):
    '''
    Calculate the log-likelihood of a given tree, with branch lengths, leaf
    rates, and mutation rates.

    mrca                       newick.Node representing the MRCA.
    branch_lengths             numpy array of branch lengths
    branch_mutation_rates      numpy array of branch mutation rates
    leaf_rates                 numpy array of drift accumulation rates at the
                               leaves
    leaf_mutation_rates        numpy array of leaf mutation rates
    branch_indices             dict, giving index (in above arrays) of each
                               branch label
    leaf_indices               dict, giving index (in above arrays) of each
                               leaf label
    ages                       numpy array of ages for each locus
    transitions                two-dimensional transition data
    ascertainment_mrca         newick.Node representing the tree, if
                               ascertainment is True
    leaf_likelihoods           dict of allele frequency likelihoods at each
                               leaf. Each value is a two-dimensional numpy
                               array A of likelihoods, where A[i,j] gives the
                               likelihood of the jth frequency at the ith locus
    stationary_distribution    numpy array giving stationary distribution of
                               allele frequencies at the root
    '''
    mrca = inf.tree
    leaf_likelihoods = inf.leaf_likes_dict
    leaf_indices = inf.leaf_indices
    branch_indices = inf.branch_indices
    multiplier_dict = inf.multiplierdict
    data = inf.data
    num_loci = inf.num_loci
    transitions = inf.transition_data
    bottlenecks = inf.bottleneck_data

    for node in mrca.walk(mode = 'postorder'):
        _likes.reset_likes_ones(node.cond_probs)

    for node in mrca.walk(mode = 'postorder'):
        if node == mrca:
            break
        name = node.name.encode('ascii')
        branch_index = branch_indices[name]
        multipliername = node.multipliername

        ancestor = node.ancestor
        mut_rate = mutation_rates[branch_index]
        ancestor_likes = ancestor.cond_probs

        if node.is_leaf:
            node_likes = leaf_likelihoods[name].copy()
        else:
            node_likes = node.cond_probs

        if multipliername is not None:
            if node.is_mut:
                # assume that the associated multipliername is an age
                ages = data[multipliername].values
                # minimum age at which mutation begins to have an effect
                mut_time = branch_lengths[branch_index]
                _likes.compute_mutation_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        ages,
                        mut_time,
                        mut_rate,
                        transitions)
            else:
                node_lengths = (data[multipliername].values *
                        branch_lengths[branch_index])
                _likes.compute_leaf_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        node_lengths,
                        mut_rate,
                        transitions)

        else:  # multipliername is None
            if node.is_bottleneck:
                if bottlenecks is None:
                    raise ValueError('pre-computed bottleneck data needed')
                bottleneck_size = branch_lengths[branch_index]
                _likes.compute_bottleneck_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        bottleneck_size,
                        mut_rate,
                        bottlenecks)

            else:  # not a bottleneck
                node_length = branch_lengths[branch_index]
                _likes.compute_branch_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        node_length,
                        mut_rate,
                        transitions)

    loglikes = _likes.compute_root_locus_log_likes(
            mrca.cond_probs,
            stationary_distribution,
            use_counts)

    #if loglike == -np.inf:
    #    print >> sys.stderr, 'bad params:'
    #    for bl in branch_lengths:
    #        print >> sys.stderr, bl
    #    for mr in mutation_rates:
    #        print >> sys.stderr, mr
    #    raise ValueError('bad params')

    return loglikes


if __name__ == '__main__':
    from . import transition_data_mut as tdm
    from . import params as par
    import sys
    import time
    import numpy.random as npr
    from . import data as da
    from . import inference as inf
    from .simulate import get_parameters

    
    transition_file = 'transition_matrices_mutation_gens3_symmetric.h5'
    inference = inf.Inference('test_data_short.txt', transition_file,
            'somatic.newick', "lbfgs", 'somatic_parameters.txt', False, False,
            None, 16500, 1, True, False)
    inf_log_like = -inference.like_obj(inference.true_params)
    print('inf log like:', inf_log_like)

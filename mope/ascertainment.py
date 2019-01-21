from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os 
import numpy as np

from . import likelihoods as lis
from . import _likes

def get_ascertainment_prob_for_binom(x, inf):
    '''
    x     branchnames space, not multiplied by avg_multiplier
    '''
    nb = inf.num_branches
    branch_lengths = x[:nb]
    mutation_rates = x[nb:2*nb]
    ab = 10**x[2*nb]
    stat_dist = lis.get_stationary_distribution_beta(inf.freqs, inf.breaks,
            inf.transition_N, ab, ab)
    return get_ascertainment_prob_somatic_newick_min_freq(
            branch_lengths, mutation_rates, stat_dist, inf, inf.min_freq)


def get_family_log_ascertainment_probs(x, inf):
    '''
    gets the ascertainment probabilities for each family


    x     branchnames space, not multiplied by avg_multiplier
    inf   Inference object
    '''
    nb = inf.num_branches
    branch_lengths = x[:nb]
    mutation_rates = x[nb:2*nb]
    ab = 10**x[2*nb]
    stat_dist = lis.get_stationary_distribution_beta(inf.freqs, inf.breaks,
            inf.transition_N, ab, ab)
    return get_locus_asc_probs(
            branch_lengths, mutation_rates, stat_dist, inf, inf.min_freq)


def get_ascertainment_prob_somatic_newick_min_freq(
        branch_lengths,
        mutation_rates,
        stationary_distribution,
        inf,
        min_freq):
    '''
    Calculate the sum of the logs of probability of ascertainment

    branch_lengths             numpy array of branch lengths, branchparams
    mutation_rates             numpy array of mutation rates, branch_params
    stationary_distribution    numpy array giving stationary distribution of
                               allele frequencies at the root
    inf                        Inference object
    min_freq                   minimum frequency considered to be
                               heteroplasmic, assumed to be symmetric
    '''
    nfreqs = stationary_distribution.shape[0]
    transitions = inf.transition_data
    bottlenecks = inf.bottleneck_data
    asc_mrca = inf.asc_tree
    asc_ages = inf.asc_ages
    branch_indices = inf.branch_indices
    counts = inf.asc_ages['count'].values

    freqs = inf.freqs

    e0 = np.zeros(nfreqs)
    low_filt = freqs < min_freq
    e0[low_filt] = 1.0
    eN = np.zeros(nfreqs)
    high_filt = freqs > 1-min_freq
    eN[high_filt] = 1.0
    eboth = np.vstack((e0,eN) * inf.num_asc_combs)

    for node in asc_mrca.walk(mode = 'postorder'):
        _likes.reset_likes_ones(node.cond_probs)
    
    for node in asc_mrca.walk(mode = 'postorder'):
        if node == asc_mrca:
            break
        branch_index = branch_indices[node.name]
        multipliername = node.multipliername

        if node.is_leaf:
            node_likes = eboth.copy()
        else:
            node_likes = node.cond_probs

        mut_rate = mutation_rates[branch_index]
        ancestor = node.ancestor
        ancestor_likes = ancestor.cond_probs

        if multipliername is not None:
            node_lengths = np.repeat((asc_ages[multipliername].values *
                    branch_lengths[branch_index]), 2)
            assert node_lengths.shape[0] == 2*inf.num_asc_combs
            _likes.compute_rate_transition_likelihood(
                    node_likes,
                    ancestor_likes,
                    node_lengths,
                    mut_rate,
                    transitions)
        else:
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
                _likes.compute_length_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        node_length,
                        mut_rate,
                        transitions)

    log_asc_prob = _likes.compute_log_asc_prob_2(asc_mrca.cond_probs,
            stationary_distribution, counts)
    return log_asc_prob

def get_locus_asc_probs(
        branch_lengths,
        mutation_rates,
        stationary_distribution,
        inf,
        min_freq,
        tree_idx):
    '''
    Calculate the per-locus logs of probability of ascertainment

    asc_mrca                   newick.Node representing the MRCA, initialized
                               with *two*num_loci* loci (for fix, loss,
                               for each locus)
    branch_lengths             numpy array of branch lengths
    branch_mutation_rates      numpy array of branch mutation rates
    leaf_rates                 numpy array of drift accumulation rates at the
                               leaves
    leaf_mutation_rates        numpy array of leaf mutation rates
    branch_indices             dict, giving index (in above arrays) of each
                               branch label
    leaf_indices               dict, giving index (in above arrays) of each
                               leaf label
    multipliers                numpy array of unique multiplier (age)
                               combinations for which to calculate asc prob
                               shape (nuniq, nmult), where nuniq is the number
                               of unique multiplier combinations (this will be
                               roughly equal to the number of families), and
                               nmult is the number of distinct multiplier
                               variables
    counts                     counts of each unique multiplier set,
                               shape (nuniq,)
    transitions                two-dimensional transition data
    stationary_distribution    numpy array giving stationary distribution of
                               allele frequencies at the root
    '''
    nfreqs = stationary_distribution.shape[0]
    transitions = inf.transition_data
    bottlenecks = inf.bottleneck_data
    asc_mrca = inf.asc_trees[tree_idx]
    asc_ages = inf.asc_ages[tree_idx]
    branch_indices = inf.branch_indices[tree_idx]
    counts = asc_ages['count'].values
    num_asc_combs = inf.num_asc_combs[tree_idx]

    freqs = inf.freqs

    e0 = np.zeros(nfreqs)
    low_filt = freqs < min_freq
    e0[low_filt] = 1.0
    eN = np.zeros(nfreqs)
    high_filt = freqs > 1-min_freq
    eN[high_filt] = 1.0
    eboth = np.vstack((e0,eN) * num_asc_combs)

    for node in asc_mrca.walk(mode = 'postorder'):
        _likes.reset_likes_ones(node.cond_probs)
    
    for node in asc_mrca.walk(mode = 'postorder'):
        if node == asc_mrca:
            break
        branch_index = branch_indices[node.name]
        multipliername = node.multipliername

        if node.is_leaf:
            node_likes = eboth.copy()
        else:
            node_likes = node.cond_probs

        mut_rate = mutation_rates[branch_index]
        ancestor = node.ancestor
        ancestor_likes = ancestor.cond_probs

        if multipliername is not None:
            node_lengths = np.repeat((asc_ages[multipliername].values *
                    branch_lengths[branch_index]), 2)
            assert node_lengths.shape[0] == 2*num_asc_combs
            _likes.compute_rate_transition_likelihood(
                    node_likes,
                    ancestor_likes,
                    node_lengths,
                    mut_rate,
                    transitions)
        else:
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
                _likes.compute_length_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        node_length,
                        mut_rate,
                        transitions)

    logprobs = _likes.get_individual_locus_asc_logprobs(asc_mrca.cond_probs,
            stationary_distribution, counts)
    return logprobs


def get_family_asc_probs(
        branch_lengths,
        mutation_rates,
        stationary_distribution,
        inf):
    '''
    Calculate the per-locus logs of probability of ascertainment

    branch_lengths             numpy array of branch lengths
    mutation_rates             numpy array of mutation rates
    stationary_distribution    numpy array giving stationary distribution of
                               allele frequencies at the root
    inf                        Inference object
    '''
    nfreqs = stationary_distribution.shape[0]
    transitions = inf.transition_data
    asc_mrca = inf.asc_tree
    asc_ages = inf.asc_ages
    branch_indices = inf.branch_indices
    counts = inf.asc_counts

    e0 = np.zeros(nfreqs)
    e0[0] = 1.0
    eN = np.zeros(nfreqs)
    eN[nfreqs-1] = 1.0
    eboth = np.vstack((e0,eN) * inf.num_asc_combs)

    for node in asc_mrca.walk(mode = 'postorder'):
        _likes.reset_likes_ones(node.cond_probs)
    
    for node in asc_mrca.walk(mode = 'postorder'):
        if node == asc_mrca:
            break
        branch_index = branch_indices[node.name]
        multipliername = node.multipliername

        if node.is_leaf:
            node_likes = eboth.copy()
        else:
            node_likes = node.cond_probs

        mut_rate = mutation_rates[branch_index]
        ancestor = node.ancestor
        ancestor_likes = ancestor.cond_probs

        if multipliername is not None:
            node_lengths = np.repeat((asc_ages[multipliername] *
                    branch_lengths[branch_index]), 2)
            assert node_lengths.shape[0] == 2*inf.num_asc_combs
            _likes.compute_rate_transition_likelihood(
                    node_likes,
                    ancestor_likes,
                    node_lengths,
                    mut_rate,
                    transitions)
        else:
            node_length = branch_lengths[branch_index]
            _likes.compute_length_transition_likelihood(
                    node_likes,
                    ancestor_likes,
                    node_length,
                    mut_rate,
                    transitions)

    logprobs = _likes.get_individual_locus_asc_logprobs(asc_mrca.cond_probs,
            stationary_distribution, counts)
    return logprobs



if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from . import newick
    from . import transition_data_mut as tdm
    from . import params as par
    import sys
    import time
    import numpy.random as npr
    from . import data as da

    from functools import partial

    from . import _binom
    from .simulate import get_parameters
    from . import likelihoods as lis
    from . import inference

    import argparse

    parser = argparse.ArgumentParser(
            description='calculate ascertainment probabilities',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('tree', help = "", type = str)
    parser.add_argument('params')
    parser.add_argument('data', help = "", type = str)
    parser.add_argument('transitions')
    args = parser.parse_args()

    transitions = args.transitions
    inf = inference.Inference(
            data_file = args.data,
            transitions_file = transitions,
            tree_file = args.tree,
            true_parameters = args.params,
            start_from_true = True,
            data_are_freqs = True,
            genome_size = 10000,
            num_processes = 1)

    avg_multipliers = []
    for bn in inf.branch_names:
        mult = inf.multiplierdict[bn]
        if mult is not None:
            avg_multipliers.append(inf.data[mult].mean())
        else:
            assert mult is None
            avg_multipliers.append(1.0)
    avg_multipliers = np.array(avg_multipliers)
    avg_multipliers = np.concatenate((avg_multipliers, np.repeat(1.,
        avg_multipliers.shape[0]), (1.0,)))

    tp = inf.true_params[inf.translate_indices]
    #tp *= avg_multipliers
    tp, _ = inference.get_bounds_penalty_som(tp, inf)
    branch_lengths = tp[:inf.num_branches]
    mutation_rates = tp[inf.num_branches:2*inf.num_branches]
    ab = 10**tp[2*inf.num_branches]
    print('branch lengths:', branch_lengths)
    print('mutation rates:', mutation_rates)
    print('ab:', ab)

    stat_dist = lis.get_stationary_distribution_beta(inf.freqs, inf.breaks,
            inf.transition_N, ab, ab)

    log_asc_prob = get_locus_asc_probs(
            branch_lengths,
            mutation_rates,
            stat_dist,
            inf)
    print(np.exp(log_asc_prob))

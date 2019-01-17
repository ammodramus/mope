from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
import os
import os.path
import emcee
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from collections import defaultdict


from . import likelihoods as lis
from . import transition_data_mut as tdm
from . import params as par
from functools import partial
import multiprocessing as mp
import numpy.random as npr

from . import newick
from . import inference as inf
from . import util as ut
from . import data as da
from . import _binom
from . import ascertainment as asc
from . import _likes



def _get_ancestors(focalnode):
    ancs = []
    curnode = focalnode
    while True:
        curnode = curnode.ancestor
        ancs.append(curnode.name)
        if not curnode.ancestor:
            break
    return ancs


def fill_cond_probs(
        inf,
        tree_idx,
        varparams,
        include_mutation_only = False):

    trans_idxs = inf.translate_indices[tree_idx]
    params = varparams[trans_idxs]
    num_branches = inf.num_branches[tree_idx]
    branch_lengths = 10**params[:num_branches]
    mutation_rates = 10**params[num_branches:]
    alphabeta, polyprob = 10**varparams[-2:]
    stat_dist = lis.get_stationary_distribution_double_beta(inf.freqs,
            inf.breaks, inf.transition_N, alphabeta, polyprob)

    mrca = inf.trees[tree_idx]
    # leaf_likelihoods is a dictionary, keys with leaf names, values ndarrays
    # with likelihoods, shape is (nloci, nfreqs)
    leaf_likelihoods = inf.leaf_likes[tree_idx]
    leaf_indices = inf.leaf_indices[tree_idx]
    branch_indices = inf.branch_indices[tree_idx]
    multiplier_dict = inf.multiplierdict[tree_idx]
    data = inf.data[tree_idx]
    num_loci = inf.num_loci[tree_idx]
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
    root_cond_probs = inf.trees[tree_idx].cond_probs
    overall_loglikes = np.log(np.dot(root_cond_probs, stat_dist))
    if include_mutation_only:
        stat_dist[1:-1] = 0
        mut_only_loglikes =  np.log(np.dot(root_cond_probs, stat_dist))
        return overall_loglikes, mut_only_loglikes
    else:
        return overall_loglikes


def fill_cond_probs_mutation(
        focalnode,
        inf,
        tree_idx,
        varparams):

    trans_idxs = inf.translate_indices[tree_idx]
    params = varparams[trans_idxs]
    num_branches = inf.num_branches[tree_idx]
    branch_lengths = 10**params[:num_branches]
    mutation_rates = 10**params[num_branches:]
    alphabeta, polyprob = 10**varparams[-2:]
    stat_dist = lis.get_stationary_distribution_double_beta(inf.freqs,
            inf.breaks, inf.transition_N, alphabeta, polyprob)

    mrca = inf.trees[tree_idx]
    # leaf_likelihoods is a dictionary, keys with leaf names, values ndarrays
    # with likelihoods, shape is (nloci, nfreqs)
    leaf_likelihoods = inf.leaf_likes[tree_idx]
    leaf_indices = inf.leaf_indices[tree_idx]
    branch_indices = inf.branch_indices[tree_idx]
    multiplier_dict = inf.multiplierdict[tree_idx]
    data = inf.data[tree_idx]
    num_loci = inf.num_loci[tree_idx]
    transitions = inf.transition_data
    bottlenecks = inf.bottleneck_data

    for node in mrca.walk(mode = 'postorder'):
        _likes.reset_likes_ones(node.cond_probs)

    # determine which nodes are descendents of the focal node, and which aren't
    descendants = set([node.name for node in focalnode.walk() if node != focalnode])

    for node in mrca.walk(mode = 'postorder'):
        if node == mrca:
            break
        name = node.name
        is_descendant = name in descendants
        is_focal = name == focalnode.name
        branch_index = branch_indices[name]
        multipliername = node.multipliername

        ancestor = node.ancestor
        mut_rate = mutation_rates[branch_index]
        ancestor_likes = ancestor.cond_probs

        if node.is_leaf:
            node_likes = leaf_likelihoods[name].copy()  # shape: (nloci, nfreqs)
            if is_descendant:
                node_likes[:,[0,-1]] = 0
            else:
                node_likes[:,1:-1] = 0  # linear scale... but this shouldn't matter
        else:
            node_likes = node.cond_probs

        if multipliername is not None:
            node_lengths = (data[multipliername].values *
                    branch_lengths[branch_index])
            if is_descendant:
                _likes.compute_leaf_zero_transition_likelihood_descendant(
                        node_likes,
                        ancestor_likes,
                        node_lengths,
                        mut_rate,
                        transitions)
            elif node != focalnode:  # not a descendant, not the focal node
                _likes.compute_leaf_zero_transition_likelihood(
                        node_likes,
                        ancestor_likes,
                        node_lengths,
                        mut_rate,
                        transitions)
            else:
                assert is_focal
                _likes.compute_leaf_focal_node_zero_transition_likelihood(
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
                if is_descendant:
                    _likes.compute_bottleneck_transition_likelihood_zero_descendant(
                            node_likes,
                            ancestor_likes,
                            bottleneck_size,
                            mut_rate,
                            bottlenecks)
                elif node.name != focalnode.name:
                    _likes.compute_bottleneck_transition_likelihood_zero(
                            node_likes,
                            ancestor_likes,
                            bottleneck_size,
                            mut_rate,
                            bottlenecks)
                else:
                    assert node.name == focalnode.name
                    _likes.compute_bottleneck_transition_likelihood_zero_focalnode(
                            node_likes,
                            ancestor_likes,
                            bottleneck_size,
                            mut_rate,
                            bottlenecks)

            else:  # not a bottleneck
                node_length = branch_lengths[branch_index]
                if is_descendant:
                    _likes.compute_branch_transition_likelihood_zero_descendant(
                            node_likes,
                            ancestor_likes,
                            node_length,
                            mut_rate,
                            transitions)
                elif node.name != focalnode.name:  # non-descendant
                    _likes.compute_branch_transition_likelihood_zero(
                            node_likes,
                            ancestor_likes,
                            node_length,
                            mut_rate,
                            transitions)
                else:
                    assert node == focalnode
                    _likes.compute_branch_transition_likelihood_zero_focalnode(
                            node_likes,
                            ancestor_likes,
                            node_length,
                            mut_rate,
                            transitions)


    root_cond_probs = inf.trees[tree_idx].cond_probs
    overall_loglikes = np.log(root_cond_probs[:,0]*stat_dist[0])
    return overall_loglikes

def run_posterior_mut_loc(args):
    global inf_data
    '''
    =========================
    Initialization / setup
    =========================
    '''

    lower_dr, upper_dr = args.drift_limits

    data_files, tree_files, age_files = ut.get_input_files(args)

    inf_data = inf.Inference(
            data_files = data_files,
            tree_files = tree_files,
            age_files = age_files,
            transitions_file = args.drift,
            true_parameters = None,
            start_from = 'prior',
            data_are_freqs = args.data_are_frequencies,
            genome_size = args.genome_size,
            bottleneck_file = args.bottlenecks,
            min_freq = args.min_het_freq,
            poisson_like_penalty = args.asc_prob_penalty,
            print_debug = False,
            log_unif_drift = True,
            inverse_bot_priors = False,
            post_is_prior = False,
            lower_drift_limit = lower_dr,
            upper_drift_limit = upper_dr,
            min_phred_score = args.min_phred_score)

    # load in posterior data, too
    posterior_data = pd.read_csv(args.posteriorsamples, header = 0, comment = '#', sep = '\t').iloc[:,1:]
    posterior_data = posterior_data.iloc[int(args.burnin_frac*posterior_data.shape[0]+0.5):,:]


    # TODO (?) rewrite this to remove redundancy
    if not args.overall_mutation:
        has_header = False
        log_probs = defaultdict(lambda: defaultdict(list))
        log_probs_all_nodes = {}
        rep_idx = 0
        for tree_idx in range(len(inf_data.trees)):
            tree_leaf_likes = inf_data.leaf_likes[tree_idx]
            for idx, sampled_varpars in posterior_data.sample(args.numposteriorsamples, axis = 0).iterrows():
                sampled_varpars = sampled_varpars.values
                log_overall_probs = fill_cond_probs(inf_data, tree_idx, sampled_varpars)
                for node in inf_data.trees[tree_idx].walk('postorder'):
                    if node == inf_data.trees[tree_idx]:   # the MRCA
                        continue
                    log_mut_probs = fill_cond_probs_mutation(node, inf_data, tree_idx, sampled_varpars)
                    log_p_this_node = log_mut_probs - log_overall_probs
                    datp = pd.DataFrame(log_p_this_node)
                    datp.columns = ['logprob']
                    datp['family'] = inf_data.data[tree_idx]['family']
                    datp['position'] = inf_data.data[tree_idx]['position'].astype(np.int32)
                    datp['nodename'] = node.name
                    datp['rep'] = rep_idx
                    if not has_header:
                        datp.to_csv(sys.stdout, index = False, sep = str('\t'))
                        has_header = True
                    else:
                        datp.to_csv(sys.stdout, index = False, header = False, sep = str('\t'))
                    rep_idx += 1



    else:
        log_probs = defaultdict(list)
        log_probs_all_nodes = {}
        for tree_idx in range(len(inf_data.trees)):
            tree_leaf_likes = inf_data.leaf_likes[tree_idx]
            for idx, sampled_varpars in posterior_data.sample(args.numposteriorsamples, axis = 0).iterrows():
                sampled_varpars = sampled_varpars.values
                log_overall_probs, log_mut_only_probs = fill_cond_probs(inf_data, tree_idx, sampled_varpars, True)
                log_p_this_node = log_mut_only_probs - log_overall_probs
                log_probs[tree_idx].append(log_p_this_node)

        for tree_idx in range(len(inf_data.trees)):
            log_cond_probs = np.array(log_probs[tree_idx]).T
            log_mut_probs = pd.DataFrame(log_cond_probs)
            log_mut_probs['family'] = inf_data.data[tree_idx]['family']
            log_mut_probs['position'] = inf_data.data[tree_idx]['position'].astype(np.int32)
            log_probs[tree_idx] = pd.melt(log_mut_probs, id_vars = ['family', 'position'], var_name = 'rep', value_name = 'logprob')
        all_log_probs = pd.concat(log_probs.values(), ignore_index = True)
        all_log_probs.to_csv(sys.stdout, index = False, sep = str('\t'))

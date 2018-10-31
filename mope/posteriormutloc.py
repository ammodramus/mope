from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
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


def fill_cond_probs(
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
    loglikes = np.log(np.dot(root_cond_probs, stat_dist))
    return loglikes


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

    log_probs = defaultdict(lambda: defaultdict(list))
    log_probs_all_nodes = {}

    for tree_idx in range(len(inf_data.trees)):
        tree_leaf_likes = inf_data.leaf_likes[tree_idx]
        inf_data.data[tree_idx]['position'] = inf_data.data[tree_idx]['position'].astype(np.int32)
        for idx, sampled_varpars in posterior_data.sample(args.numposteriorsamples, axis = 0).iterrows():
            log_overall_probs = fill_cond_probs(inf_data, tree_idx, sampled_varpars.values)
            for node in inf_data.trees[tree_idx].walk('postorder'):
                subtending_cond_from_zero = node.cond_probs[:,0]
                subtending_leaf_names = node.get_leaf_names()
                non_subtending_leaf_log_likes_zero = np.log([tree_leaf_likes[el][:,0] for el in tree_leaf_likes.keys() if el not in subtending_leaf_names])
                non_subtending_log_like_zero = np.sum(non_subtending_leaf_log_likes_zero, axis = 0)
                # TODO check this is right... giving positive logprobs, so probably not right!
                log_p_this_node = np.log(subtending_cond_from_zero) + non_subtending_log_like_zero - log_overall_probs
                log_probs[tree_idx][node.name].append(log_p_this_node)
        for name, log_p_this_nodes in log_probs[tree_idx].iteritems():
            # TODO: need to get overall likelihood as well, subtract log-overallprob from above probs
            log_cond_probs = np.array(log_probs[tree_idx][name]).T
            log_mut_probs = pd.DataFrame(log_cond_probs)
            log_mut_probs['family'] = inf_data.data[tree_idx]['family']
            log_mut_probs['position'] = inf_data.data[tree_idx]['position']
            log_mut_probs['nodename'] = name
            log_probs[tree_idx][name] = pd.melt(log_mut_probs, id_vars = ['family', 'position', 'nodename'], var_name = 'rep', value_name = 'logprob')
        log_probs_all_nodes[tree_idx] = pd.concat(log_probs[tree_idx].values(), ignore_index = True)
    all_log_probs = pd.concat(log_probs_all_nodes.values(), ignore_index = True)
    all_log_probs.to_csv(sys.stdout, index = False, sep = str('\t'))

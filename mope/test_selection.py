from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
import os.path
import emcee
import argparse
import sys

import os
os.environ['OPL_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
import numpy as np
import scipy.optimize as opt
import pandas as pd
from functools import partial
import multiprocessing as mp
import numpy.random as npr

from mope import newick
from mope import inference as inf
from mope import util as ut
from mope import data as da
from mope import _binom
from mope.pso import pso
from mope._util import print_csv_line, print_csv_lines
from mope._util import print_parallel_csv_lines
from mope import ascertainment as asc
from mope import likelihoods as li
from mope import transition_data_2d as tdm
from mope import params as par

inf_data = None

def test_one_locus():
    global inf_data
    '''
    =========================
    Initialization / setup
    =========================
    '''

    data_files = ['../examples/data/data_test_allele_counts_with_position.txt']
    tree_files = ['../examples/trees/test_mother_child_simple_tree.newick']
    age_files = ['../examples/data/data_test_allele_counts_with_position_ages.txt']

    inf_data = inf.Inference(
        data_files = data_files,
        tree_files = tree_files,
        age_files = age_files,
        transitions_file = (
            '../examples/transitions_selection/selection_transitions.h5'),
        true_parameters = None,
        start_from = 'prior',
        data_are_freqs = False,
        genome_size = 16569,
        bottleneck_file = None,
        min_freq = 0.0,
        poisson_like_penalty = 1.0,
        print_debug = False,
        log_unif_drift = True,
        inverse_bot_priors = False,
        post_is_prior = False,
        lower_drift_limit = 1e-3,
        upper_drift_limit = 3,
        min_phred_score = 1000,
        selection_model=True
    )
    
    n_branches = 1

    npr.seed(3)
    x0 = npr.uniform(inf_data.lower, inf_data.upper)
    blood_length = 10**x0[0]
    relative_pop_size = 1
    dfe_params = x0[2*n_branches:(2*n_branches+inf_data.dfes[0].nparams)]
    root_start = 2*n_branches+inf_data.dfes[0].nparams
    alpha_start = root_start + 2    # there are two root parameters
    root_params = x0[root_start:alpha_start]
    log10ab, log10polyprob = root_params
    if (-inf_data.abs_sel_zero_value < x0[alpha_start]
            and x0[alpha_start] < inf_data.abs_sel_zero_value):
        alpha = 0.0
    else:
        alpha_var = x0[alpha_start]  # the unadjusted alpha variable
        alpha = np.sign(alpha_var) * (np.abs(alpha_var) -
                                      inf_data.abs_sel_zero_value)

    ll = inf_data.loglike(x0)


    # Get the transition matrix.
    transition_mat = (
        inf_data.transition_data.get_transition_probabilities_2d(
            blood_length, alpha))

    # Get the stationary distribution.
    freqs = inf_data.transition_data.get_bin_frequencies()
    breaks = inf_data.transition_data.get_breaks()
    N = inf_data.transition_data.get_N()
    stat_dist = li.get_stationary_distribution_double_beta(
        freqs, breaks, N, 10**log10ab, 10**log10polyprob)

    # Get the allele frequency likelihoods.
    leaf_likes_mother_blood = inf_data.leaf_likes[0]['mother_blood'][0]
    leaf_likes_child_blood = inf_data.leaf_likes[0]['child_blood'][0]

    # Calculate the unconditional likelihoods.
    root_freq_likes = (
        np.dot(transition_mat, leaf_likes_mother_blood)
        * np.dot(transition_mat, leaf_likes_child_blood))

    allele_frequency_loglike = np.log(np.dot(stat_dist, root_freq_likes))
    print('test ped_ll', allele_frequency_loglike)
    #assert np.isclose(allele_frequency_loglike, -20.24688293)

    # Calculate the probability of ascertainment, i.e., the probability of
    # polymorphism in both the mother and the child.
    e0 = np.zeros_like(stat_dist)
    e0[0] = 1.0
    e1 = np.zeros_like(stat_dist)
    e1[-1] = 1.0
    eboth = np.hstack((e0[:,np.newaxis], e1[:,np.newaxis]))

    # Calculate the probability that both left and right are polymorphic.
    prob_left_poly = 1.0 - np.dot(transition_mat, eboth).sum(1)
    prob_right_poly = 1.0 - np.dot(transition_mat, eboth).sum(1)
    prob_both_poly = prob_left_poly * prob_right_poly
    log_prob_asc = np.log(np.dot(stat_dist, prob_both_poly))
    #assert np.isclose(log_prob_asc, -9.01845689)

    ascertained_ll = (allele_frequency_loglike
                        - log_prob_asc)

    # Get the DFE probabilities.
    alpha_arr = np.array([alpha])
    ratio_pos_to_neg = 10**dfe_params[-2]
    from scipy.special import expit
    prob_zero = expit(dfe_params[0])
    prob_pos = expit(dfe_params[1])
    log10ratiopostoneg = dfe_params[2]
    log10negalphamean = dfe_params[3]
    blood_neg_alpha_mean = 10**log10negalphamean
    blood_pos_alpha_mean = ratio_pos_to_neg * blood_neg_alpha_mean

    print('alpha = ', alpha)

    if alpha == 0.0:
        test_dfe_ll = np.log(prob_zero)
    elif alpha > 0.0:
        test_dfe_ll = (
            np.log(1-prob_zero) + np.log(prob_pos)
            + np.log(blood_pos_alpha_mean) - blood_pos_alpha_mean*alpha)
    elif alpha < 0.0:
        test_dfe_ll = (
            np.log(1-prob_zero) + np.log(1-prob_pos)
            + np.log(blood_neg_alpha_mean) - blood_neg_alpha_mean*alpha)



    prog_dfe_ll = inf_data.dfes[0].get_loglike(
        dfe_params, alpha_arr)
    assert np.isclose(prog_dfe_ll, test_dfe_ll)

    final_ll = test_dfe_ll + ascertained_ll
    assert np.isclose(final_ll, ll)

def test_one_locus_twice():

    # Now testing data with one locus, repeated twice in the same allele count configuration.
    data_files = ['../examples/data/data_test_allele_counts_with_position_twice.txt']
    tree_files = ['../examples/trees/test_mother_child_simple_tree.newick']
    age_files = ['../examples/data/data_test_allele_counts_with_position_two_loci_ages.txt']

    inf_data = inf.Inference(
        data_files = data_files,
        tree_files = tree_files,
        age_files = age_files,
        transitions_file = (
            '../examples/transitions_selection/selection_transitions.h5'),
        true_parameters = None,
        start_from = 'prior',
        data_are_freqs = False,
        genome_size = 16569,
        bottleneck_file = None,
        min_freq = 0.0,
        poisson_like_penalty = 1.0,
        print_debug = False,
        log_unif_drift = True,
        inverse_bot_priors = False,
        post_is_prior = False,
        lower_drift_limit = 1e-3,
        upper_drift_limit = 3,
        min_phred_score = 1000,
        selection_model=True
    )

    npr.seed(3)
    x0 = npr.uniform(inf_data.lower, inf_data.upper)
    n_branches = 1
    ll = inf_data.loglike(x0)

    # After uncommenting inference.py line matching SELTEST, you should see
    # ll ped_ll: [-30.07488094 -30.07488094]
    # ll ped_log_asc_prob: [-23.27427311 -23.27427311]

    # OLD ll ped_ll: [-20.24688293 -20.24688293]
    # OLD ll ped_log_asc_prob: [-16.40931194 -16.40931194]

    # The rest is not tested.


def test_two_loci_two_families():

    # Now testing data with two positions, two families, and a rate parameter.
    data_files = ['../examples/data/data_test_allele_counts_with_position_two_loci.txt']
    tree_files = ['../examples/trees/test_mother_child_rate_tree.newick']
    age_files = ['../examples/data/data_test_allele_counts_with_position_two_loci_ages.txt']

    inf_data = inf.Inference(
        data_files = data_files,
        tree_files = tree_files,
        age_files = age_files,
        transitions_file = (
            '../examples/transitions_selection/selection_transitions.h5'),
        true_parameters = None,
        start_from = 'prior',
        data_are_freqs = False,
        genome_size = 16569,
        bottleneck_file = None,
        min_freq = 0.0,
        poisson_like_penalty = 1.0,
        print_debug = False,
        log_unif_drift = True,
        inverse_bot_priors = False,
        post_is_prior = False,
        lower_drift_limit = 1e-3,
        upper_drift_limit = 3,
        min_phred_score = 1000,
        selection_model=True
    )

    n_branches = 2

    npr.seed(3)
    x0 = npr.uniform(inf_data.lower, inf_data.upper)
    blood_length = 10**x0[0]
    blood_rate = 10**x0[1]

    # Confirmed that length comes before rate.
    blood_length_rel_pop_size = 1.0
    blood_rate_rel_pop_size = 10**x0[3]

    relative_population_sizes = np.array(
        (blood_length_rel_pop_size,
         blood_rate_rel_pop_size))/blood_length_rel_pop_size


    dfe_params = x0[2*n_branches:(2*n_branches+inf_data.dfes[0].nparams)]

    root_start = 2*n_branches+inf_data.dfes[0].nparams
    alpha_start = root_start + 2   # two root parameters
    root_params = x0[root_start:alpha_start]


    log10ab, log10polyprob = root_params

    # Absolute adjustment happens before relative adjustment.
    orig_alpha_loc0 = x0[alpha_start]
    if np.abs(orig_alpha_loc0) < inf_data.abs_sel_zero_value:
        actual_alpha_loc0_l = 0.0
    else:
        actual_alpha_loc0_l = (np.sign(orig_alpha_loc0)
                               * (np.abs(orig_alpha_loc0)
                                  -inf_data.abs_sel_zero_value))
    orig_alpha_loc1 = x0[alpha_start+1]
    if np.abs(orig_alpha_loc1) < inf_data.abs_sel_zero_value:
        actual_alpha_loc1_l = 0.0
    else:
        actual_alpha_loc1_l = (np.sign(orig_alpha_loc1)
                               * (np.abs(orig_alpha_loc1)
                                  -inf_data.abs_sel_zero_value))

    actual_alpha_loc0_r = actual_alpha_loc0_l * relative_population_sizes[1]
    actual_alpha_loc1_r = actual_alpha_loc1_l * relative_population_sizes[1]
    print('test alpha length', actual_alpha_loc0_l, actual_alpha_loc1_l)
    print('test alpha rate', actual_alpha_loc0_r, actual_alpha_loc1_r)

    ll = inf_data.loglike(x0)

    # Get the transition matrix for the non-rate blood.
    transition_mat_length_loc0 = (
        inf_data.transition_data.get_transition_probabilities_2d(
            blood_length, actual_alpha_loc0_l))

    transition_mat_length_loc1 = (
        inf_data.transition_data.get_transition_probabilities_2d(
            blood_length, actual_alpha_loc1_l))

    transition_mat_rate_fam_0_loc0 = (
        inf_data.transition_data.get_transition_probabilities_2d(
            blood_rate*inf_data.data[0]['child_age'].iloc[0], actual_alpha_loc0_r))

    transition_mat_rate_fam_0_loc1 = (
        inf_data.transition_data.get_transition_probabilities_2d(
            blood_rate*inf_data.data[0]['child_age'].iloc[0], actual_alpha_loc1_r))

    transition_mat_rate_fam_1_loc1 = (
        inf_data.transition_data.get_transition_probabilities_2d(
            blood_rate*inf_data.data[0]['child_age'].iloc[2], actual_alpha_loc1_r))


    # Get the stationary distribution.
    freqs = inf_data.transition_data.get_bin_frequencies()
    breaks = inf_data.transition_data.get_breaks()
    N = inf_data.transition_data.get_N()
    stat_dist = li.get_stationary_distribution_double_beta(
        freqs, breaks, N, 10**log10ab, 10**log10polyprob)

    # Get the allele frequency likelihoods.
    leaf_likes_mother_blood = inf_data.leaf_likes[0]['mother_blood']
    leaf_likes_child_blood = inf_data.leaf_likes[0]['child_blood']

    # Calculate the unconditional likelihoods for locus 0, family 0
    root_freq_likes_fam_0_loc_0 = (
        np.dot(transition_mat_length_loc0, leaf_likes_mother_blood[0])
        * np.dot(transition_mat_rate_fam_0_loc0, leaf_likes_child_blood[0]))

    root_freq_likes_fam_0_loc_1 = (
        np.dot(transition_mat_length_loc1, leaf_likes_mother_blood[1])
        * np.dot(transition_mat_rate_fam_0_loc1, leaf_likes_child_blood[1]))

    root_freq_likes_fam_1_loc_1 = (
        np.dot(transition_mat_length_loc1, leaf_likes_mother_blood[2])
        * np.dot(transition_mat_rate_fam_1_loc1, leaf_likes_child_blood[2]))

    root_freq_likes = np.concatenate(
        (
            root_freq_likes_fam_0_loc_0[np.newaxis,:],
            root_freq_likes_fam_0_loc_1[np.newaxis,:],
            root_freq_likes_fam_1_loc_1[np.newaxis,:]
        ))
    allele_frequency_loglikes = np.log(np.dot(root_freq_likes, stat_dist))
    print('test ped_ll:', allele_frequency_loglikes)

    allele_frequency_loglike = np.sum(allele_frequency_loglikes)

    # For each locus/family combination (family 0: loci 0, 1; family 1: locus
    # 1), need to calculate the ascertainment probability

    e0 = np.zeros_like(stat_dist)
    e0[0] = 1.0
    e1 = np.zeros_like(stat_dist)
    e1[-1] = 1.0
    eboth = np.hstack((e0[:,np.newaxis], e1[:,np.newaxis]))

    # family 0, locus 0
    prob_left_poly_loc0 = 1.0 - np.dot(transition_mat_length_loc0, eboth).sum(1)
    prob_right_poly_loc0_fam0 = 1.0 - np.dot(transition_mat_rate_fam_0_loc0, eboth).sum(1)
    prob_both_poly_loc0_fam0 = prob_left_poly_loc0 * prob_right_poly_loc0_fam0
    log_prob_asc_loc0_fam0 = np.log(np.dot(stat_dist, prob_both_poly_loc0_fam0))
    #assert np.isclose(log_prob_asc_loc0_fam0, -2.40282664)

    # family 0, locus 1
    prob_left_poly_loc1 = 1.0 - np.dot(transition_mat_length_loc1, eboth).sum(1)
    prob_right_poly_loc1_fam0 = 1.0 - np.dot(transition_mat_rate_fam_0_loc1, eboth).sum(1)
    prob_both_poly_loc1_fam0 = prob_left_poly_loc1 * prob_right_poly_loc1_fam0
    log_prob_asc_loc1_fam0 = np.log(np.dot(stat_dist, prob_both_poly_loc1_fam0))
    #assert np.isclose(log_prob_asc_loc1_fam0, -4.04565476)

    # family 1, locus 1
    prob_right_poly_loc1_fam1 = 1.0 - np.dot(transition_mat_rate_fam_1_loc1, eboth).sum(1)
    prob_both_poly_loc1_fam1 = prob_left_poly_loc1 * prob_right_poly_loc1_fam1
    log_prob_asc_loc1_fam1 = np.log(np.dot(stat_dist, prob_both_poly_loc1_fam1))
    #assert np.isclose(log_prob_asc_loc1_fam1, -2.39888754)

    log_prob_asc = np.sum(
        [log_prob_asc_loc0_fam0, log_prob_asc_loc1_fam0,
         log_prob_asc_loc1_fam1])

    ascertained_ll = (allele_frequency_loglike
                        - log_prob_asc)


    # Get the DFE probabilities.
    focal_alpha_arr = np.array([actual_alpha_loc0_l, actual_alpha_loc1_l])
    ratio_pos_to_neg = 10**dfe_params[-2]
    focal_neg_alpha_mean = 10**dfe_params[-1]
    focal_pos_alpha_mean = ratio_pos_to_neg * focal_neg_alpha_mean
    from scipy.special import expit

    prob_zero = expit(dfe_params[0])
    prob_pos = expit(dfe_params[1])
    prob_neg = 1.0 - prob_pos
    log10ratiopostoneg = dfe_params[2]

    prog_dfe_ll = inf_data.dfes[0].get_loglike(dfe_params, focal_alpha_arr)
    
    # DFE for each position
    dfe_loc0 = (np.log(1-prob_zero) + np.log(prob_neg)
                + np.log(focal_neg_alpha_mean)
                - focal_neg_alpha_mean*focal_alpha_arr[0])

    dfe_loc1 = np.log(prob_zero)
    test_dfe_ll = dfe_loc0 + dfe_loc1
    assert np.isclose(prog_dfe_ll, test_dfe_ll)

    final_ll = test_dfe_ll + ascertained_ll
    assert np.isclose(final_ll, ll)


def test_two_loci_two_families_two_dfes():

    # Now testing data with two positions, two families, and a rate parameter.
    data_files = ['../examples/data/data_test_allele_counts_with_position_two_loci_with_dfes.txt']
    tree_files = ['../examples/trees/test_mother_child_rate_tree.newick']
    age_files = ['../examples/data/data_test_allele_counts_with_position_two_loci_ages.txt']

    inf_data = inf.Inference(
        data_files = data_files,
        tree_files = tree_files,
        age_files = age_files,
        transitions_file = (
            '../examples/transitions_selection/selection_transitions.h5'),
        true_parameters = None,
        start_from = 'prior',
        data_are_freqs = False,
        genome_size = 16569,
        bottleneck_file = None,
        min_freq = 0.0,
        poisson_like_penalty = 1.0,
        print_debug = False,
        log_unif_drift = True,
        inverse_bot_priors = False,
        post_is_prior = False,
        lower_drift_limit = 1e-3,
        upper_drift_limit = 3,
        min_phred_score = 1000,
        selection_model=True
    )

    n_branches = 2

    npr.seed(1)
    for rep in range(500):
        print('\rrep {}'.format(rep), end='')
        x0 = npr.uniform(inf_data.lower, inf_data.upper)
        blood_length = 10**x0[0]
        blood_rate = 10**x0[1]

        # Confirmed that length comes before rate.
        blood_length_rel_pop_size = 1.0
        blood_rate_rel_pop_size = 10**x0[3]

        relative_population_sizes = np.array(
            (blood_length_rel_pop_size,
             blood_rate_rel_pop_size))/blood_length_rel_pop_size


        dfe0_params = x0[2*n_branches:(2*n_branches+inf_data.dfes[0].nparams)]
        dfe1_params = x0[(2*n_branches+inf_data.dfes[0].nparams):((2*n_branches+inf_data.dfes[0].nparams)+inf_data.dfes[1].nparams)]
        all_dfe_params = np.concatenate((dfe0_params, dfe1_params))

        root_start = 2*n_branches+inf_data.total_num_dfe_params
        alpha_start = root_start + 2   # two root parameters
        root_params = x0[root_start:alpha_start]


        log10ab, log10polyprob = root_params

        # Absolute adjustment happens before relative adjustment.
        orig_alpha_loc0 = x0[alpha_start]
        if np.abs(orig_alpha_loc0) < inf_data.abs_sel_zero_value:
            actual_alpha_loc0_l = 0.0
        else:
            actual_alpha_loc0_l = (np.sign(orig_alpha_loc0)
                                   * (np.abs(orig_alpha_loc0)
                                      -inf_data.abs_sel_zero_value))
        orig_alpha_loc1 = x0[alpha_start+1]
        if np.abs(orig_alpha_loc1) < inf_data.abs_sel_zero_value:
            actual_alpha_loc1_l = 0.0
        else:
            actual_alpha_loc1_l = (np.sign(orig_alpha_loc1)
                                   * (np.abs(orig_alpha_loc1)
                                      -inf_data.abs_sel_zero_value))

        actual_alpha_loc0_r = actual_alpha_loc0_l * relative_population_sizes[1]
        actual_alpha_loc1_r = actual_alpha_loc1_l * relative_population_sizes[1]
        #print('test alpha length', actual_alpha_loc0_l, actual_alpha_loc1_l)
        #print('test alpha rate', actual_alpha_loc0_r, actual_alpha_loc1_r)

        ll = inf_data.loglike(x0)

        # Get the transition matrix for the non-rate blood.
        transition_mat_length_loc0 = (
            inf_data.transition_data.get_transition_probabilities_2d(
                blood_length, actual_alpha_loc0_l))

        transition_mat_length_loc1 = (
            inf_data.transition_data.get_transition_probabilities_2d(
                blood_length, actual_alpha_loc1_l))

        transition_mat_rate_fam_0_loc0 = (
            inf_data.transition_data.get_transition_probabilities_2d(
                blood_rate*inf_data.data[0]['child_age'].iloc[0], actual_alpha_loc0_r))

        transition_mat_rate_fam_0_loc1 = (
            inf_data.transition_data.get_transition_probabilities_2d(
                blood_rate*inf_data.data[0]['child_age'].iloc[0], actual_alpha_loc1_r))

        transition_mat_rate_fam_1_loc1 = (
            inf_data.transition_data.get_transition_probabilities_2d(
                blood_rate*inf_data.data[0]['child_age'].iloc[2], actual_alpha_loc1_r))


        # Get the stationary distribution.
        freqs = inf_data.transition_data.get_bin_frequencies()
        breaks = inf_data.transition_data.get_breaks()
        N = inf_data.transition_data.get_N()
        stat_dist = li.get_stationary_distribution_double_beta(
            freqs, breaks, N, 10**log10ab, 10**log10polyprob)

        # Get the allele frequency likelihoods.
        leaf_likes_mother_blood = inf_data.leaf_likes[0]['mother_blood']
        leaf_likes_child_blood = inf_data.leaf_likes[0]['child_blood']

        # Calculate the unconditional likelihoods for locus 0, family 0
        root_freq_likes_fam_0_loc_0 = (
            np.dot(transition_mat_length_loc0, leaf_likes_mother_blood[0])
            * np.dot(transition_mat_rate_fam_0_loc0, leaf_likes_child_blood[0]))

        root_freq_likes_fam_0_loc_1 = (
            np.dot(transition_mat_length_loc1, leaf_likes_mother_blood[1])
            * np.dot(transition_mat_rate_fam_0_loc1, leaf_likes_child_blood[1]))

        root_freq_likes_fam_1_loc_1 = (
            np.dot(transition_mat_length_loc1, leaf_likes_mother_blood[2])
            * np.dot(transition_mat_rate_fam_1_loc1, leaf_likes_child_blood[2]))

        root_freq_likes = np.concatenate(
            (
                root_freq_likes_fam_0_loc_0[np.newaxis,:],
                root_freq_likes_fam_0_loc_1[np.newaxis,:],
                root_freq_likes_fam_1_loc_1[np.newaxis,:]
            ))
        allele_frequency_loglikes = np.log(np.dot(root_freq_likes, stat_dist))
        #print('test ped_ll:', allele_frequency_loglikes)

        allele_frequency_loglike = np.sum(allele_frequency_loglikes)

        # For each locus/family combination (family 0: loci 0, 1; family 1: locus
        # 1), need to calculate the ascertainment probability

        e0 = np.zeros_like(stat_dist)
        e0[0] = 1.0
        e1 = np.zeros_like(stat_dist)
        e1[-1] = 1.0
        eboth = np.hstack((e0[:,np.newaxis], e1[:,np.newaxis]))

        # family 0, locus 0
        prob_left_poly_loc0 = 1.0 - np.dot(transition_mat_length_loc0, eboth).sum(1)
        prob_right_poly_loc0_fam0 = 1.0 - np.dot(transition_mat_rate_fam_0_loc0, eboth).sum(1)
        prob_both_poly_loc0_fam0 = prob_left_poly_loc0 * prob_right_poly_loc0_fam0
        log_prob_asc_loc0_fam0 = np.log(np.dot(stat_dist, prob_both_poly_loc0_fam0))

        # family 0, locus 1
        prob_left_poly_loc1 = 1.0 - np.dot(transition_mat_length_loc1, eboth).sum(1)
        prob_right_poly_loc1_fam0 = 1.0 - np.dot(transition_mat_rate_fam_0_loc1, eboth).sum(1)
        prob_both_poly_loc1_fam0 = prob_left_poly_loc1 * prob_right_poly_loc1_fam0
        log_prob_asc_loc1_fam0 = np.log(np.dot(stat_dist, prob_both_poly_loc1_fam0))

        # family 1, locus 1
        prob_right_poly_loc1_fam1 = 1.0 - np.dot(transition_mat_rate_fam_1_loc1, eboth).sum(1)
        prob_both_poly_loc1_fam1 = prob_left_poly_loc1 * prob_right_poly_loc1_fam1
        log_prob_asc_loc1_fam1 = np.log(np.dot(stat_dist, prob_both_poly_loc1_fam1))

        #print('test log_asc_prob:', np.array(
        #    [log_prob_asc_loc0_fam0, log_prob_asc_loc1_fam0,
        #     log_prob_asc_loc1_fam1]))

        log_prob_asc = np.sum(
            [log_prob_asc_loc0_fam0, log_prob_asc_loc1_fam0,
             log_prob_asc_loc1_fam1])

        ascertained_ll = (allele_frequency_loglike
                            - log_prob_asc)


        #---------------
        # Get the DFE probabilities.
        from scipy.special import expit

        #print('test all_dfe_params:', all_dfe_params)

        focal_alpha_arr = np.array([actual_alpha_loc0_l, actual_alpha_loc1_l])

        ratio_pos_to_neg_l0 = 10**dfe0_params[-2]
        focal_neg_alpha_mean_l0 = 10**dfe0_params[-1]
        focal_pos_alpha_mean_l0 = ratio_pos_to_neg_l0 * focal_neg_alpha_mean_l0
        prob_zero_l0 = expit(dfe0_params[0])
        prob_pos_l0 = expit(dfe0_params[1])
        prob_neg_l0 = 1.0 - prob_pos_l0
        log10ratiopostoneg_l0 = dfe0_params[2]

        ratio_pos_to_neg_l1 = 10**dfe1_params[-2]
        focal_neg_alpha_mean_l1 = 10**dfe1_params[-1]
        focal_pos_alpha_mean_l1 = ratio_pos_to_neg_l1 * focal_neg_alpha_mean_l1
        prob_zero_l1 = expit(dfe1_params[0])
        prob_pos_l1 = expit(dfe1_params[1])
        prob_neg_l1 = 1.0 - prob_pos_l1
        log10ratiopostoneg_l1 = dfe1_params[2]

        prog_dfe_ll = inf_data.get_dfe_loglikes(all_dfe_params, focal_alpha_arr)
        
        # DFE for each position

        a0 = focal_alpha_arr[0]
        a1 = focal_alpha_arr[1]

        if a0 == 0.0:
            dfe_loc0 = np.log(prob_zero_l0)
        else:
            dfe_loc0 = np.log(1-prob_zero_l0)
            if a0 < 0:
                dfe_loc0 += (np.log(prob_neg_l0) + np.log(focal_neg_alpha_mean_l0)
                             - focal_neg_alpha_mean_l0*a0)
            else:
                dfe_loc0 += (np.log(prob_pos_l0) + np.log(focal_pos_alpha_mean_l0)
                             - focal_pos_alpha_mean_l0*a0)

        if a1 == 0.0:
            dfe_loc1 = np.log(prob_zero_l1)
        else:
            dfe_loc1 = np.log(1-prob_zero_l1)
            if a1 < 0:
                dfe_loc1 += (np.log(prob_neg_l1) + np.log(focal_neg_alpha_mean_l1)
                             - focal_neg_alpha_mean_l1*a1)
            else:
                dfe_loc1 += (np.log(prob_pos_l1) + np.log(focal_pos_alpha_mean_l1)
                             - focal_pos_alpha_mean_l1*a1)
                                                          
        test_dfe_ll = dfe_loc0 + dfe_loc1
        #print('test_dfe_ll:', test_dfe_ll)
        #print('prog_dfe_ll:', prog_dfe_ll)
        assert np.isclose(prog_dfe_ll, test_dfe_ll)

        final_ll = test_dfe_ll + ascertained_ll
        assert np.isclose(final_ll, ll)


if __name__ == '__main__':
    #test_one_locus()
    #test_one_locus_twice()
    test_two_loci_two_families_two_dfes()

    print('\nTESTING PASSED\n')

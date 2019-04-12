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

    x0 = np.array([-1.54289540e+00, -1.39242535e-01,  1.23316051e+00,
                   5.38598196e-01, -6.10761605e-01, -3.18695298e+00,
                   -5.06171510e+00,  2.42899260e+02])
    blood_length = 10**x0[0]
    blood_neg_alpha_mean = 10**x0[1]
    dfe_params = x0[2*n_branches:(2*n_branches+inf_data.dfes[0].nparams)]
    root_start = 2*n_branches+inf_data.dfes[0].nparams
    alpha_start = root_start + 2
    root_params = x0[root_start:alpha_start]
    log10ab, log10polyprob = root_params
    alpha = x0[alpha_start] - inf_data.abs_sel_zero_value

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
    assert np.isclose(allele_frequency_loglike, -20.24688293)

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
    assert np.isclose(log_prob_asc, -16.40931194)

    ascertained_ll = (allele_frequency_loglike
                        - log_prob_asc)

    # Get the DFE probabilities.
    alpha_arr = np.array([alpha])
    ratio_pos_to_neg = 10**dfe_params[-1]
    blood_pos_alpha_mean = ratio_pos_to_neg * blood_neg_alpha_mean
    from scipy.special import expit
    prob_zero = expit(dfe_params[0])
    prob_pos = expit(dfe_params[1])
    log10ratiopostoneg = dfe_params[2]

    prog_dfe_ll = inf_data.dfes[0].get_loglike(
        dfe_params, blood_neg_alpha_mean, alpha_arr)
    test_dfe_ll = (
        np.log(1-prob_zero) + np.log(prob_pos)
        + np.log(blood_pos_alpha_mean) - blood_pos_alpha_mean*alpha)
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

    x0 = np.array([-1.54289540e+00, -1.39242535e-01,  1.23316051e+00,  5.38598196e-01,
                   -6.10761605e-01, -3.18695298e+00, -5.06171510e+00,  2.42899260e+02])
    n_branches = 1
    ll = inf_data.loglike(x0)

    # After uncommenting inference.py line matching SELTEST, you should see
    # ll ped_ll: [-20.24688293 -20.24688293]
    # ll ped_log_asc_prob: [-16.40931194 -16.40931194]

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

    x0 = np.array([ -1.5428954,  -1.81236204,  -0.5889465,  -0.82046727,
                -0.91614241,   1.75072936,  -0.49930231,  -0.97404299,
                -0.32703516, -72.26625833, 180.86952361])
    blood_length = 10**x0[0]
    blood_rate = 10**x0[1]

    # Confirmed that length comes before rate.
    blood_length_neg_alpha_mean = 10**x0[2]
    blood_rate_neg_alpha_mean = 10**x0[3]

    relative_alpha_means = np.array((blood_length_neg_alpha_mean, blood_rate_neg_alpha_mean))/blood_length_neg_alpha_mean

    dfe_params = x0[2*n_branches:(2*n_branches+inf_data.dfes[0].nparams)]

    root_start = 2*n_branches+inf_data.dfes[0].nparams
    alpha_start = root_start + 2
    root_params = x0[root_start:alpha_start]

    log10ab, log10polyprob = root_params

    # Absolute adjustment happens before relative adjustment.
    orig_alpha_loc0 = x0[alpha_start]
    actual_alpha_loc0_l = np.sign(orig_alpha_loc0) * (np.abs(orig_alpha_loc0)-inf_data.abs_sel_zero_value)
    orig_alpha_loc1 = x0[alpha_start+1]
    actual_alpha_loc1_l = np.sign(orig_alpha_loc1) * (np.abs(orig_alpha_loc1)-inf_data.abs_sel_zero_value)

    actual_alpha_loc0_r = actual_alpha_loc0_l * relative_alpha_means[1]
    actual_alpha_loc1_r = actual_alpha_loc1_l * relative_alpha_means[1]

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
    #assert np.all(np.isclose(allele_frequency_loglikes, np.array([-14.76720745, -23.1917792,  -19.16966547])))
    assert np.all(np.isclose(allele_frequency_loglikes, np.array([-13.97914171, -22.82499637, -18.80418358])))
    print(np.sum(allele_frequency_loglikes), np.sum(np.array([-13.97914171, -22.82499637, -18.80418358])))
    assert np.isclose(allele_frequency_loglike, -55.60832166)

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
    assert np.isclose(log_prob_asc_loc0_fam0, -2.40282664)

    # family 0, locus 1
    prob_left_poly_loc1 = 1.0 - np.dot(transition_mat_length_loc1, eboth).sum(1)
    prob_right_poly_loc1_fam0 = 1.0 - np.dot(transition_mat_rate_fam_0_loc1, eboth).sum(1)
    prob_both_poly_loc1_fam0 = prob_left_poly_loc1 * prob_right_poly_loc1_fam0
    log_prob_asc_loc1_fam0 = np.log(np.dot(stat_dist, prob_both_poly_loc1_fam0))
    assert np.isclose(log_prob_asc_loc1_fam0, -4.04565476)

    # family 1, locus 1
    prob_right_poly_loc1_fam1 = 1.0 - np.dot(transition_mat_rate_fam_1_loc1, eboth).sum(1)
    prob_both_poly_loc1_fam1 = prob_left_poly_loc1 * prob_right_poly_loc1_fam1
    log_prob_asc_loc1_fam1 = np.log(np.dot(stat_dist, prob_both_poly_loc1_fam1))
    assert np.isclose(log_prob_asc_loc1_fam1, -2.39888754)

    log_prob_asc = np.sum(
        [log_prob_asc_loc0_fam0, log_prob_asc_loc1_fam0,
         log_prob_asc_loc1_fam1])

    ascertained_ll = (allele_frequency_loglike
                        - log_prob_asc)


    # Get the DFE probabilities.
    focal_alpha_arr = np.array([actual_alpha_loc0_l, actual_alpha_loc1_l])
    ratio_pos_to_neg = 10**dfe_params[-1]
    focal_neg_alpha_mean = blood_length_neg_alpha_mean
    focal_pos_alpha_mean = ratio_pos_to_neg * blood_length_neg_alpha_mean
    from scipy.special import expit

    prob_zero = expit(dfe_params[0])
    prob_pos = expit(dfe_params[1])
    prob_neg = 1.0 - prob_pos
    log10ratiopostoneg = dfe_params[2]

    prog_dfe_ll = inf_data.dfes[0].get_loglike(
        dfe_params, blood_length_neg_alpha_mean, focal_alpha_arr)
    
    # DFE for each position
    dfe_loc0 = (np.log(1-prob_zero) + np.log(prob_neg)
                + np.log(focal_neg_alpha_mean)
                - focal_neg_alpha_mean*focal_alpha_arr[0])

    dfe_loc1 = (np.log(1-prob_zero) + np.log(prob_pos)
                + np.log(focal_pos_alpha_mean)
                - focal_pos_alpha_mean*focal_alpha_arr[1])
    test_dfe_ll = dfe_loc0 + dfe_loc1
    assert np.isclose(prog_dfe_ll, test_dfe_ll)

    final_ll = test_dfe_ll + ascertained_ll
    assert np.isclose(final_ll, ll)


if __name__ == '__main__':
    test_one_locus()
    test_one_locus_twice()
    test_two_loci_two_families()

    print('\nTESTING PASSED\n')

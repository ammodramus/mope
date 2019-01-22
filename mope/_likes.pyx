#cython: boundscheck=True, wraparound=True
from __future__ import division
import numpy as np
cimport numpy as np
import pandas as pd
from libc.math cimport log
from libc.string cimport memset
cimport cython


cdef int ONE = 1
cdef int ZERO = 0
cdef double DONE = 1
cdef double DZERO = 0

#################
# resetting
#################

@cython.boundscheck(False)
@cython.wraparound(False)
def reset_likes_ones(np.ndarray[np.float64_t, ndim=2] likes): 
    cdef int nrow = likes.shape[0], ncol = likes.shape[1]
    cdef int i,j 
    for i in range(nrow):
        for j in range(ncol):
            likes[i,j] = 1.0

@cython.boundscheck(False)
@cython.wraparound(False)
def reset_likes_zeros(np.ndarray[np.float64_t, ndim=2] likes): 
    cdef int nrow = likes.shape[0], ncol = likes.shape[1]
    cdef int i,j 
    for i in range(nrow):
        for j in range(ncol):
            likes[i,j] = 0.0

#####################
# newick functions
#####################

def compute_rate_transition_likelihood(
        np.ndarray[np.float64_t,ndim=2] leaf_likes,
        np.ndarray[np.float64_t,ndim=2] ancestor_likes,
        np.ndarray[np.float64_t,ndim=1] leaf_lengths,
        double mut_rate,
        transitions):

    cdef int num_loci = leaf_likes.shape[0]
    cdef int i
    cdef double time
    for i in range(num_loci):
        time = leaf_lengths[i]
        P = transitions.get_transition_probabilities_2d(
                time, mut_rate)
        ancestor_likes[i,:] *= np.dot(P, leaf_likes[i,:])


def compute_rate_transition_likelihood_selection(
        np.ndarray[np.float64_t,ndim=2] leaf_likes,
        np.ndarray[np.float64_t,ndim=2] ancestor_likes,
        np.ndarray[np.float64_t,ndim=1] leaf_lengths,
        np.ndarray[np.float64_t,ndim=1] leaf_locus_alphas,
        transitions):

    cdef int num_loci = leaf_likes.shape[0]
    cdef int i
    cdef double time, alpha
    for i in range(num_loci):
        time = leaf_lengths[i]
        alpha = leaf_locus_alphas[i]
        P = transitions.get_transition_probabilities_2d(
                time, alpha)
        ancestor_likes[i,:] *= np.dot(P, leaf_likes[i,:])


def compute_leaf_zero_transition_likelihood(
        np.ndarray[np.float64_t,ndim=2] leaf_likes,
        np.ndarray[np.float64_t,ndim=2] ancestor_likes,
        np.ndarray[np.float64_t,ndim=1] leaf_lengths,
        double mut_rate,
        transitions):

    cdef int num_loci = leaf_likes.shape[0]
    cdef int i
    cdef double time, zero_prob
    for i in range(num_loci):
        time = leaf_lengths[i]
        P = transitions.get_transition_probabilities_2d(
                time, mut_rate)
        Pzero = np.zeros_like(P)
        Pzero[0,0] = P[0,0]
        ancestor_likes[i,:] *= np.dot(Pzero, leaf_likes[i,:])


def compute_leaf_focal_node_zero_transition_likelihood(
        np.ndarray[np.float64_t,ndim=2] leaf_likes,
        np.ndarray[np.float64_t,ndim=2] ancestor_likes,
        np.ndarray[np.float64_t,ndim=1] leaf_lengths,
        double mut_rate,
        transitions):

    cdef int num_loci = leaf_likes.shape[0]
    cdef int i
    cdef double time, zero_prob
    for i in range(num_loci):
        time = leaf_lengths[i]
        P = transitions.get_transition_probabilities_2d(
                time, mut_rate)
        Pzero = np.zeros_like(P)
        Pzero[0,1:-1] = P[0,1:-1]
        ancestor_likes[i,:] *= np.dot(Pzero, leaf_likes[i,:])


def compute_leaf_zero_transition_likelihood_descendant(
        np.ndarray[np.float64_t,ndim=2] leaf_likes,
        np.ndarray[np.float64_t,ndim=2] ancestor_likes,
        np.ndarray[np.float64_t,ndim=1] leaf_lengths,
        double mut_rate,
        transitions):

    cdef int num_loci = leaf_likes.shape[0]
    cdef int i
    cdef double time, zero_prob
    for i in range(num_loci):
        time = leaf_lengths[i]
        P = transitions.get_transition_probabilities_2d(
                time, mut_rate)
        Pzero = np.zeros_like(P)
        Pzero[1:-1,1:-1] = P[1:-1,1:-1]
        ancestor_likes[i,:] *= np.dot(Pzero, leaf_likes[i,:])


def compute_length_transition_likelihood(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double node_length,
            double mut_rate,
            transitions):

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = transitions.get_transition_probabilities_2d(
            node_length,
            mut_rate)
    for i in range(num_loci):
        ancestor_likes[i] *= np.dot(P, node_likes[i])


def compute_length_transition_likelihood_selection(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double node_length,
            np.ndarray[np.float64_t,ndim=1] leaf_locus_alphas,
            transitions):

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    cdef double alpha
    for i in range(num_loci):
        alpha = leaf_locus_alphas[i]
        P = transitions.get_transition_probabilities_2d(
                node_length,
                alpha)
        ancestor_likes[i] *= np.dot(P, node_likes[i])


def compute_length_transition_likelihood_zero(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double node_length,
            double mut_rate,
            transitions):

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = transitions.get_transition_probabilities_2d(
            node_length,
            mut_rate)
    Pzero = np.zeros_like(P)
    Pzero[0,0] = P[0,0]
    for i in range(num_loci):
        ancestor_likes[i] *= np.dot(Pzero, node_likes[i])


def compute_length_transition_likelihood_zero_focalnode(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double node_length,
            double mut_rate,
            transitions):

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = transitions.get_transition_probabilities_2d(
            node_length,
            mut_rate)
    Pzero = np.zeros_like(P)
    Pzero[0,1:-1] = P[0,1:-1]
    for i in range(num_loci):
        ancestor_likes[i] *= np.dot(Pzero, node_likes[i])


def compute_length_transition_likelihood_zero_descendant(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double node_length,
            double mut_rate,
            transitions):

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = transitions.get_transition_probabilities_2d(
            node_length,
            mut_rate)
    Pzero = np.zeros_like(P)
    Pzero[1:-1,1:-1] = P[1:-1,1:-1]
    for i in range(num_loci):
        ancestor_likes[i] *= np.dot(Pzero, node_likes[i])
        

def compute_bottleneck_transition_likelihood(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double bottleneck_size,
            double mut_rate,
            bottlenecks):
    '''
    this could just be the same function as compute_branch_transition_...
    '''

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = bottlenecks.get_transition_probabilities_2d(
            bottleneck_size,
            mut_rate)
    for i in range(num_loci):
        ancestor_likes[i] *= np.dot(P, node_likes[i])


def compute_bottleneck_transition_likelihood_zero(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double bottleneck_size,
            double mut_rate,
            bottlenecks):
    '''
    this could just be the same function as compute_branch_transition_...
    '''

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = bottlenecks.get_transition_probabilities_2d(
            bottleneck_size,
            mut_rate)
    Pzero = np.zeros_like(P)
    Pzero[0,0] = P[0,0]
    for i in range(num_loci):
        ancestor_likes[i] *= np.dot(Pzero, node_likes[i])


def compute_bottleneck_transition_likelihood_zero_focalnode(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double bottleneck_size,
            double mut_rate,
            bottlenecks):
    '''
    this could just be the same function as compute_branch_transition_...
    '''

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = bottlenecks.get_transition_probabilities_2d(
            bottleneck_size,
            mut_rate)
    Pzero = np.zeros_like(P)
    Pzero[0,1:-1] = P[0,1:-1]
    for i in range(num_loci):
        ancestor_likes[i,:] *= np.dot(Pzero, node_likes[i,:])


def compute_bottleneck_transition_likelihood_zero_descendant(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double bottleneck_size,
            double mut_rate,
            bottlenecks):
    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = bottlenecks.get_transition_probabilities_2d(
            bottleneck_size,
            mut_rate)
    Pzero = np.zeros_like(P)
    Pzero[1:-1,1:-1] = P[1:-1,1:-1]
    for i in range(num_loci):
        ancestor_likes[i,:] *= np.dot(Pzero, node_likes[i,:])


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_root_log_like(
            np.ndarray[np.float64_t,ndim=2] root_cond_probs,
            np.ndarray[np.float64_t,ndim=1] stationary_distribution):
    cdef int i, num_loci = root_cond_probs.shape[0]
    cdef double locus_loglike, loglike
    loglike = 0.0
    for i in range(num_loci):
        locus_loglike = (
                log(np.dot(root_cond_probs[i], stationary_distribution)))
        loglike += locus_loglike
    return loglike

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_root_locus_log_likes(
            np.ndarray[np.float64_t,ndim=2] root_cond_probs,
            np.ndarray[np.float64_t,ndim=1] stationary_distribution):
    cdef int i, num_loci = root_cond_probs.shape[0]
    cdef double locus_loglike, loglike
    cdef list loglikes = []
    for i in range(num_loci):
        locus_loglike = (
                log(np.dot(root_cond_probs[i], stationary_distribution)))
        loglikes.append(locus_loglike)
    return loglikes



###############
# log-space
###############

def compute_log_leaf_transition_likelihood(
        np.ndarray[np.float64_t,ndim=2] leaf_likes,
        np.ndarray[np.float64_t,ndim=2] log_ancestor_likes,
        np.ndarray[np.float64_t,ndim=1] leaf_lengths,
        double mut_rate,
        transitions):

    cdef int num_loci = leaf_likes.shape[0]
    cdef int i
    cdef double time
    for i in range(num_loci):
        time = leaf_lengths[i]
        P = transitions.get_transition_probabilities_2d(
                time, mut_rate)
        log_ancestor_likes[i] += np.log(np.dot(P, leaf_likes[i]))

def compute_log_branch_transition_likelihood(
            np.ndarray[np.float64_t,ndim=2] log_node_likes,
            np.ndarray[np.float64_t,ndim=2] log_ancestor_likes,
            double node_length,
            double mut_rate,
            transitions):

    cdef int num_loci = log_node_likes.shape[0]
    cdef int i
    P = transitions.get_transition_probabilities_2d(
            node_length,
            mut_rate)
    for i in range(num_loci):
        log_ancestor_likes[i] += np.log(np.dot(P, np.exp(log_node_likes[i])))

def compute_root_log_like_log_space(
            np.ndarray[np.float64_t,ndim=2] log_root_cond_probs,
            np.ndarray[np.float64_t,ndim=1] stationary_distribution,
            np.ndarray[np.int_t,ndim=1] counts):
    cdef int i, num_loci = log_root_cond_probs.shape[0]
    cdef double loglike = 0.0
    for i in range(num_loci):
        loglike += (log(np.dot(np.exp(log_root_cond_probs[i]),
            stationary_distribution)) *
                counts[i])
    return loglike

#######################################
# Ascertainment
#######################################


def compute_log_asc_prob(
            np.ndarray[np.float64_t,ndim=2] root_non_poly_probs,
            np.ndarray[np.float64_t,ndim=1] stationary_distribution):
    cdef int i, num_loci = root_non_poly_probs.shape[0]
    assert num_loci == 2
    cdef double ppoly = 1.0
    ppoly -= np.dot(stationary_distribution, root_non_poly_probs[0])
    ppoly -= np.dot(stationary_distribution, root_non_poly_probs[1])
    assert ppoly >= 0 and ppoly <= 1
    return log(ppoly)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def compute_log_asc_prob_2(
            np.ndarray[np.float64_t,ndim=2] root_non_poly_probs,
            np.ndarray[np.float64_t,ndim=1] stationary_distribution,
            np.ndarray[np.long_t,ndim=1] locus_counts_np):
    cdef int i, num_loci = root_non_poly_probs.shape[0]/2
    cdef long [:] locus_counts = locus_counts_np
    cdef double ppoly, log_asc_prob
    log_asc_prob = 0.0
    for i in range(num_loci):
        ppoly = 1.0
        ppoly -= np.dot(stationary_distribution, root_non_poly_probs[2*i])
        ppoly -= np.dot(stationary_distribution, root_non_poly_probs[2*i+1])
        log_asc_prob += log(ppoly) * locus_counts[i]

    return log_asc_prob

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_individual_locus_asc_logprobs(
            np.ndarray[np.float64_t,ndim=2] root_non_poly_probs,
            np.ndarray[np.float64_t,ndim=1] stationary_distribution,
            np.ndarray[np.long_t,ndim=1] locus_counts_np):
    cdef int i, num_loci = root_non_poly_probs.shape[0]/2
    cdef long [:] locus_counts = locus_counts_np
    cdef double ppoly, log_asc_prob
    cdef np.ndarray[np.float64_t,ndim=1] logprobs_np = np.zeros(num_loci)
    cdef double [:] logprobs = logprobs_np
    log_asc_prob = 0.0
    for i in range(num_loci):
        ppoly = 1.0
        ppoly -= np.dot(stationary_distribution, root_non_poly_probs[2*i])
        ppoly -= np.dot(stationary_distribution, root_non_poly_probs[2*i+1])
        logprobs[i] = log(ppoly)

    return logprobs_np


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def compute_log_asc_prob_both_children(
        np.ndarray[np.float64_t,ndim=2] left_poly_probs,
        np.ndarray[np.float64_t,ndim=2] right_poly_probs,
        np.ndarray[np.float64_t,ndim=1] stationary_distribution):
    cdef int i, num_loci = left_poly_probs.shape[0]
    cdef double ppoly
    cdef np.ndarray[np.float64_t,ndim=1] probs_np = np.ones(num_loci)
    cdef double [:] probs = probs_np

    for i in range(num_loci):
        ppoly = 1.0
        ppoly *= np.dot(stationary_distribution, left_poly_probs[i])
        ppoly *= np.dot(stationary_distribution, right_poly_probs[i])
        probs[i] = log(ppoly)

    return probs_np

#cython: boundscheck=True, wraparound=True
from __future__ import division
import numpy as np
cimport numpy as np
import pandas as pd
import transition_data_mut as tdm
from libc.math cimport log, exp
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

cdef int _binary_index(double freq, double [:] freqs, int nfreqs):
    cdef int start, end, mid
    cdef double midval, midvalp1

    start = 0
    end = nfreqs - 1

    if freq > freqs[end]:
        raise ValueError('freq ({}) greater than maximum in freqs'.format(
            freq))
    if freq < freqs[start]:
        raise ValueError('freq ({}) less than minimum in freqs'.format(
            freq))

    while True:
        mid = start + (end-start)//2
        midval = freqs[mid]
        midvalp1 = freqs[mid+1]
        if midval <= freq:
            if midvalp1 > freq:
                return mid
            elif midvalp1 <= freq:
                start = mid+1
                continue
        if midval > freq:
            end = mid
            continue

@cython.boundscheck(False)
@cython.wraparound(False)
def reset_likes_zeros(np.ndarray[np.float64_t, ndim=2] likes): 
    cdef int nrow = likes.shape[0], ncol = likes.shape[1]
    cdef int i,j 
    for i in range(nrow):
        for j in range(ncol):
            likes[i,j] = 0.0

def _get_transition_probabilities_just_mutation(
        double scaled_time,
        double mut_rate,
        double [:] freqs,
        int nfreqs):

    P_np = np.zeros((nfreqs, nfreqs), dtype = np.float64)
    cdef double [:,:] P = P_np
    cdef double xt, frac_l, frac_u, freqlx, frequx
    cdef int lx  # index of the freq less than or equal to xt

    for i in xrange(nfreqs):
        x = freqs[i]
        xt = 0.5 + (x-0.5)*exp(-2.0*mut_rate*scaled_time)
        lx = _binary_index(xt, freqs, nfreqs)
        freqlx = freqs[lx]
        frequx = freqs[lx+1]
        frac_u = (xt-freqlx)/(frequx-freqlx)
        frac_l = 1.0 - frac_u
        P[i,lx] = frac_l
        P[i,lx+1] = frac_u

    return P
        

#####################
# newick functions
#####################

def compute_leaf_transition_likelihood(
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
        P = transitions.get_transition_probabilities_time_mutation(
                time, mut_rate)
        ancestor_likes[i,:] *= np.dot(P, leaf_likes[i,:])

def compute_mutation_transition_likelihood(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            np.ndarray[np.float64_t,ndim=1] mother_birth_ages,
            double first_mut_time,
            double mut_rate,
            transitions):
    '''
    mother_birth_ages are just the node_lengths from the Inference object
    first_mut_time is the *age* at which mutation begins to have an effect
    mut_rate is the rate of convergence to 0.5 in years
    '''

    cdef np.ndarray[np.float64_t,ndim=1] freqs_np = transitions._frequencies
    cdef double [:] freqs = freqs_np
    cdef int nfreqs = freqs_np.shape[0]
    cdef int i
    cdef int num_loci = node_likes.shape[0]
    cdef double age, mut_time

    for i in range(num_loci):
        age = mother_birth_ages[i]
        if age > first_mut_time:
            mut_time = age - first_mut_time
            P = _get_transition_probabilities_just_mutation(
                    mut_time, mut_rate, freqs, nfreqs)
            ancestor_likes[i] *= np.dot(P, node_likes[i])
        else:
            # identity: no effect of mutation
            ancestor_likes[i] = node_likes[i].copy()



def compute_branch_transition_likelihood(
            np.ndarray[np.float64_t,ndim=2] node_likes,
            np.ndarray[np.float64_t,ndim=2] ancestor_likes,
            double node_length,
            double mut_rate,
            transitions):

    cdef int num_loci = node_likes.shape[0]
    cdef int i
    P = transitions.get_transition_probabilities_time_mutation(
            node_length,
            mut_rate)
    for i in range(num_loci):
        ancestor_likes[i] *= np.dot(P, node_likes[i])




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
    P = bottlenecks.get_transition_probabilities_time_mutation(
            bottleneck_size,
            mut_rate)
    if P is None:
        raise ValueError('invalid bottleneck size: {} or mut rate: {}'.format(
            bottleneck_size, mut_rate))
    for i in range(num_loci):
        ancestor_likes[i] *= np.dot(P, node_likes[i])


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
        P = transitions.get_transition_probabilities_time_mutation(
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
    P = transitions.get_transition_probabilities_time_mutation(
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

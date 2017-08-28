from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport ceil

from _wf cimport binom_rv

cdef void reset_zeros(double [:,:] x, int n):
    cdef int i, j
    for i in range(n):
        for j in range(n):
            x[i,j] = 0.0

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def get_non_asc_probs(
        np.ndarray[np.float64_t,ndim=1] qualities_np,
        np.ndarray[np.float64_t,ndim=1] freqs_np,
        double min_empirical_freq):
    '''
    qualities             base qualities of reads calling either major or minor
                          allele
    freqs                 true frequencies for which we need an ascertainment
                          probability
    min_empirical_freq    minimum empirical frequency of the alternative
                          allele
    '''
    cdef int i, j, k, nqual, nfreqs, min_alt_count
    cdef double error_prob, q, alt_prob, f
    cdef double [:] alt_probs
    cdef double [:] freqs
    cdef double [:] qualities
    cdef double [:] non_asc_probs
    cdef double [:,:] P

    nqual = qualities_np.shape[0]
    nfreqs = freqs_np.shape[0]
    qualities = qualities_np
    freqs = freqs_np

    non_asc_probs_np = np.zeros(nfreqs)
    non_asc_probs = non_asc_probs_np

    min_alt_count = <int>ceil(min_empirical_freq*nqual)

    '''
    goal is to find P_{nqual}(min_alt_count), where P_j(i) is the probability
    of having i alternative alleles after considering j reads. this can be
    calculated dynamically.
    '''

    # alt_probs[j] will store the alternative-allele probability of the j'th
    # read, *0-based*
    alt_probs_np = np.zeros(nqual, dtype = np.float64)
    alt_probs = alt_probs_np

    # P[i-1,j] is P_i(j), the probability of having j alternative alleles after
    # i reads. We want \sum_{i=0}^{min_alt_count-1} P[nqual-1,i] for each freq,
    # which will be stored in non_asc_probs_np.
    P_np = np.zeros((nqual, nqual))
    P = P_np

    for i in range(nfreqs):
        f = freqs[i]
        #P_np.fill(0)
        reset_zeros(P, nqual)
        # set P[0,0] and P[0,1]
        q = qualities[0]
        error_prob = 10.0**(-q/10.0)
        alt_probs[0] = f*(1.0-error_prob) + (1-f)*error_prob
        P[0,0] = (1-alt_probs[0])
        P[0,1] = alt_probs[0]
        for j in range(1, nqual):
            q = qualities[j]
            error_prob = 10.0**(-q/10.0)
            alt_probs[j] = f*(1.0-error_prob) + (1-f)*error_prob
            alt_prob = alt_probs[j]

            # take care of P[j,0]
            P[j,0] = (1-alt_prob)*P[j-1,0]
            # j+1 is the max number of successes, so loop through j+2
            for k in range(1,j+2):
                P[j,k] = alt_prob*P[j-1,k-1] + (1-alt_prob)*P[j-1,k]
        non_asc_probs[i] = 0.0
        for j in range(min_alt_count):
            non_asc_probs[i] += P[nqual-1,j]

    return non_asc_probs_np

def empirical_non_asc_probs(
        np.ndarray[np.float64_t,ndim=1] qualities,
        int num_reps, 
        double f,
        double min_freq
        ):
    cdef int rep
    cdef double count
    error_probs = 10.0**(-qualities/10.0)
    cdef np.ndarray[np.float64_t,ndim=1] ps_np = f*(1-error_probs) + (1-f)*error_probs

    cdef double [:] ps = ps_np

    cdef double p, alt

    counts = []
    for rep in range(num_reps):
        count = 0
        for p in ps:
            alt = binom_rv(p,1)
            count += alt
        counts.append(count)
    counts = np.array(counts)
    emp_asc_prob = (counts/ps.shape[0] >= min_freq).sum() / num_reps
    return emp_asc_prob

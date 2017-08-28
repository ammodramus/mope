import time
cimport numpy as np
import numpy as np
cimport cython

cdef extern from "gsl/gsl_rng.h":
   ctypedef struct gsl_rng_type:
       pass
   ctypedef struct gsl_rng:
       pass
   gsl_rng_type *gsl_rng_mt19937
   gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
   void gsl_rng_set(gsl_rng *, unsigned long s)

cdef extern from "gsl/gsl_randist.h":
   double gsl_ran_binomial(gsl_rng *r, double p, unsigned int n)
  
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
gsl_rng_set(r, <unsigned long>time.time())

cpdef double binom_rv(double p, int n):
    cdef unsigned int un = <unsigned int>n
    return gsl_ran_binomial(r, p, un)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int simulate_wright_fisher_transition(int N, int i0, int num_gens,
        double mut_rate):
    '''
    simulate Wright-Fisher transitions in allele frequencies with genetic drift
    and mutation

    N                      haploid population size
    i0                     initial allele count
    num_gens               number of generations
    mut_rate               mutation rate
    '''
    cdef int gen
    cdef double u, x, j
    j = float(i0)
    u = mut_rate
    for gen in range(num_gens):
        x = j/N
        x = (1-u)*x + u*(1-x)
        j = binom_rv(x, N)
    return int(j)

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_transitions(np.ndarray[np.int32_t,ndim=1] anc_counts_np,
                         np.ndarray[np.int32_t,ndim=1] gens_np,
                         int N,
                         double gen_mut_rate):
    cdef int [:] anc_counts = anc_counts_np
    cdef int [:] gens = gens_np
    cdef int i, nreps

    nreps = anc_counts_np.shape[0]

    cdef np.ndarray[np.int32_t,ndim=1] ret_counts_np = np.zeros(nreps,
            dtype = np.int32)
    cdef int [:] ret_counts = ret_counts_np

    for i in range(nreps):
        ret_counts[i] = simulate_wright_fisher_transition(N, anc_counts[i],
                gens[i], gen_mut_rate)

    return ret_counts_np


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def bin_sample_freqs(np.ndarray[np.int32_t,ndim=1] results_np,
                     int N,
                     np.ndarray[np.int32_t,ndim=1] coverages_np):

    cdef int [:] results = results_np
    cdef int [:] coverages = coverages_np
    cdef int i, nreps
    cdef double p

    nreps = results_np.shape[0]

    cdef np.ndarray[np.int32_t,ndim=1] ret_results_np = np.empty(
            results_np.shape[0], dtype = np.int32)
    cdef int [:] ret_results = ret_results_np

    for i in range(nreps):
        p = <double>results[i] / N
        ret_results[i] = <int>binom_rv(p, coverages[i])

    return ret_results_np


@cython.cdivision(True)
def simulate_bottlenecks(np.ndarray[np.int32_t,ndim=1] anc_counts_np,
        int N0, int Nb, double mu):
    '''
    Simulate a bottleneck, followed by doubling back to the original population
    size. If the original population size is not N0*2**k for some k, the final
    generation is a final expansion (not quite doubling) up to the original
    population size.

    N0    original haploid population size
    f0    frequency in the pre-bottleneck population
    Nb    haploid bottleneck size
    mu    per-generation mutation rate

    '''

    cdef np.ndarray[np.int32_t,ndim=1] ret_counts
    cdef int N, idx
    cdef double i, f, fp
    cdef int [:] anc_counts = anc_counts_np

    ret_counts = np.zeros(anc_counts_np.shape[0], dtype = np.int32)

    for idx in range(anc_counts_np.shape[0]):
        N = Nb
        f0 = anc_counts[idx] / float(N0)
        f0 = (1-mu)*f0 + mu*(1-f0)
        i = binom_rv(f0,N)
        while N <= N0/2:
            f = float(i)/N
            f = (1-mu)*f + mu*(1-f)
            i = binom_rv(f, 2*N)
            N *= 2

        if N < N0:
            f = float(i)/N
            f = (1-mu)*f + mu*(1-f)
            i = binom_rv(f, N0)

        ret_counts[idx] = <int>i

    return ret_counts

import numpy as np
cimport numpy as np
cimport cython
from numpy.math cimport INFINITY
from libc.math cimport log

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_binomial_pdf(const unsigned int k, const double p,
            const unsigned int n)

cdef extern from "gsl/gsl_sf_gamma.h":
    double gsl_sf_lnchoose(unsigned int n, unsigned int m)

cdef extern from "gsl/gsl_sys.h":
    double gsl_log1p (const double x) 

cdef extern from "math.h":
    bint isnan(double x)

cpdef double binom_pmf(double k, double n, double p):
    cdef unsigned int uk = <unsigned int>k
    cdef unsigned int un = <unsigned int>n
    return gsl_ran_binomial_pdf(uk, p, un)

cdef double lnchoose(int n, int m):
    cdef unsigned int un = <unsigned int>n
    cdef unsigned int um = <unsigned int>m
    return gsl_sf_lnchoose(un,um)

cdef double log1p(double x):
    return gsl_log1p(x)

cpdef log_binom_pmf(int k, double p, int n):
    cdef double lp, lchoose
    if k > n:
        return -1*INFINITY
    else:
        if p == 0:
            lp = 0 if k == 0 else -1*INFINITY
        elif p == 1:
            lp = 0 if k == n else -1*INFINITY
        else:
            lchoose = lnchoose(n, k)
            lp = lchoose + k*log(p) + (n-k) * log1p(-p)
        return lp

@cython.wraparound(False)
@cython.boundscheck(False)
def get_binom_likelihoods_cython(
        np.ndarray[np.float64_t, ndim=2] Xs,
        np.ndarray[np.float64_t, ndim=2] n,
        np.ndarray[np.float64_t, ndim=1] freqs,
        double min_phred_score = -1):
    '''
    binomial sampling likelihoods for the leaves.

    Xs     counts of the focal allele at each locus, length k, where k is
           number of loci
    n      sample sizes, same shape as Xs
    freqs  true allele frequencies to consider, [numpy array, shape (l,)]

    returns a k x l numpy array P of binomial sampling probabilities, where
    P[i,j] is the sampling probability at locus i given true frequency j
    '''
    cdef double p, perr
    cdef double [:,:] Xs_c = Xs
    cdef double [:,:] n_c = n
    cdef double [:] freqs_c = freqs
    cdef int num_loci = Xs.shape[0]
    cdef int num_samples = Xs.shape[1]
    cdef int num_freqs = freqs.shape[0]
    likes = np.zeros((num_loci, num_samples, num_freqs))
    cdef double [:,:,:] likes_arr = likes
    cdef int i, j, k

    if min_phred_score != -1.0:
        perr = 10.0**(-1.0*min_phred_score/10.0)
    else:
        perr = 0

    for i in range(num_loci):
        for j in range(num_samples):
            if isnan(Xs[i,j]):
                # missing data
                for k in range(num_freqs):
                    likes_arr[i,j,k] = 1.0
            else:
                for k in range(num_freqs):
                    # assume that each base is equally likely...
                    p = (1.0-perr)*freqs_c[k] + (perr/3.0)*(1.0-freqs_c[k])
                    likes_arr[i,j,k] = binom_pmf(Xs_c[i,j], n_c[i,j], p)
    return likes

##########################################
# nearest neighbors likelihoods
##########################################

cdef bint is_close_enough(double x, double y, double atol):
    if (max(x,y)-min(x,y)) <= atol:
        return 1
    else:
        return 0

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef bint get_lower_index(double [:] freqs, double f, int nfreqs,
        int* retval):
    '''
    returns 1 on exact match
            0 on non-match
    '''
    cdef int L, R, m
    L = 0
    R = nfreqs-1
    while True:
        if L > R:
            retval[0] = L
            return 0
        m = (L + R)/2
        if is_close_enough(freqs[m], f, 1e-8):
            retval[0] = m
            return 1
        if freqs[m] < f:
            L = m+1
        elif freqs[m] > f:
            R = m-1


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.cdivision(True)
cdef np.ndarray[np.float64_t,ndim=1] get_neighbor_likes(
        double x,
        np.ndarray[np.float64_t,ndim=1] freqs):
    cdef int idx, nfreqs
    cdef double nearest_low, nearest_high, frac_high
    cdef bint success
    cdef double [:] freqs_c
    cdef double [:] ret_c

    freqs_c = freqs
    nfreqs = freqs.shape[0]
    ret = np.zeros(nfreqs, dtype = np.float64)
    ret_c = ret
    success = get_lower_index(freqs_c, x, nfreqs, &idx)
    if success:
        ret_c[idx] = 1.0
    else:
        nearest_low = freqs_c[idx-1]
        nearest_high = freqs[idx]
        frac_high = (x-nearest_low)/(nearest_high-nearest_low)
        ret_c[idx-1] = 1-frac_high
        ret_c[idx] = frac_high
    
    return ret

#@cython.boundscheck(False)
#@cython.wraparound(False)
def get_nearest_likelihoods_cython(
        np.ndarray[np.float64_t,ndim=2] heteroplasmy_freqs,
        np.ndarray[np.float64_t,ndim=1] transition_freqs):
    '''
    nearest neighbor weighted likelihoods for the leaves, from called allele
    frequency likelihoods given by the original paper

    heterplasmy_freqs      counts of the focal allele at each locus, length k,
                           where k is number of loci [numpy array, shape
                           (num_loci, num_samples)]
    transition_freqs       true allele frequencies to consider, [numpy array,
                           shape (l,)]

    returns a k x n, a numpy array P of sampling probabilities, where P[i,j,l]
    is the sampling probability at locus i, sample j, true frequency l
    '''
    cdef double [:,:] hets_c = heteroplasmy_freqs
    cdef double [:] trans_c = transition_freqs
    cdef int num_loci = heteroplasmy_freqs.shape[0]
    cdef int num_samples = heteroplasmy_freqs.shape[1]
    cdef int num_trans_freqs = transition_freqs.shape[0]
    likes = np.zeros((num_loci, num_samples, num_trans_freqs))
    cdef double [:,:,:] likes_arr = likes
    cdef int i, j, k

    for i in range(num_loci):
        for j in range(num_samples):
            if isnan(heteroplasmy_freqs[i,j]):
                for k in range(num_trans_freqs):
                    likes_arr[i,j,k] = 1.0
            else:
                likes[i,j,:] = get_neighbor_likes(hets_c[i,j],transition_freqs)
    return likes

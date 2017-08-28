from __future__ import division
cimport numpy as np
import numpy as np
from _binom cimport binom_pmf

cpdef np.ndarray[np.float64_t, ndim=2] get_transition_matrix_var_N(
        int N1,
        int N2,
        int N0,
        double mu):
    '''
    get a transition matrix from population size N1 to N2


    N1   ancestor population size
    N2   present population size
    N0   some initial population size, such that the size of the returned
         matrix is (N0+1) x (N0+1). must be at least max(N1, N2)
    mu   symmetric per-generation mutation probability
    '''

    cdef np.ndarray[np.float64_t,ndim=2] P_np = np.zeros((N0+1,N0+1))
    cdef double [:,:] P = P_np
    cdef int i, j

    domain_size = N2+1
    x = np.arange(N2+1)
    for i in range(N1+1):
        f = i/N1
        f = (1-mu)*f + (1-f)*mu
        for j in range(N2+1):
            P[i,j] = binom_pmf(j,N2,f)
    return P_np

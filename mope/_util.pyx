from __future__ import print_function
cimport numpy as np
cimport cython
from libc.stdio cimport printf

def print_csv_line(np.ndarray[np.float64_t, ndim = 1] arr):
    cdef int i
    cdef int arrlen = arr.shape[0]
    for i in range(arrlen-1):
        print(arr[i], "\t", sep = '', end='') 
    print(arr[arrlen-1])

def print_csv_lines(np.ndarray[np.float64_t, ndim = 2] arr,
        np.ndarray[np.float64_t,ndim=1] lnprobs):
    cdef int i, j, npos, ncols
    npos = arr.shape[0]
    ncols = arr.shape[1]
    for i in range(npos):
        print(lnprobs[i], "\t", sep = '', end = '')
        for j in range(ncols-1):
            print(arr[i,j], "\t", sep = '', end='') 
        print(arr[i,ncols-1])


def translate_positions(
        np.ndarray[np.float64_t,ndim=2] pos,
        np.ndarray[np.float64_t,ndim=1] lower,
        int num_varnames):
    cdef:
        int i, j
        int num_walkers = pos.shape[0]
        np.ndarray[np.float64_t,ndim=2] ret = pos.copy()

    for i in range(num_walkers):
        # The first num_varnames parameters are genetic
        # drift parameters, which are zero when these
        # values are less than lower+1.
        for j in range(num_varnames):
            if pos[i,j] <= lower[j]+1:
                ret[i,j] = 0
            else:
                ret[i,j] = 10**pos[i,j]
        # The next num_varnames parameters are mutation
        # rates, which are also in log10-space.
        for j in range(num_varnames, 2*num_varnames):
            ret[i,j] = 10**pos[i,j]
        # We leave the other parameters (root parameters,
        # DFE params, selection parameters) in their
        # original space.
    return ret


def translate_positions_parallel(
        np.ndarray[np.float64_t,ndim=3] pos,
        np.ndarray[np.float64_t,ndim=1] lower,
        int num_branches):
    cdef:
        int i, j, k, nchains, nwalkers
        np.ndarray[np.float64_t,ndim=3] ret = pos.copy()

    nchains = pos.shape[0]
    nwalkers = pos.shape[1]
    for i in range(nchains):
        for j in range(nwalkers):
            for k in range(num_branches):
                if pos[i,j,k] <= lower[k]+1:
                    ret[i,j,k] = 0
                else:
                    ret[i,j,k] = 10**pos[i,j,k]
            # The next num_branches parameters are mutation
            # rates, which are also in log10-space.
            for k in range(num_branches, 2*num_branches):
                ret[i,j,k] = 10**pos[i,j,k]
            # We leave the other parameters (root parameters,
            # DFE params, selection parameters) in their
            # original space.
    return ret


def print_parallel_csv_lines(np.ndarray[np.float64_t,ndim=3] pos_np,
        np.ndarray[np.float64_t,ndim=2] lnprobs_np,
        np.ndarray[np.float64_t,ndim=2] lnlikes_np):
    cdef int i, j, k, nchains, nwalkers, ndim
    cdef double [:,:,:] pos = pos_np
    cdef double [:,:] lnprobs = lnprobs_np
    cdef double [:,:] lnlikes = lnlikes_np
    nchains = pos.shape[0]
    nwalkers = pos.shape[1]
    ndim = pos.shape[2]
    for i in range(nchains):
        for j in range(nwalkers):
            printf("%i\t%.11f\t%.11f\t", i, lnprobs[i,j], lnlikes[i,j])
            for k in range(ndim-1):
                printf("%.11e\t", pos[i,j,k])
            printf("%.11e\n", pos[i,j,ndim-1])



def print_sims(np.ndarray[np.float64_t,ndim=1] ages,
        np.ndarray[np.int32_t,ndim=2] results,
        sample_size,
        frequencies):

    cdef int i, b, nreps, ncols

    b = 0
    if sample_size:
        b = sample_size

    nreps = results.shape[0]
    ncols = results.shape[1]
    if not frequencies:
        #leaves, coverage, age
        for i in range(nreps):
            for j in range(ncols):
                print(results[i,j], "\t", sep = '', end = '')
            print(sample_size, "\t", ages[i], sep = '')

    else:
        for i in range(nreps):
            for j in range(ncols):
                print((<double>results[i,j]) / sample_size, "\t", sep ='',
                        end = '')
            print(ages[i])


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

def print_parallel_csv_lines(np.ndarray[np.float64_t,ndim=3] pos_np,
        np.ndarray[np.float64_t,ndim=2] lnprobs_np):
    cdef int i, j, k, nchains, nwalkers, ndim
    cdef double [:,:,:] pos = pos_np
    cdef double [:,:] lnprobs = lnprobs_np
    nchains = pos.shape[0]
    nwalkers = pos.shape[1]
    ndim = pos.shape[2]
    for i in range(nchains):
        printf("%i\t", i)
        for j in range(nwalkers):
            printf("%.11f\t", lnprobs[i,j])
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


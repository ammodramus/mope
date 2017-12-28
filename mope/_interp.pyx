import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def bilinear_interpolate(
        np.ndarray[np.float64_t, ndim = 2] fQ11,
        np.ndarray[np.float64_t, ndim = 2] fQ12,
        np.ndarray[np.float64_t, ndim = 2] fQ21,
        np.ndarray[np.float64_t, ndim = 2] fQ22,
        double denom,
        double weight11,
        double weight12,
        double weight21,
        double weight22):

    cdef int matdim = fQ11.shape[0]
    cdef int i, j
    cdef unsigned int ui, uj

    cdef np.ndarray[np.float64_t, ndim = 2] distn = np.zeros((matdim, matdim), order = 'F')
    cdef double [:,:] distn_c = distn

    cdef double [:,:] fQ11_c = fQ11
    cdef double [:,:] fQ12_c = fQ12
    cdef double [:,:] fQ21_c = fQ21
    cdef double [:,:] fQ22_c = fQ22

    for j in range(matdim):
        for i in range(matdim):
            ui = <unsigned int>i
            uj = <unsigned int>j
            distn_c[ui,uj] = (fQ11_c[ui,uj]*weight11 + fQ21_c[ui,uj]*weight21 +
                    fQ12_c[ui,uj]*weight12 + fQ22_c[ui,uj]*weight22)

    return distn

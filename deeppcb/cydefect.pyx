import numpy as np
cimport numpy as np
cimport cython

cdef calc_edge_dist(np.ndarray[np.int_t, ndim=2] pts0, np.ndarray[np.int_t, ndim=2] pts1):
    cdef float min_d = float('inf')
    cdef float d = 0

    for x0, y0 in pts0:
        for x1, y1 in pts1:
            d = (x0 - x1)**2 + (y0 - y1)**2
            if d < min_d:
                min_d = d

    return min_d ** 0.5

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_edge_distmap(list edge_list, np.ndarray[np.int_t, ndim=2] nn_idxs):
    cdef int N = len(edge_list)
    cdef np.ndarray[np.int_t, ndim=2] epi
    cdef np.ndarray[np.int_t, ndim=2] epj

    out = 10000 * np.ones((N, N))
    for i, epi in enumerate(edge_list):
        for j in nn_idxs[i][1:]:
            epj = edge_list[j]
            out[i, j] = calc_edge_dist(epi, epj)

    return out

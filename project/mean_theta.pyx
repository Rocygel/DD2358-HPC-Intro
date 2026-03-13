# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, atan2

def find_mean_theta(double[:] x, double[:] y, double[:] theta, int N, int R):
    cdef double[:] mean_theta = np.empty(N, dtype=np.float64)
    cdef double r_sq = R * R
    cdef double sx, sy, dist_sq
    cdef unsigned int i, j

    for i in range(N):
        sx, sy = 0.0, 0.0
        for j in range(N):
            dist_sq = (x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j])
            
            if dist_sq < r_sq:
                sx += cos(theta[j])
                sy += sin(theta[j])
            
        mean_theta[i] = atan2(sy, sx)

    return mean_theta



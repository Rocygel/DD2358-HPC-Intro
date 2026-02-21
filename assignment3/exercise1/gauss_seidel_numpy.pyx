## gauss_seidel_parallel.pyx
import numpy as np
cimport numpy as np
from numpy import roll

def gauss_seidel(float[:,:] f):
    """Gauss-Seidel formula implemented using NumPy"""
#    cdef unsigned int i, j
#    for i in range(1,len(f)-1):
#        for j in range(1,len(f)-1):
#            f[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] +
#                                  f[i+1][j] + f[i-1][j])
    f = 0.25 * (roll(f, 1, axis=1)
                + roll(f, -1, axis=1)
                + roll(f, 1, axis=0)
                + roll(f, -1, axis=0))
    return f

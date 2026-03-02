## gauss_seidel_cupy.pyx
#import numpy as np
import cupy as cp
cimport cupy as cp
from cupy import roll
#from numpy import roll

def gauss_seidel(f):
    """Gauss-Seidel formula implemented using CuPy"""
#    cdef unsigned int i, j
#    for i in range(1,len(f)-1):
#        for j in range(1,len(f)-1):
#            f[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] +
#                                  f[i+1][j] + f[i-1][j])
    f = 0.25 * (roll(f, 1, 1)
                + roll(f, -1, 1)
                + roll(f, 1, 0)
                + roll(f, -1, 0))
    return f

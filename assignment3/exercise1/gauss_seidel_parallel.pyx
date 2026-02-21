## gauss_seidel_parallel.pyx
import torch
import pytorch_ocl
from torch import roll

def gauss_seidel(f):
    """Gauss-Seidel formula implemented using NumPy"""
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

## gauss_seidel.pyx
#cython: boundscheck=False, wraparound=False
def gauss_seidel(f):
    """Gauss-seidel using list based on lab-PM code with set iterations"""
    cdef unsigned int i, j
    for i in range(1,len(f)-1):
        for j in range(1,len(f)-1):
            f[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] +
                                  f[i+1][j] + f[i-1][j])
    return f

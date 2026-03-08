## mandelbrot_vectorized.pyx
import numpy as np
cimport numpy as np

# def mandelbrot(double complex c, unsigned int max_iter=100):
#     """
#     Computes the number of iterations before divergence.
#     Taken from the course Canvas page.
#     """
#     cdef unsigned int n
#     cdef double complex z
#     z = 0
#     for n in range(max_iter):
#         if abs(z) > 2:
#             return n
#         z = z*z + c
#     return max_iter

def mandelbrot_set(unsigned int width, unsigned int height, int x_min, int x_max, int y_min, int y_max, unsigned int max_iter=100):
    """
    Computes a heat map of the Mandelbrot set.
    Vectorized version.
    """
    cdef float[:] x_vals = np.linspace(x_min, x_max, width, dtype=np.float32)
    cdef float[:] y_vals = np.linspace(y_min, y_max, height, dtype=np.float32)
    
    gridtuple = np.meshgrid(x_vals, y_vals)
    cdef float[:,:] x_vals_grid = gridtuple[0]
    cdef float[:,:] y_vals_grid = gridtuple[1]
    cdef float complex[:,:] cmplx_grid = np.add(x_vals_grid, np.multiply(y_vals_grid, 1j))
    cdef float complex[:,:] result_grid = np.zeros((height, width), dtype=np.complex64)
    cdef int[:,:] image = np.zeros((height, width), dtype=np.int32)

    for i in range(max_iter):
        result_grid = np.where(np.abs(result_grid) < 2, np.add(np.pow(result_grid, 2), cmplx_grid), result_grid)
        image = np.where(np.abs(result_grid) < 2, np.add(image, 1), image)
    
    return image


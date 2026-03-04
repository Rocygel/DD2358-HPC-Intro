## mandelbrot.pyx
import numpy as np
cimport numpy as np

def mandelbrot(double complex c, unsigned int max_iter=100):
    """
    Computes the number of iterations before divergence.
    Taken from the course Canvas page.
    """
    cdef unsigned int n
    cdef double complex z
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(unsigned int width, unsigned int height, int x_min, int x_max, int y_min, int y_max, unsigned int max_iter=100):
    """
    Generates the Mandelbrot set image.
    Taken from the course Canvas page.
    """
    cdef unsigned int i
    cdef unsigned int j
    cdef double complex c

    cdef float[:] x_vals = np.linspace(x_min, x_max, width, dtype=np.float32)
    cdef float[:] y_vals = np.linspace(y_min, y_max, height, dtype=np.float32)
    x_vals, y_vals = np.meshgrid(x_vals, y_vals)
    cdef float complex[:,:] imag_plane = x_vals + y_vals * 1j
    #cdef int[:,:] image = np.zeros((height, width), dtype=np.int32)

#    for i in range(height):
#        for j in range(width):
#            c = complex(x_vals[j], y_vals[i])
#            image[i, j] = mandelbrot(c, max_iter)
    for i in range(max_iter):
        imag_plane = imag_plane 

    return image


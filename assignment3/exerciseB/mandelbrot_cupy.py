## mandelbrot_cupy.py
import cupy as cp

def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100):
    """
    Computes a heat map of the Mandelbrot set.
    Vectorized version.
    """
    x_vals = cp.linspace(x_min, x_max, width, dtype=cp.float32)
    y_vals = cp.linspace(y_min, y_max, height, dtype=cp.float32)
    
    gridtuple = cp.meshgrid(x_vals, y_vals)
    x_vals_grid = gridtuple[0]
    y_vals_grid = gridtuple[1]
    cmplx_grid = cp.add(x_vals_grid, cp.multiply(y_vals_grid, 1j))
    result_grid = cp.zeros((height, width), dtype=cp.complex64)
    image = cp.zeros((height, width), dtype=cp.int32)

    for i in range(max_iter):
        result_grid = cp.where(cp.abs(result_grid) < 2, cp.add(cp.pow(result_grid, 2), cmplx_grid), result_grid)
        image = cp.where(cp.abs(result_grid) < 2, cp.add(image, 1), image)

    image = cp.ndarray.get(image)
    
    return image


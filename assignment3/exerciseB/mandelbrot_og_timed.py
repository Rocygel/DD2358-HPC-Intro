import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import wraps

# decorator to time
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time

def mandelbrot(c, max_iter=100):
    """Computes the number of iterations before divergence."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

@timefn
def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100):
    """Generates the Mandelbrot set image."""
    x_vals = np.linspace(x_min, x_max, width)
    y_vals = np.linspace(y_min, y_max, height)
    image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            c = complex(x_vals[j], y_vals[i])
            image[i, j] = mandelbrot(c, max_iter)

    return image

def main():
    # Parameters
    width, height = 1000, 800
    x_min, x_max, y_min, y_max = -2, 1, -1, 1

    # Generate fractal
    image = mandelbrot_set(width, height, x_min, x_max, y_min, y_max)

    # Display
    plt.imshow(image, cmap='inferno', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar()
    plt.title("Mandelbrot Set")
    plt.show()

if __name__ == "__main__":
    main()


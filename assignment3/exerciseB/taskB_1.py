## taskB_1.py
import numpy as np
import matplotlib.pyplot as plt
from mandelbrot import mandelbrot_set

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


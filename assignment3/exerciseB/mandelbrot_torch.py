## mandelbrot_torch.py
import torch

def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100):
    """
    Computes a heat map of the Mandelbrot set.
    PyTorch version.
    """

    x_vals = torch.linspace(x_min, x_max, width, dtype=torch.float32)
    y_vals = torch.linspace(y_min, y_max, height, dtype=torch.float32)
    
    gridtuple = torch.meshgrid(y_vals, x_vals)
    x_vals_grid = gridtuple[1]
    y_vals_grid = gridtuple[0]
    cmplx_grid = torch.add(x_vals_grid, torch.multiply(y_vals_grid, 1j)).cuda()
    result_grid = torch.zeros((height, width), dtype=torch.complex64).cuda()
    image = torch.zeros((height, width), dtype=torch.int32).cuda()

    for i in range(max_iter):
        result_grid = torch.where(torch.abs(result_grid) < 2, torch.add(torch.pow(result_grid, 2), cmplx_grid), result_grid)
        image = torch.where(torch.abs(result_grid) < 2, torch.add(image, 1), image)

    image = torch.Tensor.cpu(image)
    
    return image


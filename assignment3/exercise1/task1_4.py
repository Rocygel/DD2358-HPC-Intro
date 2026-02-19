from timeit import default_timer as timer
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
from gauss_seidel import gauss_seidel # Our cythonized function

def main():
    # instantiate 
    N = [4, 8, 16, 32, 64] # grid sizes
    rng = np.random.default_rng()
    
    gs_iterations = 1000
    run_iterations = 10
    total_time = []

    for n in N:
        x = rng.random((n, n))

        # set grid values at boundaries to zero
        for j in range(n):
            x[j][0] = 0
            x[0][j] = 0
            x[j][n-1] = 0
            x[n-1][j] = 0

        t0 = timer()
        for run in range (run_iterations):
            for i in range(gs_iterations):
                x = gauss_seidel(x)
        t1 = timer()

        total_time.append((t1 - t0) / run_iterations)

    print(total_time)
    plt.plot(N, total_time)
    plt.title(f"Execution time for varying grid size ({gs_iterations} iterations)")
    plt.xlabel("Grid lengths (N for N x N grid)")
    plt.ylabel("Computation time (s)")
    plt.show()


if __name__ == "__main__":
    main()

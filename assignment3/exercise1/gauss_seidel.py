
from timeit import default_timer as timer
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt


# decorator to time
#def timefn(fn):
#    @wraps(fn)
#    def measure_time(*args, **kwargs):
#        t1 = timer()
#        result = fn(*args, **kwargs)
#        t2 = timer()
#        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
#        return result
#    return measure_time


## from course site
#@timefn
@profile                    # for 1.2
def gauss_seidel(f):
    """Gauss-seidel using list based on lab-PM code with set iterations"""
    #for x in range(iterations):
    for i in range(1,len(f)-1):
        for j in range(1,len(f)-1):
            #f[i,j] = 0.25 * (f[i,j+1] + f[i,j-1] +
            #                       f[i+1,j] + f[i-1,j])
            f[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] +
                                  f[i+1][j] + f[i-1][j])
    return f


def main():
    # instantiate 
    #N = [4, 8, 16, 32, 64]                 # grid sizes
    N = [16]
    rng = np.random.default_rng()
    
    gs_iterations = 1000
    run_iterations = 1
    #total_time = []

    for n in N:
        x = rng.random((n, n))

        # set grid values at boundaries to zero
        for j in range(n):
            x[j][0] = 0
            x[0][j] = 0
            x[j][n-1] = 0
            x[n-1][j] = 0

        # run 1000 iterations.
        # loop inside instead of outside to time with decorator
        #x = gauss_seidel(x, iterations)

        #t0 = timer()
        for run in range (run_iterations):
            for i in range(gs_iterations):
                x = gauss_seidel(x)
        #t1 = timer()

        #total_time.append((t1 - t0) / run_iterations)

    #print(total_time)
    #plt.plot(N, total_time)
    #plt.title(f"Execution time for varying grid size ({gs_iterations} iterations)")
    #plt.xlabel("Grid lengths (N for N x N grid)")
    #plt.ylabel("Computation time (s)")
    #plt.show()


if __name__ == "__main__":
    main()
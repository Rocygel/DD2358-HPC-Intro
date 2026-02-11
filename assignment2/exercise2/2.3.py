import array as arr
import numpy as np
from time import time as timer
from functools import wraps


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        times = []
        iterations = 10
        for _ in range(iterations):
            t1 = timer()
            result = fn(*args, **kwargs)
            t2 = timer()
            times.append(t2-t1)
        avg_t = sum(times) / iterations
        min_t, max_t = min(times), max(times)
        # calculate standard deviation:
        d = 0
        for i in range(iterations):
            d += np.power(times[i] - avg_t, 2)
        #d = np.power(d, 2)
        std_d = np.sqrt(d/iterations)    

        print(f"\nAverage of {iterations} runs for {fn.__name__}: {avg_t:.10f} seconds")
        print(f"\nMinimum of {iterations} runs for {fn.__name__}: {min_t:.10f} seconds")
        print(f"\nMaximum of {iterations} runs for {fn.__name__}: {max_t:.10f} seconds")
        print(f"\nStd. d. of {iterations} runs for {fn.__name__}: {std_d:.10f} seconds")
        return result
    return measure_time


# dgemm with list
@timefn
def dgemm_list(a, b, c):
    for i in range(len(a)):
        for j in range(len(b)):
            for k in range(len(c)):
                c[i][j] += a[i][k] * b[k][j]
    return c


@timefn
# dgemm with array
def dgemm_array(a, b, c, n):
    for i in range(n):
        i_n = i*n
        for j in range(n):
            c_index = i_n + j
            for k in range(n):
                c[c_index] += a[i_n + k] * b[k * n + j]
    return c


# dgemm with numpy(?) unsure if this is intended data structure
@timefn
def dgemm_numpy(a, b, c):
    for i in range(len(a)):
        for j in range(len(b)):
            for k in range(len(c)):
                c[i,j] += a[i,k] * b[k,j]
    return c


# for 2.5
@timefn
def dgemm_matmul(a, b, c):
    return (a @ b) + c


### for 2.3: large random matrix means its not
### reproduceable but easiest way to make large
### data set. to be fair there are probably 
### large datasets to be found online.

if __name__ == "__main__":
    SIZES = [50, 100, 200] # included 400 800 for matmul in 2.5
    
    for size in SIZES:
        print(f"size {size} x {size}")

        rng = np.random.default_rng()

        a_np = rng.random((size, size))
        b_np = rng.random((size, size))
        c_np = rng.random((size, size))
        
        #a_list = a_np.tolist()
        #b_list = b_np.tolist()
        #c_list = c_np.tolist()
        
        #a_arr = arr.array('d', a_np.flatten())
        #b_arr = arr.array('d', b_np.flatten())
        #c_arr = arr.array('d', c_np.flatten())

        #print(f" ----- list --------")
        #dgemm_list(a_list, b_list, c_list)

        #print(f" ------- array -------")
        #dgemm_array(a_arr, b_arr, c_arr, size)

        #print(f" ------ numpy -------")
        dgemm_numpy(a_np, b_np, c_np.copy())

        # for 2.5
        #print(f"------ for 2.5 --------")
        #dgemm_matmul(a_np, b_np, c_np.copy())

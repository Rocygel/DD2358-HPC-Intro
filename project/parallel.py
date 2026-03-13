#import matplotlib.pyplot as plt
from multiprocessing import shared_memory, Barrier, Process

"""
Create Your Own Active Matter Simulation (With Python)
Philip Mocz (2021) Princeton University, @PMocz

Simulate Viscek model for flocking birds

Parallel implementation using multiprocessing.

KTH course DD2358 by Group 10 (eaeklof@kth.se , rogerche@kth.se)
"""

def vicsek_worker(x, y, mean_theta, theta, eta, vx, vy, dt, Nt, L, N, R, v0, start, end, barrier):
    # Simulation Main Loop
    for i in range(Nt):
        # move
        x[start:end] += vx[start:end] * dt
        y[start:end] += vy[start:end] * dt

        # apply periodic BCs
        x[start:end] = x[start:end] % L
        y[start:end] = y[start:end] % L

        # sync to avoid race condition between processes on x, y
        barrier.wait()

        # find mean angle of neighbors within R
        mean_theta[start:end] = theta[start:end]
        for b in range(start, end):
            neighbors = (x - x[b]) ** 2 + (y - y[b]) ** 2 < R**2
            sx = np.sum(np.cos(theta[neighbors]))
            sy = np.sum(np.sin(theta[neighbors]))
            mean_theta[b] = np.arctan2(sy, sx)

        # sync to avoid race condition between processes on theta
        barrier.wait()

        # add random perturbations
        theta[start:end] = mean_theta[start:end] + eta * (np.random.rand(end - start, 1) - 0.5)

        # update velocities
        vx[start:end] = v0 * np.cos(theta[start:end])
        vy[start:end] = v0 * np.sin(theta[start:end])

def main():
    """Finite Volume simulation"""

    # Simulation parameters
    v0 = 1.0  # velocity
    eta = 0.5  # random fluctuation in angle (in radians)
    L = 10  # size of box
    R = 1  # interaction radius
    dt = 0.2  # time step
    Nt = 200  # number of time steps
    N = 500  # number of birds
    dtype = np.float64
    bufsize = N * np.dtype(dtype).itemsize
    num_processes = 1
    bar = Barrier(num_processes)

    # Create shared memory objects
    x_shm = shared_memory.SharedMemory(create=True, size=bufsize)
    x = np.ndarray((N,1), dtype=dtype, buffer=x_shm.buf)

    y_shm = shared_memory.SharedMemory(create=True, size=bufsize)
    y = np.ndarray((N,1), dtype=dtype, buffer=y_shm.buf)

    theta_shm = shared_memory.SharedMemory(create=True, size=bufsize)
    theta = np.ndarray((N,1), dtype=dtype, buffer=theta_shm.buf)

    vx_shm = shared_memory.SharedMemory(create=True, size=bufsize)
    vx = np.ndarray((N,1), dtype=dtype, buffer=vx_shm.buf)

    vy_shm = shared_memory.SharedMemory(create=True, size=bufsize)
    vy = np.ndarray((N,1), dtype=dtype, buffer=vy_shm.buf)

    mean_theta_shm = shared_memory.SharedMemory(create=True, size=bufsize)
    mean_theta = np.ndarray((N,1), dtype=dtype, buffer=mean_theta_shm.buf)

    # Initialize
    np.random.seed(17)  # set the random number generator seed

    # bird positions
    x[:] = np.random.rand(N, 1) * L
    y[:] = np.random.rand(N, 1) * L

    # bird velocities
    theta[:] = np.random.rand(N, 1) * 2 * np.pi
    vx[:] = v0 * np.cos(theta)
    vy[:] = v0 * np.sin(theta)

    list_args = []
    batch_size = N // num_processes
    for i in range(num_processes):
        list_args.append((x[:], y[:], mean_theta[:], theta[:], eta, vx[:], vy[:], dt, Nt, L, N, R, v0, (batch_size * i), (batch_size * (i+1)), bar))

    procs = []
    for pid in range(num_processes):
        p = Process(target=vicsek_worker, args=list_args[pid])
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    return 0


if __name__ == "__main__":
    main()

%lprun -f main main()
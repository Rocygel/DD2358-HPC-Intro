import numpy as np
from timeit import default_timer as timer
from mean_theta import find_mean_theta

"""
Create Your Own Active Matter Simulation (With Python)
Philip Mocz (2021) Princeton University, @PMocz

Simulate Viscek model for flocking birds

"""

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

    # Initialize
    np.random.seed(17)  # set the random number generator seed

    # bird positions
    x = np.random.rand(N) * L
    y = np.random.rand(N) * L

    # bird velocities
    theta = 2 * np.pi * np.random.rand(N)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    t0 = timer()
    # Simulation Main Loop
    for i in range(Nt):
        # move
        x += vx * dt
        y += vy * dt

        # apply periodic BCs
        x = x % L
        y = y % L

        # find mean angle of neighbors within R
        mean_theta = find_mean_theta(x, y, theta, N, R)

        #for b in range(N):
        #    neighbors = (x - x[b]) ** 2 + (y - y[b]) ** 2 < R**2
        #    sx = np.sum(np.cos(theta[neighbors]))
        #    sy = np.sum(np.sin(theta[neighbors]))
        #    mean_theta[b] = np.arctan2(sy, sx)

        # add random perturbations
        theta = mean_theta + eta * (np.random.rand(N) - 0.5)

        # update velocities
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)

    t1 = timer()

    print(t1 - t0)

    return 0


if __name__ == "__main__":
    main()


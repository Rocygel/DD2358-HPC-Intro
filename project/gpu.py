"""
Create Your Own Active Matter Simulation (With Python)
Philip Mocz (2021) Princeton University, @PMocz

Simulate Viscek model for flocking birds

GPU-implementation
KTH course DD2358 by Group 10 (eaeklof@kth.se , rogerche@kth.se)

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
    cp.random.seed(17)  # set the random number generator seed

    # bird positions
    x = cp.random.rand(N, 1) * L
    y = cp.random.rand(N, 1) * L

    # bird velocities
    theta = 2 * cp.pi * cp.random.rand(N, 1)
    vx = v0 * cp.cos(theta)
    vy = v0 * cp.sin(theta)

    # Simulation Main Loop
    for i in range(Nt):
        # move
        x += vx * dt
        y += vy * dt

        # apply periodic BCs
        x = x % L
        y = y % L

        # find mean angle of neighbors within R
        # calculate all distances
        dx = x - x.T
        dy = y - y.T

        # birds within radius are neighbors
        neighbours = (dx**2 + dy**2) < R**2

        # matrix multiplication instead of for-loop summation
        sx = neighbours @ cp.cos(theta)
        sy = neighbours @ cp.sin(theta)

        # new mean angles
        mean_theta = cp.arctan2(sy, sx)

        # add random perturbations
        theta = mean_theta + eta * (cp.random.rand(N, 1) - 0.5)

        # update velocities
        vx = v0 * cp.cos(theta)
        vy = v0 * cp.sin(theta)

    return 0


if __name__ == "__main__":
    main()

%lprun -f main main()
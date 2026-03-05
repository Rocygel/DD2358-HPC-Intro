
import numpy as np
import pyvtk
import random
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 


def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning
    
    return forest, burn_time


def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors


def simulate_wildfire():
    """Simulates wildfire spread over time."""
    forest, burn_time = initialize_forest()
    
    fire_spread = []  # Track number of burning trees each day
    
    for day in range(DAYS):
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break
    
    return fire_spread


def save_to_vtk(filename, burn_time, forest):
    """
    Save the evolving simulation data as a VTK file for visualization in ParaView.
    """

    burn_time_flat = burn_time.T.flatten()
    forest_flat = forest.T.flatten()

    # Create VTK structure
    vtk_data = pyvtk.VtkData(
        pyvtk.StructuredPoints([GRID_SIZE, GRID_SIZE, 1]),
        pyvtk.PointData(
            pyvtk.Scalars(burn_time_flat, name="burn time"),
            pyvtk.Scalars(forest_flat, name="forest")
            #pyvtk.Scalars(rho_flat, name="density"),
            #pyvtk.Vectors(np.column_stack((vx_flat, vy_flat, np.zeros_like(vx_flat))), name="velocity"),
            #pyvtk.Scalars(P_flat, name="pressure")
        )
    )
    vtk_data.tofile(filename)
    print(f"Saved VTK file: {filename}")


def main():
    forest, burn_time = initialize_forest()
    fire_spread = []

    outputCount = 1
    t = 0
    # Simulation Main Loop
    while t < DAYS:
        new_forest = forest.copy()

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))

        # Update time
        t += 1
        
        # Save VTK at every day
        vtk_filename = f"frame_{outputCount:03d}.vtk"
        save_to_vtk(vtk_filename, burn_time, forest)
        outputCount += 1

        # Plot grid every 5 days
        if t % 20 == 0 or t == DAYS - 1:
            plt.figure(figsize=(6, 6))
            plt.imshow(forest, cmap='viridis', origin='upper')
            plt.title(f"Wildfire Spread - Day {t}")
            plt.colorbar(label="State: 0=Empty, 1=Tree, 2=Burning, 3=Ash")
            plt.show()
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break

    print("Simulation complete!")

if __name__ == "__main__":
    main()
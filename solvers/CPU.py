from config import diffusion_number
import time
import numpy as np

def CPU1D(config):
    """
    Explicit finite-difference solver (1D heat equation, CPU version).

    This function:
    - Builds the spatial grid
    - Applies initial + fixed temperature conditions
    - Performs explicit FD time marching
    - Stores full temperature history
    - Enforces Neumann BCs at ends of the rod
    - Returns final distribution, history array, and runtime
    """
    print(f"1D CPU solver running...")
    r = diffusion_number(config)     # Stability ratio α·dt/dx²

    dx = config['dx']
    x_size = config['x_size']
    T_start = config['T']

    # Ensure dx divides domain length so the grid aligns correctly
    valid_step_size = np.isclose(x_size / dx, round(x_size / dx))
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return

    # Build spatial grid and initial uniform temperature field
    X = np.arange(0, x_size+dx, dx)
    T_init = np.ones_like(X) * T_start

    # -------------------------------------------------------------
    # Insert fixed temperature points (from temp_deltas)
    # These are overwritten each iteration to enforce Dirichlet BCs
    # -------------------------------------------------------------
    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}                 # stores grid index → fixed temperature

    for i, temp in temp_deltas.items():
        # Coordinates must be inside rod
        if i > x_size or i < 0:
            print("Error: A position in temp_delta is out of bounds")
            return

        # Convert physical position to nearest grid index
        idx = int(round(i/dx))
        T_init[idx] = temp
        delta_index[idx] = temp

    # Working array
    T = T_init.copy()
    t_steps = int(config['t_steps'])

    # History list (one entry per timestep)
    T_hist = []
    T_hist.append(T)

    # -------------------------------------------------------------
    # MAIN TIME-STEPPING LOOP (explicit forward-Euler FD)
    # -------------------------------------------------------------
    start = time.perf_counter()
    for n in range(t_steps):

        # Copy to avoid in-place update while reading neighbours
        Tnew = T.copy()

        # Central-difference Laplacian update for interior nodes
        for i in range(1, len(X)-1):
            Tnew[i] = T[i] + r * (T[i+1] + T[i-1] - 2*T[i])

        # Neumann (zero-gradient) boundary conditions
        Tnew[0]   = Tnew[1]          # left boundary ∂T/∂x = 0
        Tnew[-1]  = Tnew[-2]         # right boundary ∂T/∂x = 0

        # Reapply fixed temperature points (Dirichlet)
        for i, temp in delta_index.items():
            Tnew[i] = temp

        # Store history
        T_hist.append(Tnew.copy())

        # Advance solution
        T = Tnew

    end = time.perf_counter()
    runtime = end - start

    return T, T_hist, runtime


def CPU2D(config):
    """
    Explicit finite-difference solver (2D heat equation, CPU version).

    This performs the same steps as CPU1D but extended to a 2D
    structured grid with (i,j) indexing.

    Steps:
    - Build 2D grid
    - Apply measurement / heat-source points from temp_deltas
    - Compute explicit FD update Tnew = T + r·Laplacian(T)
    - Apply Neumann BCs on all four edges
    - Reapply fixed temperature points (Dirichlet)
    - Store full history
    """
    print(f"2D CPU solver running...")
    r = diffusion_number(config)

    dx = config['dx']
    x_size = config['x_size']
    T_start = config['T']

    # Ensure grid alignment
    valid_step_size = np.isclose(x_size / dx, round(x_size / dx))
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return

    # Grid size (#points along one dimension)
    length = int(x_size / dx) + 1

    # Initial uniform plate temperature
    T_init = np.ones((length, length)) * T_start

    # -------------------------------------------------------------
    # Apply any fixed (Dirichlet) temperature points from config
    # Keys are given as (x,y) positions in metres → convert to indices
    # -------------------------------------------------------------
    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}

    for (i, j), temp in temp_deltas.items():

        # Ensure physical coordinates lie in domain
        if i > x_size or i < 0 or j > x_size or j < 0:
            print("Error: A position in temp_delta is out of bounds")
            return

        # Convert physical coords → nearest grid indices
        idx_x = int(round(i/dx))
        idx_y = int(round(j/dx))

        # Apply temperature directly to initial field
        T_init[(idx_y, idx_x)] = temp
        delta_index[(idx_y, idx_x)] = temp

    # Working copy
    T = T_init.copy()
    t_steps = int(config['t_steps'])

    # Store temperature history (each entry is a 2D array)
    T_hist = []
    T_hist.append(T)

    # -------------------------------------------------------------
    # MAIN TIME-STEPPING LOOP (explicit 2D forward-Euler FD)
    # -------------------------------------------------------------
    start = time.perf_counter()
    for n in range(t_steps):

        Tnew = T.copy()

        # Interior update: 5-point Laplacian stencil
        for i in range(1, length-1):
            for j in range(1, length-1):
                Tnew[i, j] = (
                    T[i, j]
                    + r * (
                        T[i+1, j] + T[i-1, j] +   # vertical neighbours
                        T[i, j+1] + T[i, j-1]     # horizontal neighbours
                        - 4*T[i, j]               # centre coefficient
                    )
                )

        # ---------------------------------------------------------
        # Apply Neumann boundaries (zero-gradient on all edges)
        # ---------------------------------------------------------
        Tnew[0, :]   = Tnew[1, :]       # top
        Tnew[-1, :]  = Tnew[-2, :]      # bottom
        Tnew[:, 0]   = Tnew[:, 1]       # left
        Tnew[:, -1]  = Tnew[:, -2]      # right

        # Reapply Dirichlet fixed temperature points
        for (y, x), temp in delta_index.items():
            Tnew[y][x] = temp

        # Log history
        T_hist.append(Tnew.copy())

        # Advance solution
        T = Tnew

    end = time.perf_counter()
    runtime = end - start

    return T, T_hist, runtime

from config import diffusion_number
import time
import cupy as cp

# =====================================================================
# GPU_CuPy1D / GPU_CuPy2D
# ---------------------------------------------------------------------
# These solvers are direct GPU-accelerated equivalents of CPU1D / CPU2D.
#
# Key differences:
#   • Uses CuPy instead of NumPy (arrays live on GPU)
#   • Finite-difference updates are vectorised rather than loop-based
#   • Memory copies avoided except for history recording / final output
#   • Logic, stability conditions, boundary conditions, and temperature
#     application are identical to the CPU versions.
#
# All core finite-difference physics stays the same — only the execution
# backend changes (NumPy → CuPy), which dramatically speeds up 1D & 2D.
# =====================================================================

def GPU_CuPy1D(config):
    print(f"1D GPU CuPy solver running...\n")
    r = diffusion_number(config)
    
    dx = config['dx']
    x_size = config['x_size']
    T_start = config['T']

    # Same dx divisibility check as CPU solver
    ratio = x_size / dx
    valid_step_size = float(ratio).is_integer()
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return
    
    # GPU array construction
    X = cp.arange(0, x_size+dx, dx)
    T_init = cp.ones_like(X) * T_start

    # Index fixed-temperature locations
    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}   
    for i, temp in temp_deltas.items():
        
        if i > x_size or i < 0:
            print("Error: A position in temp_delta is out of bounds")
            return

        # Same nearest-grid-index logic as CPU version
        idx = int(round(i/dx))
        T_init[idx] = temp
        delta_index[idx] = temp

    T = T_init.copy()
    t_steps = int(config['t_steps'])
    T_hist = []
    T_hist.append(T)

    # ================================
    # MAIN TIME LOOP (vectorised FD)
    # ================================
    start = time.perf_counter()
    for _ in range(t_steps):
        Tnew = T.copy()

        # Vectorised central-difference update over interior points
        Tnew[1:-1] = T[1:-1] + r * (T[2:] + T[:-2] - 2*T[1:-1])
        
        # Neumann BCs (same as CPU)
        Tnew[0]  = Tnew[1]
        Tnew[-1] = Tnew[-2]
        
        # Reapply fixed Dirichlet temperatures
        for i, temp in delta_index.items():
            Tnew[i] = temp

        T_hist.append(Tnew.copy())
        T = Tnew

    end = time.perf_counter()
    runtime = end - start
    return T, T_hist, runtime


def GPU_CuPy2D(config):
    print(f"2D GPU CuPy solver running...")
    r = diffusion_number(config)
    
    dx = config['dx']
    x_size = config['x_size']
    T_start = config['T']
    
    # Same dx divisibility check as CPU2D
    ratio = x_size / dx
    valid_step_size = float(ratio).is_integer()
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return

    length = int(x_size / dx) + 1

    # GPU 2D array construction
    T_init = cp.ones((length, length)) * T_start

    # Apply fixed-temperature deltas (same logic as CPU2D)
    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}   
    for (i, j), temp in temp_deltas.items():
        
        if i > x_size or i < 0 or j > x_size or j < 0:
            print("Error: A position in temp_delta is out of bounds")
            return

        # Convert physical coords to nearest indices
        idx_x = int(round(i/dx))
        idx_y = int(round(j/dx))

        T_init[(idx_y, idx_x)] = temp
        delta_index[(idx_y, idx_x)] = temp

    T = T_init.copy()
    t_steps = int(config['t_steps'])
    T_hist = []
    T_hist.append(T)

    # ================================
    # MAIN TIME LOOP (vectorised 2D FD)
    # ================================
    start = time.perf_counter()
    for _ in range(t_steps):
        Tnew = T.copy()

        # 5-point Laplacian stencil (fully vectorised)
        Tnew[1:-1, 1:-1] = (
            T[1:-1, 1:-1] + r * (
                T[2:,   1:-1] +   # down
                T[:-2,  1:-1] +   # up
                T[1:-1, 2:]   +   # right
                T[1:-1, :-2]  -   # left
                4*T[1:-1, 1:-1]
            )
        )
        
        # Neumann boundaries (same pattern as CPU2D)
        Tnew[0, :]   = Tnew[1, :]
        Tnew[-1, :]  = Tnew[-2, :]
        Tnew[:, 0]   = Tnew[:, 1]
        Tnew[:, -1]  = Tnew[:, -2]
        
        # Reapply fixed temps
        for (y,x), temp in delta_index.items():
            Tnew[y][x] = temp
        
        T_hist.append(Tnew.copy())
        T = Tnew
    
    end = time.perf_counter()
    runtime = end - start
    return T, T_hist, runtime


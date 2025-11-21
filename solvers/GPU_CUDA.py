from config import diffusion_number
import time
import cupy as cp

# ======================================================================
# GPU_CUDA1D / GPU_CUDA2D — Raw CUDA Kernel Heat Solvers
# ----------------------------------------------------------------------
# These solvers implement *the exact same finite-difference logic* as
# CPU1D / CPU2D:
#
#   • Same explicit forward-Euler time-marching
#   • Same 1D / 2D central-difference stencils
#   • Same Dirichlet and Neumann boundary conditions
#   • Same mapping of physical coordinates → grid indices
#
# The only differences:
#
#   1) The FD updates are executed inside custom CUDA __global__ kernels  
#      (`heat1d` and `heat2d`) instead of Python loops.
#
#   2) CuPy is used purely as a GPU memory manager for:
#         - allocating arrays (T, Tnew)
#         - flattening / reshaping for the kernels
#         - launching kernels through cp.RawKernel
#
#   3) Each GPU thread handles exactly one grid point calculation,
#      giving massive acceleration for large grids.
#
# Apart from these execution-level changes, the underlying numerical
# method is identical to the CPU solvers.
# ======================================================================


kernel_1d = cp.RawKernel(r"""
extern "C" __global__
void heat1d(double* Tnew, const double* T, const double r, const int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= 1 && i < N-1) {
        Tnew[i] = T[i] + r * (T[i+1] + T[i-1] - 2.0 * T[i]);
    }
}
""", "heat1d")


kernel_2d = cp.RawKernel(r"""
extern "C" __global__
void heat2d(double* Tnew, const double* T, const double r,
            const int Nx, const int Ny)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x; // x index
    int i = blockDim.y * blockIdx.y + threadIdx.y; // y index

    if (i >= 1 && i < Ny-1 && j >= 1 && j < Nx-1) {
        int idx = i * Nx + j;

        double up    = T[(i-1)*Nx + j];
        double down  = T[(i+1)*Nx + j];
        double left  = T[i*Nx + (j-1)];
        double right = T[i*Nx + (j+1)];
        double centre = T[idx];

        Tnew[idx] = centre + r * (up + down + left + right - 4.0 * centre);
    }
}
""", "heat2d")



def GPU_CUDA1D(config):
    print(f"1D GPU CUDA RawKernel solver running...\n")
    r = diffusion_number(config)
    
    dx = config['dx']
    x_size = config['x_size']
    T_start = config['T']

    # Same dx divisibility requirement as CPU version
    ratio = x_size / dx
    valid_step_size = float(ratio).is_integer()
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return
    
    # Grid construction (NumPy → CuPy equivalent)
    N = int(x_size / dx) + 1
    T_init = cp.ones(N, dtype=cp.float64) * T_start

    # Map fixed temperature positions to nearest grid indices
    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}   
    for i, temp in temp_deltas.items():
        
        if i > x_size or i < 0:
            print("Error: A position in temp_delta is out of bounds")
            return
        idx = int(round(i/dx))     # same logic as CPU solver
        T_init[idx] = temp
        delta_index[idx] = temp

    # Thread/block configuration for 1D kernel
    threads = 256
    blocks = (N + threads - 1) // threads
    print(f"There are {N} grid points so, with blocks of 256 threads,"
          f" we require {blocks} blocks\n")
    
    T = T_init.copy()
    t_steps = int(config['t_steps'])
    T_hist = []
    T_hist.append(T)

    # ------------------------------
    # Main explicit FD time loop
    # ------------------------------
    start = time.perf_counter()
    for _ in range(t_steps):
        Tnew = T.copy()

        # Launch CUDA kernel (one thread per grid point)
        kernel_1d((blocks,), (threads,), (Tnew, T, r, N))
        
        # Neumann BCs
        Tnew[0] = Tnew[1]
        Tnew[-1] = Tnew[-2]
        
        # Dirichlet fixed temps
        for i, temp in delta_index.items():
            Tnew[i] = temp

        T_hist.append(Tnew.copy())
        T = Tnew

    end = time.perf_counter()
    runtime = end - start
    return T, T_hist, runtime



def GPU_CUDA2D(config):
    print(f"2D GPU CUDA solver running...")
    r = diffusion_number(config)
    
    dx = config['dx']
    x_size = config['x_size']
    T_start = config['T']
    
    # Same dx/grid-size consistency check
    ratio = x_size / dx
    valid_step_size = float(ratio).is_integer()
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return

    # Square grid setup
    N = int(x_size / dx) + 1
    Nx = Ny = N

    T_init = cp.ones((Ny, Nx), dtype=cp.float64) * T_start
    Tnew = T_init.copy()

    # Apply temp deltas exactly as in CPU2D
    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}   
    for (i, j), temp in temp_deltas.items():
        
        if i > x_size or i < 0 or j > x_size or j < 0:
            print("Error: A position in temp_delta is out of bounds")
            return
        idx_x = int(round(i/dx))
        idx_y = int(round(j/dx))
        T_init[(idx_y, idx_x)] = temp
        delta_index[(idx_y, idx_x)] = temp

    t_steps = int(config['t_steps'])
    T_hist = []
    T_hist.append(T_init.copy())

    # Flatten arrays for CUDA kernel (faster, coalesced memory)
    T_flat = T_init.ravel()
    Tnew_flat = Tnew.ravel()

    # 16×16 thread blocks, covering the full grid
    threads = (16,16)
    blocks = ((Nx + 15)//16, (Ny + 15)//16)

    # ------------------------------
    # Main explicit FD time loop
    # ------------------------------
    start = time.perf_counter()
    for n in range(t_steps):

        # Launch 2D CUDA laplacian kernel
        kernel_2d(blocks, threads, (Tnew_flat, T_flat, r, Nx, Ny))
        
        # Reshape to 2D to apply BCs and temperature deltas
        Tnew = Tnew_flat.reshape(Ny, Nx)

        # Neumann BCs (zero-gradient)
        Tnew[0, :]  = Tnew[1, :]
        Tnew[-1, :] = Tnew[-2, :]
        Tnew[:, 0]  = Tnew[:, 1]
        Tnew[:, -1] = Tnew[:, -2]
        
        # Reapply Dirichlet fixed-temperature points
        for (y,x), temp in delta_index.items():
            Tnew[y][x] = temp
        
        T_hist.append(Tnew.copy())

        # Swap references (avoids extra memory allocation)
        T_flat, Tnew_flat = Tnew_flat, T_flat
    
    end = time.perf_counter()
    runtime = end - start
    return T_hist[-1], T_hist, runtime

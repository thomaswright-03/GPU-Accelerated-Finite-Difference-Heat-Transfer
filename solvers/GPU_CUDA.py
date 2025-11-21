from config import diffusion_number
import time
import cupy as cp

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

    ratio = x_size / dx
    valid_step_size = float(ratio).is_integer()
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return
    
    # Grid setup
    N = int(x_size / dx) + 1
    T_init = cp.ones(N, dtype=cp.float64) * T_start

    #index temp deltas
    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}   
    for i, temp in temp_deltas.items():
        
        if i > x_size or i < 0:
            print("Error: A position in temp_delta is out of bounds")
            return
        idx = int(round(i/dx)) #integer division allows temp to be placed at nearest point
        T_init[idx] = temp
        delta_index[idx] = temp

    threads = 256
    blocks = (N + threads - 1) // threads
    print(f"There are {N} grid points so, with blocks of 256 threads,"
          f" we require {blocks} blocks\n")
    
    T = T_init.copy()
    t_steps = int(config['t_steps'])
    T_hist = []
    T_hist.append(T)

    #perform finite difference analysis
    start = time.perf_counter()
    for _ in range(t_steps):
        Tnew = T.copy()

        #laucnh CUDA kernel for update with central difference
        kernel_1d((blocks,), (threads,), (Tnew, T, r, N))
        
        #Apply Neumann boundary conditions
        Tnew[0] = Tnew[1]
        Tnew[-1] = Tnew[-2]
        
        #Apply fixed temps
        for i, temp in delta_index.items():
            Tnew[i] = temp
            #print(i)
            #print(temp)  
        #Add to history of T distribution
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
    
    ratio = x_size / dx
    valid_step_size = float(ratio).is_integer()
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return

    N = int(x_size / dx) + 1
    Nx = Ny = N

    T_init = cp.ones((Ny, Nx), dtype=cp.float64) * T_start
    Tnew = T_init.copy()

    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}   
    for (i, j), temp in temp_deltas.items():
        
        if i > x_size or i < 0 or j > x_size or j < 0:
            print("Error: A position in temp_delta is out of bounds")
            return
        idx_x = int(round(i/dx)) #integer division allows temp to be placed at nearest point
        idx_y = int(round(j/dx))
        T_init[(idx_y, idx_x)] = temp
        delta_index[(idx_y, idx_x)] = temp

    t_steps = int(config['t_steps'])
    T_hist = []
    T_hist.append(T_init.copy())

    T_flat = T_init.ravel()
    Tnew_flat = Tnew.ravel()

    threads = (16,16)
    blocks = ((Nx + 15)//16, (Ny + 15)//16)

    #perform finite difference analysis
    start = time.perf_counter()
    for n in range(t_steps):
        #launch CUDA kernel and update with central difference
        kernel_2d(blocks, threads, (Tnew_flat, T_flat, r, Nx, Ny))
        
        # reshape to apply BC & deltas
        Tnew = Tnew_flat.reshape(Ny, Nx)

        ##Apply Neumann boundary conditions##
        ##Top edge
        Tnew[0, :] = Tnew[1, :]
        # Bottom edge
        Tnew[-1, :] = Tnew[-2, :]
        # Left edge
        Tnew[:, 0] = Tnew[:, 1]
        # Right edge
        Tnew[:, -1] = Tnew[:, -2]
        
        #Apply fixed temps
        for (y,x), temp in delta_index.items():
            Tnew[y][x] = temp
            #print(i)
            #print(temp) 
        
        #Add to history of T distribution
        T_hist.append(Tnew.copy())
        T_flat, Tnew_flat = Tnew_flat, T_flat
    
    end = time.perf_counter()
    runtime = end - start
    return T_hist[-1], T_hist, runtime


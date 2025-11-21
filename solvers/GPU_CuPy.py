from config import diffusion_number
import time
import cupy as cp

def GPU_CuPy1D(config):
    print(f"1D GPU CuPy solver running...\n")
    r = diffusion_number(config)
    
    dx = config['dx']
    x_size = config['x_size']
    T_start = config['T']

    ratio = x_size / dx
    valid_step_size = float(ratio).is_integer()
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return
    
    X = cp.arange(0, x_size+dx, dx)
    T_init = cp.ones_like(X) * T_start

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

    T = T_init.copy()
    t_steps = int(config['t_steps'])
    T_hist = []
    T_hist.append(T)

    #perform finite difference analysis
    start = time.perf_counter()
    for _ in range(t_steps):
        Tnew = T.copy()

        #Update with central difference
        Tnew[1:-1] = T[1:-1] + r * (T[2:] + T[:-2] - 2*T[1:-1])
        
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

def GPU_CuPy2D(config):
    print(f"2D GPU CuPy solver running...")
    r = diffusion_number(config)
    
    dx = config['dx']
    x_size = config['x_size']
    T_start = config['T']
    
    ratio = x_size / dx
    valid_step_size = float(ratio).is_integer()
    if not valid_step_size:
        print("Error: x_size must be divisible by dx.")
        return

    length = int(x_size / dx) + 1

    T_init = cp.ones((length, length)) * T_start

    temp_deltas = config.get("temp_deltas", {})
    delta_index = {}   
    for (i, j), temp in temp_deltas.items():
        
        if i > x_size or i < 0 or j > x_size or j < 0: #change to be ignore?
            print("Error: A position in temp_delta is out of bounds")
            return
        idx_x = int(round(i/dx)) #integer division allows temp to be placed at nearest point
        idx_y = int(round(j/dx))
        T_init[(idx_y, idx_x)] = temp
        delta_index[(idx_y, idx_x)] = temp

    T = T_init.copy()
    t_steps = int(config['t_steps'])
    T_hist = []
    T_hist.append(T)

    #perform finite difference analysis
    start = time.perf_counter()
    for _ in range(t_steps):
        Tnew = T.copy()

        #Update with central difference
        Tnew[1:-1, 1:-1] = (T[1:-1, 1:-1] + r * (
                           T[2:, 1:-1] + #down
                           T[:-2, 1:-1] + #up
                           T[1:-1, 2:] + #right
                           T[1:-1, :-2] - #left
                           4*T[1:-1, 1:-1]
                        )
        )
        
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

        T = Tnew
    
    end = time.perf_counter()
    runtime = end - start
    return T, T_hist, runtime


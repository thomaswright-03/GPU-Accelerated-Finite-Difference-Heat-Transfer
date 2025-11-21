import time
import numpy as np
import cupy as cp

from test import check_gpu                           # GPU detection utility
from solvers.CPU import CPU1D, CPU2D                 # Pure NumPy CPU solvers
from solvers.GPU_CuPy import GPU_CuPy1D, GPU_CuPy2D  # CuPy-vectorised GPU solvers
from solvers.GPU_CUDA import GPU_CUDA1D, GPU_CUDA2D  # Raw CUDA kernel solvers
from plotting.plot1D import heatmap1D, probe1D       # 1D plotting utilities
from plotting.plot2D import heatmap2D, probe2D       # 2D plotting utilities

# ---------------------------------------------------------
# INITIAL SETUP
# ---------------------------------------------------------

# Check GPU availability and print device info
check_gpu()

# Warm-up kernel for CuPy to reduce first-call overhead
cp.arange(10).sum()                                  # Also confirms CUDA backend works

# ---------------------------------------------------------
# 1D SIMULATION CONFIGURATION
# ---------------------------------------------------------
# All solver functions expect a config dict with standard keys:
# alpha       → thermal diffusivity
# dt, dx      → timestep & spatial increment
# dims        → dimensionality (1 or 2)
# t_steps     → number of time iterations
# x_size      → physical length of the domain
# T           → initial temperature baseline
# temp_deltas → dictionary of fixed points with elevated temperatures

config1D = {
    "alpha": 1.12e-04,      # Thermal diffusivity (m²/s)
    "dt": 0.001,            # Time step size
    "dx": 0.001,            # Spatial resolution (equal in all dimensions)
    "dims": 1,              # 1D simulation
    "t_steps": 100000,      # Total number of timesteps
    "x_size": 0.3,          # Rod length in metres
    "T": 25,                # Initial uniform temperature

    # Hot boundary / heat-source locations along the rod (1D)
    "temp_deltas": {
        0.00: 116,          # Heat applied at x = 0.0 m
        # Additional points can be added here
    }
}

# ---------------------------------------------------------
# 2D SIMULATION CONFIGURATION
# ---------------------------------------------------------

config2D = {
    "alpha": 1.12e-04,      # Same diffusivity
    "dt": 0.001,
    "dx": 0.001,
    "dims": 2,              # 2D simulation
    "t_steps": 20000,
    "x_size": 0.1,          # Square plate (0.1m × 0.1m)
    "T": 25,

    # Five randomly generated hot spots inside the 2D plate.
    # Format: {(x,y): temperature}
    "temp_deltas": {
        (0.0731, 0.0186): 191.42,
        (0.0057, 0.0829): 158.33,
        (0.0442, 0.0638): 175.89,
        (0.0914, 0.0291): 167.54,
        (0.0226, 0.0487): 198.77
    }
}

# ---------------------------------------------------------
# SOLVER WRAPPERS — each runs a solver, prints summary,
# saves final arrays, and returns (T_final, T_hist, runtime)
# ---------------------------------------------------------

def runCPU1D(config):
    T_final, T_hist, runtime = CPU1D(config)         # Run NumPy FD solver

    np.savetxt("rawdata/T_final - CPU - 1D.csv", T_final, delimiter=',')
    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    print(f"Runtime of CPU1D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runCPU2D(config):
    T_final, T_hist, runtime = CPU2D(config)

    np.savetxt("rawdata/T_final - CPU - 2D.csv", T_final, delimiter=',')
    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    print(f"Runtime of CPU2D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runGPU_CuPy1D(config):
    T_final, T_hist, runtime = GPU_CuPy1D(config)    # CuPy vectorised GPU solver

    np.savetxt("rawdata/T_final - GPU_CuPy - 1D.csv", T_final, delimiter=',')
    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    print(f"Runtime of GPU_CuPy1D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runGPU_CuPy2D(config):
    T_final, T_hist, runtime = GPU_CuPy2D(config)

    np.savetxt("rawdata/T_final - GPU_CuPy - 2D.csv", T_final, delimiter=',')
    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    print(f"Runtime of GPU_CuPy2D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runGPU_CUDA1D(config):
    T_final, T_hist, runtime = GPU_CUDA1D(config)    # Raw CUDA kernel acceleration

    np.savetxt("rawdata/T_final - GPU_CUDA - 1D.csv", T_final, delimiter=',')
    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    print(f"Runtime of GPU_CUDA1D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runGPU_CUDA2D(config):
    T_final, T_hist, runtime = GPU_CUDA2D(config)    # 2D CUDA kernel solver

    np.savetxt("rawdata/T_final - GPU_CUDA - 2D.csv", T_final, delimiter=',')
    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    print(f"Runtime of GPU_CUDA2D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

# ---------------------------------------------------------
# RUN SELECTED SOLVERS
# ---------------------------------------------------------
# Currently selected the fastest (CUDA) solvers for both 1D and 2D

#Tmap1D, Thist1D, _ = runCPU1D(config1D)
#Tmap2D, Thist2D, _ = runCPU2D(config2D)
#Tmap1D, Thist1D, _ = runGPU_CuPy1D(config1D)
#Tmap2D, Thist2D, _ = runGPU_CuPy2D(config2D)
Tmap1D, Thist1D, _ = runGPU_CUDA1D(config1D)         # Run full 1D CUDA simulation
Tmap2D, Thist2D, _ = runGPU_CUDA2D(config2D)         # Run full 2D CUDA simulation

# ---------------------------------------------------------
# PLOTTING (HEATMAPS + PROBE TIME SERIES)
# ---------------------------------------------------------
# Plots are saved to /plotting/plots with a filename prefix.

heatmap1D(Tmap1D, config1D, probeX=0.2, prefix="Simulation 1 - ")   # 1D final distribution
heatmap2D(Tmap2D, config2D, probeX=0.065, probeY=0.055, prefix="Simulation 2 - ") # 2D map

probe1D(Thist1D, config1D, probeX=0.2, prefix="Simulation 1 - ")    # 1D probe vs time
probe2D(Thist2D, config2D, probeX=0.065, probeY=0.055, prefix="Simulation 2 - ")  # 2D probe vs time

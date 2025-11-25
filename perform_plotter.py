##This script is only intended for plotting the performance curve,
##It takes a long time to run. Use main.py to perform individual FD simulations

import contextlib, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Import the three solvers used for benchmarking
from solvers.CPU import CPU2D                # Pure NumPy CPU solver
from solvers.GPU_CuPy import GPU_CuPy2D      # CuPy vectorised GPU solver
from solvers.GPU_CUDA import GPU_CUDA2D      # Raw CUDA kernel solver

# ================================================================
# LOAD BENCHMARK CONFIG DATA
# ================================================================
# The CSV contains different (N, dx, dt, t_steps) rows for benchmarking.
# They represent varying grid resolutions for a 0.1 × 0.1 m plate.
# The first few rows are "CPU-safe", meaning CPU runtime is reasonable.
# Larger grids are omitted to avoid extremely long CPU runtimes.
# ================================================================

df = pd.read_csv("plotting/performance/config_data.csv")   # Load grid settings

# ================================================================
# SELECT HOW MANY TESTS TO RUN INCLUDING CPU
# ================================================================
# CPU gets very slow for large N², so only the first few entries
# from the CSV are used for CPU timing.
# GPU solvers could handle much more, but in this script we keep
# all solvers aligned for clean multi-solver comparison.
# ================================================================

iter_with_CPU = 6                    # Number of configurations where CPU is included
df_run_CPU = df.iloc[:iter_with_CPU] # Slice the first rows only

# ================================================================
# GLOBAL SIMULATION CONSTANTS (same for all runs)
# ================================================================

alpha = 1.12e-04   # Thermal diffusivity
T0 = 25            # Baseline temperature
L  = 0.1           # Plate side length (square domain)

# ================================================================
# MAIN BENCHMARK LOOP
# ================================================================
# For each selected grid resolution:
# 1. Build a solver config dictionary
# 2. Run CPU, CuPy-GPU, and CUDA-GPU solvers
# 3. Record their runtimes
#
# The temperature distribution itself is ignored; only timing matters.
# ================================================================

results = []       # Stores runtime entries for final plotting

for idx, row in df_run_CPU.iterrows():

    print(f"Iteration Number: {idx+1}/{iter_with_CPU} (with CPU)")

    # Extract settings from CSV row
    N        = int(row["N"])
    dx       = float(row["dx"])
    dt       = float(row["dt"])
    t_steps  = int(row["t_steps"])

    # ------------------------------------------------------------
    # Build the config dictionary expected by all solver functions
    # ------------------------------------------------------------
    config = {
        "alpha": alpha,
        "dt": dt,
        "dx": dx,
        "dims": 2,
        "t_steps": t_steps,
        "x_size": L,
        "T": T0,

        # A single heat pulse at the centre of the plate
        # Ensures consistent physics across resolutions
        "temp_deltas": {
            (L/2, L/2): 150
        }
    }

    # ------------------------------------------------------------
    # RUN SOLVERS (timing taken inside solver functions)
    # ------------------------------------------------------------
    # The underscore (_) is used to ignore the returned T_final and T_hist.
    # Only runtimes (t_CPU, t_CuPy, t_CUDA) are used.
    # ------------------------------------------------------------

    _, _, t_CPU  = CPU2D(config)
    _, _, t_CuPy = GPU_CuPy2D(config)
    _, _, t_CUDA = GPU_CUDA2D(config)

    # ------------------------------------------------------------
    # Store results as a dict in a list (later converted to DataFrame)
    # ------------------------------------------------------------
    results.append({
        "N^2"       : N**2,     # Total number of grid points = resolution metric
        "CPU_time"  : t_CPU,
        "CuPy_time" : t_CuPy,
        "CUDA_time" : t_CUDA
    })

# ================================================================
# CONVERT RESULTS TO DATAFRAME + SORT
# ================================================================

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("N^2")  # Ensure ascending resolution order

# ================================================================
# PLOTTING THE PERFORMANCE CURVE
# ================================================================
# Produces a line graph comparing:
#   - CPU runtime
#   - GPU CuPy runtime
#   - GPU CUDA RawKernel runtime
#
# X-axis uses N² (grid size), representing total workload.
#
# The resulting figure is saved to /plotting/performance/
# ================================================================

plt.figure(figsize=(10,6))

# Plot CPU curve
plt.plot(df_results["N^2"], df_results["CPU_time"],
         marker='o', linewidth=2, label="CPU")

# Plot CuPy GPU curve
plt.plot(df_results["N^2"], df_results["CuPy_time"],
         marker='o', linewidth=2, label="CuPy (GPU)")

# Plot CUDA RawKernel curve
plt.plot(df_results["N^2"], df_results["CUDA_time"],
         marker='o', linewidth=2, label="CUDA RawKernel (GPU)")

plt.xlabel("Grid size (N²)", fontsize=12)
plt.ylabel("Runtime (s)", fontsize=12)
plt.title("2D Heat Equation Runtime vs Grid Size", fontsize=14)

#Set the major and minor tickers, grid, and x/y lims
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(250))

plt.grid(True, which='major', alpha=0.3)
plt.grid(True, which='minor', alpha=0.15)

plt.xlim(-500,11000) #Change if larger grids are used
plt.ylim(-5,t_CPU*1.1)

# Hardware footnote added to figure (not axis)
plt.figtext(0.99, 0.01,
            "Hardware: i7-12700H, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), CUDA 12.4",
            ha="right", fontsize=8)

# Styling
plt.legend()
plt.tight_layout()

# Save performance plot
plt.savefig("plotting/performance/performance_curve.png", dpi=200)
plt.show()

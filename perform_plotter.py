##This script is only intended for plotting the performance curve,
##It takes a long time to run. Use main.py to perform individual FD simulations

import contextlib, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from solvers.CPU import CPU2D
from solvers.GPU_CuPy import GPU_CuPy2D
from solvers.GPU_CUDA import GPU_CUDA2D


    
df = pd.read_csv("plotting/performance/config_data.csv")

iter_with_CPU = 6 
df_run_CPU = df.iloc[:iter_with_CPU]
#df_run_NoCPU = df.iloc[iter_with_CPU:]

alpha = 1.12e-04
T0 = 25
L = 0.1 # plate size

results = []

for idx, row in df_run_CPU.iterrows():

    print(f"Iteration Number: {idx+1}/{iter_with_CPU} (with CPU)")

    N = int(row["N"])
    dx = float(row["dx"])
    dt = float(row["dt"])
    t_steps = int(row["t_steps"])

    # Build dictionary for solvers
    config = {
        "alpha": alpha,
        "dt": dt,
        "dx": dx,
        "dims": 2,
        "t_steps": t_steps,
        "x_size": L,
        "T": T0,

        "temp_deltas": {
        (L/2, L/2): 150,  #delta T at centre of plate
        }
        }

    _, _, t_CPU = CPU2D(config)
    _, _, t_CuPy = GPU_CuPy2D(config)
    _, _, t_CUDA = GPU_CUDA2D(config)

    results.append({
        "N^2" : N**2,
        "CPU_time" :  t_CPU,
        "CuPy_time" : t_CuPy,
        "CUDA_time" : t_CUDA
        })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("N^2")

plt.figure(figsize=(10,6))

plt.plot(df_results["N^2"], df_results["CPU_time"], 
         marker='o', linewidth=2, label="CPU")

plt.plot(df_results["N^2"], df_results["CuPy_time"], 
         marker='o', linewidth=2, label="CuPy (GPU)")

plt.plot(df_results["N^2"], df_results["CUDA_time"], 
         marker='o', linewidth=2, label="CUDA RawKernel (GPU)")

plt.xlabel("Grid size (N²)", fontsize=12)
plt.ylabel("Runtime (s)", fontsize=12)
plt.title("2D Heat Equation Runtime vs Grid Size", fontsize=14)

plt.figtext(0.99, 0.01,
            "Hardware: i7-12700H, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), CUDA 12.4",
            ha="right", fontsize=8)


plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

# Save + show
plt.savefig("plotting/performance/performance_curve.png", dpi=200)
plt.show()
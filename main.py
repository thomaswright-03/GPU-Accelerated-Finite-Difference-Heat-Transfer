import time
import numpy as np
import cupy as cp

from test import check_gpu
from solvers.CPU import CPU1D, CPU2D
from solvers.GPU_CuPy import GPU_CuPy1D, GPU_CuPy2D
from solvers.GPU_CUDA import GPU_CUDA1D, GPU_CUDA2D
from plotting.plot1D import heatmap1D, probe1D
from plotting.plot2D import heatmap2D, probe2D

#check GPU
check_gpu()

#warm up
cp.arange(10).sum()

config1D = {
    "alpha": 1.12e-04, #Thermal Diffusivity
    "dt": 0.001,       #Time Step
    "dx": 0.001,        #Distance Step assuming dx=dy=dz
    "dims": 1,         #Number of Dimensions
    "t_steps": 100000,    #t_steps * dt = total time
    "x_size": 0.3,      #Total dimension of simulation
    "T": 25,           #Initial temp in degrees C
    
    #user-defined temperature discontinuities in format (X,Y) : Temp
    "temp_deltas": {
        0.00: 116,
        #0.75: 60,
        #0.50: 100
    }
}

config2D = {
    "alpha": 1.12e-04,
    "dt": 0.001,
    "dx": 0.001,
    "dims": 2,
    "t_steps": 20000,
    "x_size": 0.1,
    "T": 25,

    #user-defined temperature discontinuities in format (X,Y) : Temp
    "temp_deltas": {
        (0.0731, 0.0186): 191.42,
        (0.0057, 0.0829): 158.33,
        (0.0442, 0.0638): 175.89,
        (0.0914, 0.0291): 167.54,
        (0.0226, 0.0487): 198.77
    }
}

def runCPU1D(config):
    T_final, T_hist, runtime = CPU1D(config)

    np.savetxt("rawdata/T_final - CPU - 1D.csv", T_final, delimiter=',')

    #print(f"Initial Distribution: {T_hist[0]}")
    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    print(f"Runtime of CPU1D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runCPU2D(config):
    T_final, T_hist, runtime = CPU2D(config)

    np.savetxt("rawdata/T_final - CPU - 2D.csv", T_final, delimiter=',')

    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    #print(T_hist[-2])
    print(f"Runtime of CPU2D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runGPU_CuPy1D(config):
    T_final, T_hist, runtime = GPU_CuPy1D(config)

    np.savetxt("rawdata/T_final - GPU_CuPy - 1D.csv", T_final, delimiter=',')

    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    #print(T_hist[-2])
    print(f"Runtime of GPU_CuPy1D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runGPU_CuPy2D(config):
    T_final, T_hist, runtime = GPU_CuPy2D(config)

    np.savetxt("rawdata/T_final - GPU_CuPy - 2D.csv", T_final, delimiter=',')

    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    #print(T_hist[-2])
    print(f"Runtime of GPU_CuPy2D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runGPU_CUDA1D(config):
    T_final, T_hist, runtime = GPU_CUDA1D(config)

    np.savetxt("rawdata/T_final - GPU_CUDA - 1D.csv", T_final, delimiter=',')

    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    #print(T_hist[-2])
    print(f"Runtime of GPU_CUDA1D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

def runGPU_CUDA2D(config):
    T_final, T_hist, runtime = GPU_CUDA2D(config2D)

    np.savetxt("rawdata/T_final - GPU_CUDA - 2D.csv", T_final, delimiter=',')

    print(f"Final Distribution after {config['dt']*config['t_steps']:.2f} seconds: {T_final}")
    #print(T_hist[-2])
    print(f"Runtime of GPU_CUDA2D = {runtime:.4f} seconds")
    return T_final, T_hist, runtime

##Call Solvers
#Tmap1D, Thist1D, _ = runCPU1D(config1D)
#Tmap2D, Thist2D, _ = runCPU2D(config2D)
#Tmap1D, Thist1D, _ = runGPU_CuPy1D(config1D)
#Tmap2D, Thist2D, _ = runGPU_CuPy2D(config2D)
Tmap1D, Thist1D, _ = runGPU_CUDA1D(config1D)
Tmap2D, Thist2D, _ = runGPU_CUDA2D(config2D)

##Call Plotters
heatmap1D(Tmap1D, config1D, probeX=0.2, prefix="Simultion 1 - ")
heatmap2D(Tmap2D, config2D, probeX=0.065, probeY=0.055, prefix="Simultion 2 - ")

probe1D(Thist1D, config1D, probeX=0.2, prefix="Simultion 1 - ")
probe2D(Thist2D, config2D, probeX=0.065, probeY=0.055, prefix ="Simultion 2 - ")

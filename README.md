HeatTransfer Simulation Project:
    A Python project for solving 1-D and 2-D transient heat-conduction problems 
    using explicit finite-difference methods (CPU and GPU versions). Designed as
    a demo for experimentation, visualisation, and performance comparisons between 
    solvers.

HeatTransfer/
│
├── main.py               # Entry point for running simulations
├── Solvers/
│   |__ CPU.py            # CPU-based 1D and 2D solvers (explicit scheme)
│   |__ GPU_CuPy.py       # GPU version (CuPy - Vectorised finite difference update)
│   |__ GPU_CUDA.py       # GPU version (CUDA RawKernel in C)
|
|__ config.py               #Calulates HT rate and checks stability
|__ test.py                 #Checks function of GPU
|__ perform_plotter.py      #Performs and plots a series of simulations with identical physics to compare runtimes between 2D solvers
|
├── rawdata/                # Output CSV files and simulation results
|── Plotting/               # (optional) Saved figures, heatmaps, or animations
|    |__ __init.py__         # Can be ignored
|    |__ plot1D.py           # Plots 1D data onto heatmap and scatter of specified probe point
|    |__ plot2D.py           # Plots 2D data onto heatmap and scatter of specified probe point
|    |  
|    |__ plots/            # Contains .png plots
|    |__ performance/      # Contains results of runtime analysis


Features:
    1-D and 2-D explicit heat-diffusion solvers
    Custom temperature discontinuities via config dictionaries
    Neumann (insulated) boundary conditions
    Dirichlet fixed-temperature points
    Export final temperature fields to CSV
    Modular structure for future extensions (GPU, animations, measurement 'probe' plots)

Running a Simulation:
    Configure .venv with python 3.12 or newer and run:

        pip install -r requirements.txt

    This project includes GPU-accelerated solvers implemented using CuPy (vectorised kernels) and CUDA RawKernel (custom CUDA C kernels). Therefore, an NVIDIA GPU and CUDA toolkit versions compatible with CuPy are required.
    
    Modify config1D or config2D in main.py, and select solver functions, then run:

        python main.py

    with desired solver function calls uncommented

    The run functions output:
        Final temperature field
        Temperature history
        Runtime
        CSV export to the rawdata/ folder

    Then, uncomment/call desired plot functions which plot:
        A heatmap
        A scatter graph of a specified temperature point

Configuration Format (example):
    
    config2D = {
        "alpha": 1.12e-04,
        "dt": 0.001,
        "dx": 0.01,
        "dims": 2,
        "t_steps": 100000,
        "x_size": 0.1,
        "T": 25,

        "temp_deltas": {
            (0.01, 0.01): 116,
            (0, 0.01): 90,
            (0.01, 0): 100
        }
    }

WARNINGS:
    Boundary Behaviour and Corner Node Restriction (2D Solver):
        The 2D heat solver uses:
            Explicit finite-difference update
            Interior-only CUDA kernel
            Neumann (zero-flux) boundary conditions applied after each timestep
            
        Because of this combination, the four corner nodes do not participate in the finite-difference update. The boundaries are overwritten during the Neumann update step, so any fixed-temperature delta placed exactly at a corner will be replaced before it can influence neighbouring cells.

        Valid positions for fixed-temperature sources:
            Fixed-temperature points may be placed on edges, but not in the four corners:
                (0, 0) → invalid
                (0, L) → invalid
                (L, 0) → invalid
                (L, L) → invalid
            
            Instead place the heat source one point diagonally inwards i.e:
                (dx, dx) → valid

To-Do / Future Updates
    Add animation of heat propagation
    Add configurable boundary-condition types
    Add mesh-grid visualisation
    Add automated stability check
    Add logging & progress bars



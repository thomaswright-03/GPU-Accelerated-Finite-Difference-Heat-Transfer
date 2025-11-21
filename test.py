import cupy as cp

def check_gpu():
    """
    Prints information about available CUDA-capable GPUs using CuPy.
    
    This function is used at the start of the program to verify that the system
    can run GPU-accelerated solvers (CuPy or CUDA raw kernels). It checks:
    
    - Whether any CUDA devices are detected
    - The number of GPUs available
    - The name/model of each GPU detected
    
    If CUDA is unavailable, it prints a safe fallback message rather than
    crashing the main program.
    """
    try:
        # Query number of CUDA devices using the CuPy runtime
        n = cp.cuda.runtime.getDeviceCount()

        if n == 0:
            # No CUDA devices detected (CuPy installed but GPU missing)
            print("No GPU detected.")
        else:
            # Print number of GPUs available
            print(f"{n} GPU(s) detected.")
            for i in range(n):
                # Fetch properties for each GPU
                props = cp.cuda.runtime.getDeviceProperties(i)

                # GPU name is stored as bytes → decode to UTF-8 string
                name = props["name"].decode("utf-8")
                print(f" - GPU {i}: {name}")

        print("\n")  # Formatting newline

    except cp.cuda.runtime.CUDARuntimeError:
        # Raised when CUDA is not installed, incompatible, or inaccessible
        print("CUDA not available on this system.")

# check_gpu()  
# Uncomment to run this script standalone.
# Leave commented when imported into solvers or main.py to avoid duplicate output.
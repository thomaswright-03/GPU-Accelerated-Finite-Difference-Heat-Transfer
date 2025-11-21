import cupy as cp

def check_gpu():
    try:
        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            print("No GPU detected.")
        else:
            print(f"{n} GPU(s) detected.")
            for i in range(n):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props["name"].decode("utf-8")
                print(f" - GPU {i}: {name}")
        print("\n")
    except cp.cuda.runtime.CUDARuntimeError:
        print("CUDA not available on this system.")

#check_gpu() #uncomment to run this script independently
             #Comment out to prevent duplicate runs within solvers
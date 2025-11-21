import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import pandas

def heatmap2D(T, config, probeX=None, probeY=None, prefix=None):
    
    try:
        T = T.get()
    except AttributeError:
        T = np.asarray(T)

    dx = config['dx']
    dy = dx  # assume square grid
    Nx = T.shape[1]
    Ny = T.shape[0]

    Lx = config['x_size']
    Ly = Lx

    T0 = config["T"]

    T_min = min(T.min(), T0)
    T_max = max(T.max(), T0)

    time = config['t_steps'] * config['dt']

    if prefix is not None:
        title = f"{prefix} 2D Temperature Distribution of Plate"
    else:
        title = f"2D Temperature Distribution of Plate"

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(
        T,
        cmap=plt.cm.plasma,
        extent=[0, Lx, 0, Ly],
        origin="lower",
        aspect="equal",
        vmin=T_min,
        vmax=T_max
    )

    cbar = plt.colorbar()
    cbar.set_label("Temperature", fontsize=12)

    plt.title(f"{title} after {time:.1f} seconds", fontsize=16)
    plt.xlabel("X position (m)", fontsize=14)
    plt.ylabel("Y position (m)", fontsize=14)

    plt.xticks(np.linspace(0, Lx, 11))
    plt.yticks(np.linspace(0, Ly, 11))

    # ---- PROBE MARKER ----
    if probeX is not None and probeY is not None:

        # compute nearest grid indices (safety clamp)
        probe_i = int(round(probeY / dy))
        probe_j = int(round(probeX / dx))
        probe_i = min(max(0, probe_i), Ny - 1)
        probe_j = min(max(0, probe_j), Nx - 1)

        plt.scatter(
            [probeX],
            [probeY],
            marker="x",
            s=140,
            color="black",
            linewidths=3,
            label=f"Probe (x={probeX:.3f}, y={probeY:.3f})"
        )

        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"plotting/plots/{title}.png")

    return

def probe2D(T_hist, config, probeX=0, probeY=0, prefix=None):

    dx = config["dx"]
    dy = dx
    dt = config["dt"]
    t_steps = config["t_steps"]

    # Convert physical coordinates → grid indices
    probe_j = int(round(probeX / dx))   # x index (columns)
    probe_i = int(round(probeY / dy))   # y index (rows)

    probe_T = []

    for T in T_hist:
        try:
            T = T.get()  # CuPy → NumPy
        except AttributeError:
            T = np.asarray(T)

        probe_T.append(T[probe_i, probe_j])

    probe_T = np.array(probe_T)

    times = np.arange(0, (t_steps * dt) + dt, dt)

    plt.figure(figsize=(10, 6))
    plt.plot(times, probe_T, linewidth=2.5, color="tab:blue",
             label=f"Probe at (x={probeX:.3f}, y={probeY:.3f})")

    if prefix is not None:
        title = f"{prefix} Probe Temperature vs Time (2D Plate)"
    else:
        title = "Probe Temperature vs Time (2D Plate)"

    
    plt.title(title, fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Temperature", fontsize=14)

    # Set limits similar to 1D
    ymin = min(probe_T.min(), config["T"]) * 0.9
    ymax = max(probe_T) * 1.1
    plt.ylim(ymin, ymax)
    plt.yticks(np.linspace(ymin, ymax, 20))

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plotting/plots/{title}.png")

    return
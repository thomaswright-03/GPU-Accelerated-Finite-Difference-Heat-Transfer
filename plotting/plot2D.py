import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import pandas

# =====================================================================
# 2D HEATMAP AND PROBE PLOTTER
# ---------------------------------------------------------------------
# This module provides:
#
#   heatmap2D:
#       • Displays the full 2D temperature distribution of the plate
#       • Uses fixed colour scaling based on min(init_temp, T_final.min)
#         and max(init_temp, T_final.max)
#       • Places a visual probe marker (optional)
#       • Saves the final heatmap as a PNG
#
#   probe2D:
#       • Reads the full temperature history (T_hist)
#       • Extracts the temperature at a specific (x, y) probe location
#       • Generates a Temperature vs Time plot
#       • Saves the figure
#
# These plotting utilities accept both CuPy and NumPy arrays.
# All domain/range conversions are consistent with the solvers.
# =====================================================================


def heatmap2D(T, config, probeX=None, probeY=None, prefix=None):
    """
    Render a 2D temperature field as an imshow heatmap.

    Parameters:
        T       : 2D final temperature array (CuPy or NumPy)
        config  : dictionary containing dx, x_size, t_steps, dt, etc.
        probeX  : optional physical X-location of probe
        probeY  : optional physical Y-location of probe
        prefix  : optional title prefix for saving grouped simulations
    """

    # Convert CuPy → NumPy if needed
    try:
        T = T.get()
    except AttributeError:
        T = np.asarray(T)

    dx = config['dx']
    dy = dx                   # assume square spacing
    Nx = T.shape[1]           # number of columns
    Ny = T.shape[0]           # number of rows

    Lx = config['x_size']
    Ly = Lx

    T0 = config["T"]          # baseline temperature

    # Global colourbar limits for consistency
    T_min = min(T.min(), T0)
    T_max = max(T.max(), T0)

    time = config['t_steps'] * config['dt']

    # Optional title prefix
    if prefix is not None:
        title = f"{prefix} 2D Temperature Distribution of Plate"
    else:
        title = f"2D Temperature Distribution of Plate"

    # ---------------------------------------------------------
    # MAIN 2D HEATMAP
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    plt.imshow(
        T,
        cmap=plt.cm.plasma,       # thermal-style colormap
        extent=[0, Lx, 0, Ly],    # physical dimensions in metres
        origin="lower",
        aspect="equal",           # equal scaling: square cells
        vmin=T_min,
        vmax=T_max
    )

    # Colourbar with label
    cbar = plt.colorbar()
    cbar.set_label("Temperature", fontsize=12)

    # Titles and axis labels
    plt.title(f"{title} after {time:.1f} seconds", fontsize=16)
    plt.xlabel("X position (m)", fontsize=14)
    plt.ylabel("Y position (m)", fontsize=14)

    # Major grid ticks
    plt.xticks(np.linspace(0, Lx, 11))
    plt.yticks(np.linspace(0, Ly, 11))

    # ---------------------------------------------------------
    # OPTIONAL PROBE MARKER
    # ---------------------------------------------------------
    if probeX is not None and probeY is not None:

        # Convert physical location → grid index
        probe_i = int(round(probeY / dy))      # y row index
        probe_j = int(round(probeX / dx))      # x column index

        # Clamp to domain edges (safety)
        probe_i = min(max(0, probe_i), Ny - 1)
        probe_j = min(max(0, probe_j), Nx - 1)

        # Draw visible probe cross
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

    # Save figure to disk
    plt.tight_layout()
    plt.savefig(f"plotting/plots/{title}.png")

    return



def probe2D(T_hist, config, probeX=0, probeY=0, prefix=None):
    """
    Plot the temperature evolution at a fixed (x,y) probe location.

    Parameters:
        T_hist : list of 2D arrays (full time history)
        config : solver configuration dictionary
        probeX : x-position of probe (metres)
        probeY : y-position of probe (metres)
        prefix : optional figure name prefix
    """

    dx = config["dx"]
    dy = dx
    dt = config["dt"]
    t_steps = config["t_steps"]

    # Convert physical probe location → grid indices
    probe_j = int(round(probeX / dx))   # x index
    probe_i = int(round(probeY / dy))   # y index

    probe_T = []

    # ---------------------------------------------------------
    # EXTRACT PROBE TEMPERATURE FOR EACH TIMESTEP
    # ---------------------------------------------------------
    for T in T_hist:
        try:
            T = T.get()  # CuPy → NumPy
        except AttributeError:
            T = np.asarray(T)

        probe_T.append(T[probe_i, probe_j])

    probe_T = np.array(probe_T)

    # Generate matching time axis
    times = np.arange(0, (t_steps * dt) + dt, dt)

    # ---------------------------------------------------------
    # MAIN TIME-SERIES PLOT (Temperature vs Time)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        times, probe_T,
        linewidth=2.5,
        color="tab:blue",
        label=f"Probe at (x={probeX:.3f}, y={probeY:.3f})"
    )

    if prefix is not None:
        title = f"{prefix} Probe Temperature vs Time (2D Plate)"
    else:
        title = "Probe Temperature vs Time (2D Plate)"

    plt.title(title, fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Temperature", fontsize=14)

    # y-axis limits proportional to data range + baseline
    ymin = min(probe_T.min(), config["T"]) * 0.9
    ymax = max(probe_T) * 1.1
    plt.ylim(ymin, ymax)
    plt.yticks(np.linspace(ymin, ymax, 20))

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plotting/plots/{title}.png")

    return

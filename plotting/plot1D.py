import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import pandas

# =====================================================================
# 1D HEAT DISTRIBUTION PLOTTER
# ---------------------------------------------------------------------
# heatmap1D:
#   - Accepts the final temperature array (CuPy or NumPy)
#   - Converts to NumPy if necessary
#   - Builds matching spatial grid from dx, x_size
#   - Produces a line plot of Temperature vs Distance
#   - Optionally marks a probe location
#   - Saves the figure with an optional title prefix
# =====================================================================

def heatmap1D(T, config, probeX=None, prefix=None):
    
    # Handle CuPy → NumPy conversion
    try:
        T = T.get()
    except AttributeError:
        pass

    dx = config['dx']
    x_size = config['x_size']

    # Construct 1D spatial coordinate array
    X = np.arange(0, x_size+dx, dx)

    # Total simulation time for title/annotation
    time = config['t_steps'] * config['dt']

    # -------------------------------
    # MAIN TEMPERATURE LINE PLOT
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(X, T, linewidth=2, color="tab:red", label="Temperature")
    
    # Title handling with optional prefix
    if prefix is not None:
        title = f"{prefix} Temperature vs Distance of 1D 'rod'"
    else:
        title = "Temperature vs Distance of 1D 'rod'"

    # Labels and styling
    plt.title(f"{title} after {time} seconds", fontsize=14)
    plt.xlabel("Distance (m)", fontsize=12)
    plt.ylabel("Temperature", fontsize=12)
    plt.xticks(np.linspace(min(X), max(X), 10))
    
    # Ensure y-axis includes baseline T even if T never reaches it
    ymin = min(T.min(), config['T'])
    ymax = max(T)*1.1
    plt.ylim(ymin, ymax)
    plt.yticks(np.linspace(ymin, ymax, 20))

    # ----------------------------------------------------
    # PROBE MARKER (single point)
    # ----------------------------------------------------
    if probeX is not None:

        # Convert physical coordinate → nearest grid index
        probe_idx = int(round(probeX / dx))
        probe_x_val = probe_idx * dx
        probe_T_val = T[probe_idx]

        # Mark with black "X" and legend entry
        plt.scatter([probe_x_val], [probe_T_val],
                    color="black", s=100, marker="x", linewidths=3,
                    label=f"Probe (x = {probe_x_val:.3f} m)")
    
    # Final styling and save
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plotting/plots/{title}.png")
    return



# =====================================================================
# TEMPERATURE PROBE TIME-SERIES PLOTTER
# ---------------------------------------------------------------------
# probe1D:
#   - Reads the full T_hist list (one entry per timestep)
#   - Extracts the temperature at a specified x-position
#   - Builds a time array from dt, t_steps
#   - Plots Temperature vs Time
#   - Saves the figure using prefix + title
# =====================================================================

def probe1D(T_hist, config, probeX=0, prefix=None):
    
    dx = config['dx']
    dt = config['dt']

    # Convert physical location → grid index
    probe_idx = int(round(probeX / dx))
    probe_T = []

    # Extract probe temperature from each timestep
    for T in T_hist:
        
        # CuPy → NumPy conversion if necessary
        try:
            T = T.get()
        except AttributeError:
            T = np.asarray(T)
        
        probe_T.append(T[probe_idx])

    probe_T = np.array(probe_T)

    # Time array runs from 0 to total_time inclusive
    times = np.arange(0, config['t_steps']*dt+dt, dt)

    # -------------------------------
    # MAIN PROBE TIME-SERIES PLOT
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(times, probe_T, linewidth=2.5,
             color="tab:blue", label="Probe temperature")

    if prefix is not None:
        title = f"{prefix} Probe Temperature vs Time"
    else:
        title = "Probe Temperature vs Time"

    plt.title(f"{title} at x = {probeX:.3f} m", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Temperature", fontsize=14)
    plt.xticks(np.linspace(min(times), max(times), 10))
    
    # Allow baseline temp to appear even if the probe never visits it
    ymin = min(probe_T.min(), config['T'])*0.9
    ymax = max(probe_T)*1.1
    plt.ylim(ymin, ymax)
    plt.yticks(np.linspace(ymin, ymax, 20))

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plotting/plots/{title} for 1D 'rod'.png")

    return

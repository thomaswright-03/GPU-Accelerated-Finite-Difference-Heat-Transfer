import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import pandas

def heatmap1D(T, config, probeX=None, prefix=None):
    
    try:
        T = T.get()
    except AttributeError:
        pass

    dx = config['dx']
    x_size = config['x_size']
    X = np.arange(0, x_size+dx, dx)

    time = config['t_steps'] * config['dt']

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(X, T, linewidth=2, color="tab:red", label="Temperature")
    
    if prefix is not None:
        title = f"{prefix} Temperature vs Distance of 1D 'rod'"
    else:
        title = "Temperature vs Distance of 1D 'rod'"

    # Labels and style
    plt.title(f"{title} after {time} seconds", fontsize=14)
    plt.xlabel("Distance (m)", fontsize=12)
    plt.ylabel("Temperature", fontsize=12)
    plt.xticks(np.linspace(min(X), max(X), 10))
    
    ymin = min(T.min(), config['T'])
    ymax = max(T)*1.1
    plt.ylim(ymin, ymax)
    plt.yticks(np.linspace(ymin, ymax, 20))

    if probeX is not None:
        # Find nearest index
        probe_idx = int(round(probeX / dx))
        probe_x_val = probe_idx * dx
        probe_T_val = T[probe_idx]

        # Add marker (circle) and vertical line
        plt.scatter([probe_x_val], [probe_T_val],
                    color="black", s=100, marker="x", linewidths=3,
                    label=f"Probe (x = {probe_x_val:.3f} m)")
    
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plotting/plots/{title}.png")
    return

def probe1D(T_hist, config, probeX=0, prefix=None):
    
    dx = config['dx']
    dt = config['dt']

    probe_idx = int(round(probeX / dx))
    probe_T = []

    for T in T_hist:
        
        try:
            T = T.get()
        except AttributeError:
            T = np.asarray(T)
        
        probe_T.append(T[probe_idx])

    probe_T = np.array(probe_T)
    times = np.arange(0, config['t_steps']*dt+dt, dt)

    plt.figure(figsize=(10, 6))
    plt.plot(times, probe_T, linewidth=2.5, color="tab:blue", label="Probe temperature")

    if prefix is not None:
        title = f"{prefix} Probe Temperature vs Time"
    else:
        title = "Probe Temperature vs Time"

    plt.title(f"{title} at x = {probeX:.3f} m", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Temperature", fontsize=14)
    plt.xticks(np.linspace(min(times), max(times), 10))
    
    ymin = min(probe_T.min(), config['T'])*0.9
    ymax = max(probe_T)*1.1
    plt.ylim(ymin, ymax)
    plt.yticks(np.linspace(ymin, ymax, 20))

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plotting/plots/{title} for 1D 'rod'.png")

    return
    
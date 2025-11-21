def diffusion_number(config, warn=True):
    #print(f"config.py running...")
###
#    Compute the diffusion number r = alpha * dt / dx^2
#    with optional stability warning for explicit FD heat equation.
#
#    Parameters:
#        alpha : thermal diffusivity (m^2/s)
#        dt    : time step
#        dx    : grid spacing (assume uniform)
#        dims  : number of dimensions (1, 2, or 3)
#        warn  : print stability warning if True
#
#    Returns:
#        r : diffusion number
###
    alpha = config['alpha']
    dt = config['dt']
    dx = config['dx']
    dims = config['dims']

    r = alpha * dt / (dx**2)
    r_max = 1 / (2 * dims)  # Stability limit: 1/(2d)

    if warn and r > r_max:
        print(f"WARNING: Unstable timestep for {dims}D. "
              f"r = {r:.4f} > r_max = {r_max:.4f}")
    
    return r
 
# -----------------------------------------------------------------------------
# Thermal Diffusivity Reference Values (α) 
# Units: m^2/s
#
# α = k / (ρ * c_p)
#
# These values represent how quickly temperature disturbances propagate
# through a material. Higher α → faster heat diffusion.
#
# Source: Common engineering datasets (typical room-temperature values)
#
# α is treated as constant in this project
# -----------------------------------------------------------------------------
# Gases
# Helium:                  1.90e-4   m^2/s
# Hydrogen:                1.60e-4   m^2/s
# Air (0°C):               1.906e-5  m^2/s

# Metals
# Silver:                  1.6563e-4 m^2/s
# Gold:                    1.27e-4   m^2/s
# Aluminum:                9.7e-5    m^2/s
# Steel (304):             4.0e-6    m^2/s
# Copper:                  1.12e-4   m^2/s

# Non-metallic solids & liquids
# Carbon/carbon composite: 2.165e-4  m^2/s
# Glass:                   0.7e-6    m^2/s
# Water:                   1.44e-7   m^2/s
# Concrete:                0.8e-6    m^2/s
# -----------------------------------------------------------------------------

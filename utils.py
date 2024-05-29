import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


@jit
def interp_nearest_2d(angles, positions, h, d0, dd, nd, x0, dx, nx):
    angles_idx = ((angles - d0) / dd).astype("int32").clip(0, nd-1)
    positions_idx = ((positions - x0) / dx).astype("int32").clip(0, nx-1)
    idx_tot = positions_idx + angles_idx * nx
    return h.ravel()[idx_tot].reshape((idx_tot.shape))

@jit
def interp_linear_1d(delays, X, sampling_frequency):
    delays_idx = delays * sampling_frequency
    delays_idx_int = delays_idx.astype("int32").clip(0, X.shape[-1]-2)
    factor = (delays_idx - delays_idx_int).clip(0, 1)
    interpolated = X[delays_idx_int] * (1 - factor)
    interpolated += X[delays_idx_int+1] * factor
    return interpolated

@jit
def das_1tx(X, delays_t, delays_phi, apod, sampling_frequency, demod_freq):
    v_interp_1d = vmap(interp_linear_1d, (0, 0, None), 0)
    X_interp = v_interp_1d(delays_t, X, sampling_frequency) * jnp.exp(1j*2*jnp.pi*demod_freq*delays_phi)
    return (X_interp * apod).sum(axis=0)

@jit
def beamform_intermediary_sa(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
             x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq):
    # Beamform and sum but no coherent compound for SA acquisition
    N_tx = X.shape[0]

    #------- Delays ------#

    # Interpolate \delta\tau^{Tx} on h.
    # (for the SA dataset, \delta\tau^{Tx}=\delta\tau^{Rx})
    delta_tau_t = interp_nearest_2d(jnp.atan((x-probe_x) / z), probe_x,
                                    h_t, d0, dd, nd, x0, dx, nx)
    delta_tau_phi = interp_nearest_2d(jnp.atan((x-probe_x) / z), probe_x,
                                      h_phi, d0, dd, nd, x0, dx, nx)

    # Get tau = tau_tx = tau_rx
    tau = jnp.sqrt(z**2 + (x-probe_x)**2) / 1.54
    tau_t = tau + delta_tau_t
    tau_phi = tau + delta_tau_phi

    #-------- DAS --------#
    # Beamform each emission in a loop
    gamma_alpha = jnp.array([das_1tx(X[i_a], tau_t[i_a] + tau_t,
                                     tau_phi[i_a] + tau_phi, apod_rx,
                                     sampling_frequency, demod_freq) for i_a in range(N_tx)])
    return gamma_alpha
    
@jit
def beamform_intermediary_pw(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
             x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq, angles):
    # Beamform and sum but no coherent compound for PW acquisition
    N_tx = X.shape[0]

    #------- Delays ------#

    # Interpolate \delta\tau^{Tx} on h.
    delta_tau_tx_t = interp_nearest_2d(angles, x - z*jnp.tan(angles),
                                    h_t, d0, dd, nd, x0, dx, nx)
    delta_tau_tx_phi = interp_nearest_2d(angles, x - z*jnp.tan(angles),
                                      h_phi, d0, dd, nd, x0, dx, nx)
    delta_tau_rx_t = interp_nearest_2d(jnp.atan((x-probe_x) / z), probe_x,
                                    h_t, d0, dd, nd, x0, dx, nx)
    delta_tau_rx_phi = interp_nearest_2d(jnp.atan((x-probe_x) / z), probe_x,
                                      h_phi, d0, dd, nd, x0, dx, nx)

    # Get tau_tx and tau_rx
    tau_tx = (z*jnp.cos(angles) + (x - probe_x[0]*jnp.sign(angles))*jnp.sin(angles)) / 1.54
    tau_tx_t = tau_tx + delta_tau_tx_t
    tau_tx_phi = tau_tx + delta_tau_tx_phi
    tau_rx = jnp.sqrt(z**2 + (x-probe_x)**2) / 1.54
    tau_rx_t = tau_rx + delta_tau_rx_t
    tau_rx_phi = tau_rx + delta_tau_rx_phi

    #-------- DAS --------#
    # Beamform each emission in a loop
    gamma_alpha = jnp.array([das_1tx(X[i_a], tau_tx_t[i_a] + tau_rx_t,
                                     tau_tx_phi[i_a] + tau_rx_phi, apod_rx,
                                     sampling_frequency, demod_freq) for i_a in range(N_tx)])
    return gamma_alpha
    
@jit
def beamform_sa(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
             x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq):
    """
    - X input                         (Tx, Rx, t)
    - h_t aberration grid, time       (Nd, Nx)
    - h_phi aberration grid, phase    (Nd, Nx)
    - d0, dd, nd start, step and number of directions in h
    - x0, dx, nx start, step and number of positions in h
    - probe_x transducer locations    (Rx, 1)
    - x lateral positions of targets  (1, Np)
    - z axial positions of targets    (1, Np)
    - eps_d directions of h           (Nd, 1)
    - eps_x locations of h            (1, Nx)
    - apod_tx transmit apodization    (Tx, Np)
    - apod_rx receive apodization     (Rx, Np)
    
    Returns:
    - gamma beamformed image          (Np)
    """
    gamma_alpha = beamform_intermediary_sa(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
             x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq)

    # Coherent Compound
    gamma = (apod_tx * gamma_alpha).sum(axis=0)

    return gamma

@jit
def beamform_pw(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
             x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq, angles):
    """
    - X input                         (Tx, Rx, t)
    - h_t aberration grid, time       (Nd, Nx)
    - h_phi aberration grid, phase    (Nd, Nx)
    - d0, dd, nd start, step and number of directions in h
    - x0, dx, nx start, step and number of positions in h
    - probe_x transducer locations    (Rx, 1)
    - x lateral positions of targets  (1, Np)
    - z axial positions of targets    (1, Np)
    - eps_d directions of h           (Nd, 1)
    - eps_x locations of h            (1, Nx)
    - apod_tx transmit apodization    (Tx, Np)
    - apod_rx receive apodization     (Rx, Np)
    - angles Tx angles                (Tx, 1)
    
    Returns:
    - gamma beamformed image          (Np)
    """
    gamma_alpha = beamform_intermediary_pw(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
             x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq, angles)

    # Coherent Compound
    gamma = (apod_tx * gamma_alpha).sum(axis=0)

    return gamma


@jit
def coherence_sa(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
              x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq):
    """
    - X input                         (Tx, Rx, t)
    - h_t aberration grid, time       (Nd, Nx)
    - h_phi aberration grid, phase    (Nd, Nx)
    - d0, dd, nd start, step and number of directions in h
    - x0, dx, nx start, step and number of positions in h
    - probe_x transducer locations    (Rx, 1)
    - x lateral positions of targets  (1, Np)
    - z axial positions of targets    (1, Np)
    - eps_d directions of h           (Nd, 1)
    - eps_x locations of h            (1, Nx)
    - apod_tx transmit apodization    (Tx, Np)
    - apod_rx receive apodization     (Rx, Np)
    
    Returns:
    - -C opposite of coherence        (1)
    """
    N_tx = X.shape[0]
    
    gamma_alpha = beamform_intermediary_sa(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
             x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq)

    # Coherent Compound
    gamma = (apod_tx * gamma_alpha).sum(axis=0)
    
    gg = gamma * gamma.conj()
    #----- Coherence -----#
    C = 1 / N_tx * jnp.abs( gg / 
        (apod_tx * apod_tx * gamma_alpha * gamma_alpha.conj()).sum(axis=0)).mean()
    
    return -C

@jit
def coherence_pw(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
              x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq, angles):
    """
    - X input                         (Tx, Rx, t)
    - h_t aberration grid, time       (Nd, Nx)
    - h_phi aberration grid, phase    (Nd, Nx)
    - d0, dd, nd start, step and number of directions in h
    - x0, dx, nx start, step and number of positions in h
    - probe_x transducer locations    (Rx, 1)
    - x lateral positions of targets  (1, Np)
    - z axial positions of targets    (1, Np)
    - eps_d directions of h           (Nd, 1)
    - eps_x locations of h            (1, Nx)
    - apod_tx transmit apodization    (Tx, Np)
    - apod_rx receive apodization     (Rx, Np)
    - angles Tx angles                (Tx, 1)
    
    Returns:
    - -C opposite of coherence        (1)
    """
    N_tx = X.shape[0]
    
    gamma_alpha = beamform_intermediary_pw(X, h_t, h_phi, d0, dd, nd, x0, dx, nx, probe_x,
             x, z, eps_d, eps_x, apod_tx, apod_rx,
             sampling_frequency, demod_freq, angles)

    # Coherent Compound
    gamma = (apod_tx * gamma_alpha).sum(axis=0)
    
    gg = gamma * gamma.conj()
    #----- Coherence -----#
    C = 1 / N_tx * jnp.abs( gg / 
        (apod_tx * apod_tx * gamma_alpha * gamma_alpha.conj()).sum(axis=0)).mean()
    
    return -C

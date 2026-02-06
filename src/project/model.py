"""Neural network model for PINN."""

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config


def init_nn_params(
    cfg: Config, key: jnp.ndarray | None = None, seed: int | None = None
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Initialize network parameters and learnable scalars."""
    if key is None:
        key = jax.random.key(cfg.seed if seed is None else seed)

    layers = cfg.layer_sizes

    # Split keys
    key, nn_key = jax.random.split(key, 2)

    # ---- Neural network parameters ----
    nn_keys = jax.random.split(nn_key, len(layers) - 1)
    nn_params = []

    for k, (din, dout) in zip(nn_keys, zip(layers[:-1], layers[1:])):
        w_key, _ = jax.random.split(k)
        scale = jnp.sqrt(2.0 / (din + dout))
        w = jax.random.normal(w_key, (din, dout)) * scale
        b = jnp.zeros(dout)
        nn_params.append((w, b))

    return nn_params


def init_pinn_params(cfg: Config, seed: int | None = None):
    """Initialize network parameters and learnable scalars."""
    key = jax.random.key(cfg.seed if seed is None else seed)
    key, nn_key, scalars_key = jax.random.split(key, 3)

    #######################################################################
    # Oppgave 5.1: Start
    #######################################################################

    k1, k2, k3, k4 = jax.random.split(scalars_key, 4) # Splitter opp keys slik at ikke alle parametrene får initialverdi
    
    nn_params = init_nn_params(cfg,nn_key) # Definerer parametre for NN ut i fra init_nn_params()
    pinn_params = {"nn": nn_params,
                   "log_alpha" : jax.random.normal(k1, (1,)),
                   "log_k" : jax.random.normal(k2, (1,)),
                   "log_h" :jax.random.normal(k3, (1,)),
                   "log_power" :jax.random.normal(k4, (1,))} # Definerer parametre som en dictionnary, med nn som parametrene fra det nevrale nettverket, og de fysiske parameterne med logaritmisk skala for at de skal være positive 


    #######################################################################
    # Oppgave 5.1: Slutt
    #######################################################################

    return pinn_params


def forward(
    nn_params: list[tuple[jnp.ndarray, jnp.ndarray]],
    x: Union[jnp.ndarray, float],
    y: Union[jnp.ndarray, float],
    t: Union[jnp.ndarray, float],
    cfg: Config,
) -> jnp.ndarray:
    """Forward pass through the network.

    Args:
        params: Network parameters (just the list of (w, b) tuples)
        x, y, t: Input coordinates (scalars or arrays). Can mix scalars and arrays,
                 e.g., forward(params, x_array, y_array, t_scalar, cfg) will work.
        cfg: Configuration for normalization bounds

    Returns:
        Temperature prediction (scalar or array of shape (N,))
    """
    # Convert inputs to arrays and broadcast to common shape
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
    t = jnp.atleast_1d(t)

    # Broadcast all to the same shape
    x, y, t = jnp.broadcast_arrays(x, y, t)

    #######################################################################
    # Oppgave 4.1: Start
    #######################################################################

    #Normaliserer inputs til NN slik at verdiene ligger i intervallet [0,1].
    x_norm = (x - cfg.x_min) / (cfg.x_max - cfg.x_min)
    y_norm = (y - cfg.y_min) / (cfg.y_max - cfg.y_min)
    t_norm = (t - cfg.t_min) / (cfg.t_max - cfg.t_min)

    #Slår sammen normaliserte variablene til en input-vektor
    a = jnp.stack([x_norm, y_norm, t_norm], axis = -1)

    #Itererer gjennom skjulte lag, lærer ikke-lineær sammenheng mellom x, y, og t
    for w,b in nn_params[:-1]: # Skjulte lag
        a = jnp.tanh(a @ w + b) # Legger på ikke-linearitet

    #Outputlag som gir temperaturprediksjon med lineær tranformasjon
    w_out,b_out = nn_params[-1] # Output-lag
    a = a @ w_out + b_out

    #fjerner dimensjoner med størrelse 1
    out = a.squeeze()

    #######################################################################
    # Oppgave 4.1: Slutt
    #######################################################################

    return out


def predict_grid(
    nn_params: list[tuple[jnp.ndarray, jnp.ndarray]],
    x: jnp.ndarray | np.ndarray,
    y: jnp.ndarray | np.ndarray,
    t: jnp.ndarray | np.ndarray,
    cfg: Config,
) -> jnp.ndarray:
    """Predict temperature on full spatiotemporal grid.

    Args:
        nn_params: Network parameters (just the list of (w, b) tuples)
        x, y: Spatial coordinates (nx,), (ny,)
        t: Time points (nt,)
        cfg: Configuration

    Returns:
        T_pred: Predictions (nt, nx, ny)
    """

    nt, nx, ny = len(t), len(x), len(y)
    T_pred = jnp.zeros((nt, nx, ny))

    # Create meshgrid for spatial coordinates
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Flatten to 1D arrays
    x_flat = X.ravel()
    y_flat = Y.ravel()

    for n in range(nt):
        # Vectorized forward pass (t[n] is broadcast automatically)
        T_flat = forward(nn_params, x_flat, y_flat, t[n], cfg)
        # Reshape back to grid
        T_pred_t = T_flat.reshape(nx, ny)
        T_pred = T_pred.at[n].set(T_pred_t)

    return T_pred

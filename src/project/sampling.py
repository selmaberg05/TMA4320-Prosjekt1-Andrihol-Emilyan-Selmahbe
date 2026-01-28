"""Sampling utilities for collocation, initial, and boundary points."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from .config import Config


def _sample_uniform(key, n, mins, maxs):
    """Sample n points uniformly in a box."""
    mins, maxs = jnp.asarray(mins), jnp.asarray(maxs)
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, (n, mins.size))
    return mins + u * (maxs - mins), key


def _sample_qmc(n: int, mins: list, maxs: list, seed: int = 0) -> jnp.ndarray:
    """Sample n points using Quasi-Monte Carlo (Sobol sequence).

    QMC provides better space-filling properties than uniform random sampling,
    which can improve convergence for PINN training.

    Args:
        n: Number of points to sample
        mins: Lower bounds for each dimension
        maxs: Upper bounds for each dimension
        seed: Random seed for scrambling (for reproducibility)

    Returns:
        Array of shape (n, d) with sampled points
    """
    d = len(mins)
    mins, maxs = np.asarray(mins), np.asarray(maxs)

    # Use Sobol sequence with scrambling for randomization
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    samples = sampler.random(n)

    # Scale from [0,1]^d to [mins, maxs]
    samples = qmc.scale(samples, mins, maxs)

    return jnp.array(samples)


def sample_interior(key, cfg: Config):
    """Sample collocation points in the interior using QMC."""
    key, subkey = jax.random.split(key)
    seed = int(jax.random.key_data(subkey)[0])
    sampled = _sample_qmc(
        cfg.num_collocation,
        mins=[cfg.x_min, cfg.y_min, cfg.t_min],
        maxs=[cfg.x_max, cfg.y_max, cfg.t_max],
        seed=seed,
    )
    return sampled, key


def sample_ic(key, cfg: Config):
    """Sample initial condition points (t=0) using QMC."""

    key, subkey = jax.random.split(key)
    seed = int(jax.random.key_data(subkey)[0])

    pts = _sample_qmc(
        cfg.num_ic,
        mins=[cfg.x_min, cfg.y_min],
        maxs=[cfg.x_max, cfg.y_max],
        seed=seed,
    )
    t0 = jnp.zeros((cfg.num_ic, 1))
    return jnp.concatenate([pts, t0], axis=1), key


def sample_bc(key, cfg: Config):
    """Sample boundary points with outward normals [x, y, t, nx, ny]."""
    n = cfg.num_bc

    # Edge lengths
    L_x = cfg.x_max - cfg.x_min
    L_y = cfg.y_max - cfg.y_min
    perimeter = 2 * (L_x + L_y)

    # Sample edge index (0=left, 1=right, 2=bottom, 3=top)
    # Weight by edge length so each unit of boundary is sampled equally
    edge_probs = jnp.array([L_y, L_y, L_x, L_x]) / perimeter
    key, subkey = jax.random.split(key)
    edge = jax.random.choice(subkey, 4, shape=(n,), p=edge_probs)

    # Sample position along edge [0, 1] and time
    key, subkey = jax.random.split(key)
    s = jax.random.uniform(subkey, (n,))
    key, subkey = jax.random.split(key)
    t = cfg.t_min + jax.random.uniform(subkey, (n,)) * (cfg.t_max - cfg.t_min)

    # Set x, y based on edge (left/right fix x, bottom/top fix y)
    x = jnp.where(
        edge == 0,
        cfg.x_min,
        jnp.where(edge == 1, cfg.x_max, cfg.x_min + s * (cfg.x_max - cfg.x_min)),
    )

    y = jnp.where(
        edge < 2,
        cfg.y_min + s * (cfg.y_max - cfg.y_min),
        jnp.where(edge == 2, cfg.y_min, cfg.y_max),
    )

    # Set normals
    nx = jnp.where(edge == 0, -1.0, jnp.where(edge == 1, 1.0, 0.0))
    ny = jnp.where(edge == 2, -1.0, jnp.where(edge == 3, 1.0, 0.0))

    return jnp.stack([x, y, t, nx, ny], axis=1), key

"""Adam optimizer implementation."""

import jax
import jax.numpy as jnp


def init_adam(params):
    """Initialize Adam optimizer state.

    Args:
        params: Model parameters (pytree)

    Returns:
        Optimizer state dict with moment estimates and step count
    """
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": m, "v": v, "t": 0}


def adam_step(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Perform one Adam optimization step.

    Args:
        params: Current parameters
        grads: Gradients
        state: Optimizer state
        lr: Learning rate
        b1: First moment decay rate
        b2: Second moment decay rate
        eps: Numerical stability constant

    Returns:
        new_params: Updated parameters
        new_state: Updated optimizer state
    """
    t = state["t"] + 1

    # Update moment estimates
    new_m = jax.tree_util.tree_map(
        lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads
    )
    new_v = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1 - b2) * g**2, state["v"], grads
    )

    # Bias correction
    m_corr = 1 - b1**t
    v_corr = 1 - b2**t

    # Parameter update
    def update(p, m, v):
        m_hat = m / m_corr
        v_hat = v / v_corr
        return p - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    new_params = jax.tree_util.tree_map(update, params, new_m, new_v)
    new_state = {"m": new_m, "v": new_v, "t": t}

    return new_params, new_state

"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################

    # Mye her er også hentet fra oppgaveteksten, og jax funksjonene står det om i jax_intro og på nett

    # Update the nn_params and losses dictionary

    def objective_fn(nn_params):
        L_data = data_loss(nn_params, sensor_data, cfg)
        L_ic = ic_loss(nn_params, ic_epoch, cfg)
        total_loss = cfg.lambda_data * L_data + cfg.lambda_ic * L_ic
        return total_loss, (L_data, L_ic)
    
    value_and_grad = jax.value_and_grad(objective_fn, has_aux=True) 

    from tqdm import tqdm
    for _ in tqdm(range(cfg.num_epochs), desc="Training NN"):
        ic_epoch, key = sample_ic(key, cfg)
        (total_loss, (L_data, L_ic)), grads = value_and_grad(nn_params) 

        nn_params, adam_state = adam_step(nn_params, grads, adam_state, lr=cfg.learning_rate)

        losses["total"].append(total_loss)
        losses["data"].append(L_data)
        losses["ic"].append(L_ic)
    


    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################
    
    
    def objective_fn(pinn_params, interior_epoch, ic_epoch, bc_epoch):
        nn_params = pinn_params["nn"]
        L_data = data_loss(nn_params, sensor_data, cfg)
        L_physics = physics_loss(pinn_params, interior_epoch, cfg)
        L_ic = ic_loss(nn_params, ic_epoch, cfg)
        L_bc = bc_loss(pinn_params,bc_epoch,cfg)
        total_loss = cfg.lambda_data * L_data + cfg.lambda_physics * L_physics +cfg.lambda_ic * L_ic +cfg.lambda_bc * L_bc
        return total_loss, (L_data,L_physics, L_ic, L_bc)
    
    value_and_grad = jax.jit(jax.value_and_grad(objective_fn, has_aux=True))

    from tqdm import tqdm
    for _ in tqdm(range(cfg.num_epochs), desc="Training PINN"):
        interior_epoch, key = sample_interior(key, cfg)
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)
        (total_loss, (L_data, L_physics, L_ic, L_bc)), grads = value_and_grad(pinn_params,interior_epoch, ic_epoch, bc_epoch) 

        pinn_params, opt_state = adam_step(pinn_params, grads, opt_state, lr=cfg.learning_rate)

        losses["total"].append(total_loss)
        losses["data"].append(L_data)
        losses["physics"].append(L_physics)
        losses["ic"].append(L_ic)
        losses["bc"].append(L_bc)

        
    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}

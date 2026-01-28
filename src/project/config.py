"""Configuration loader for PINN project."""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import yaml


@dataclass
class Config:
    """Configuration for the PINN simulation."""

    # Domain
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float

    # Physics
    alpha: float
    k: float
    h: float
    T_outside: float

    # Source
    source_locations: jnp.ndarray
    source_sizes: jnp.ndarray
    source_strength: float

    # Grid
    nx: int
    ny: int
    nt: int

    # Sensors
    sensor_rate: float
    sensor_noise: float
    sensor_locations: jnp.ndarray

    # Training
    layer_sizes: list
    learning_rate: float
    num_epochs: int
    seed: int
    lambda_physics: float
    lambda_ic: float
    lambda_bc: float
    lambda_data: float
    num_collocation: int
    num_ic: int
    num_bc: int

    def is_source(self, x, y):
        """Check if point(s) are inside any heat source."""
        # source_locations: (S, 2), source_sizes: (S,)
        cx = self.source_locations[:, 0]  # (S,)
        cy = self.source_locations[:, 1]  # (S,)
        sizes = self.source_sizes  # (S,)

        # Broadcast x, y against source centers
        # x, y can be scalars or arrays of any shape
        dx = jnp.abs(x - cx[:, None, None])  # (S, ...) broadcasts with x
        dy = jnp.abs(y - cy[:, None, None])  # (S, ...) broadcasts with y

        inside = (dx <= sizes[:, None, None]) & (dy <= sizes[:, None, None])
        return jnp.any(inside, axis=0)  # same shape as x, y

    def heat_source(self, x, y, t):
        """Heat source term at point (x, y, t)."""
        return jnp.where(self.is_source(x, y), self.source_strength, 0.0)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(
        # Domain
        x_min=data["domain"]["x_min"],
        x_max=data["domain"]["x_max"],
        y_min=data["domain"]["y_min"],
        y_max=data["domain"]["y_max"],
        t_min=data["time"]["t_min"],
        t_max=data["time"]["t_max"],
        # Physics
        alpha=data["physics"]["alpha"],
        k=data["physics"]["k"],
        h=data["physics"]["h"],
        T_outside=data["physics"]["T_outside"],
        # Source
        source_locations=jnp.asarray(
            data["source"]["locations"],
        ),
        source_sizes=jnp.asarray(
            data["source"]["sizes"],
        ),
        source_strength=data["source"]["strength"],
        # Grid
        nx=data["grid"]["nx"],
        ny=data["grid"]["ny"],
        nt=data["grid"]["nt"],
        # Sensors
        sensor_rate=data["sensors"]["measure_rate"],
        sensor_noise=data["sensors"]["noise_std"],
        sensor_locations=jnp.asarray(
            data["sensors"]["locations"],
        ),  # shape (n_sensors, 2)
        # Training
        layer_sizes=data["training"]["layer_sizes"],
        learning_rate=data["training"]["learning_rate"],
        num_epochs=data["training"]["num_epochs"],
        seed=data["training"]["seed"],
        lambda_physics=data["training"]["lambda_physics"],
        lambda_ic=data["training"]["lambda_ic"],
        lambda_bc=data["training"]["lambda_bc"],
        lambda_data=data["training"]["lambda_data"],
        num_collocation=data["training"]["num_collocation"],
        num_ic=data["training"]["num_ic"],
        num_bc=data["training"]["num_bc"],
    )

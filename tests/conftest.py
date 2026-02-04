"""Pytest fixtures for PINN project tests."""

import jax
import pytest

from project.config import Config


@pytest.fixture
def cfg():
    """Simple test configuration with small grid for fast tests."""
    return Config(
        # Domain
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=5.0,
        t_min=0.0,
        t_max=24.0,
        # Physics
        alpha=0.15,
        k=0.026,
        h=5.0,
        T_outside=0.0,
        # Source
        source_locations=jax.numpy.array([[5.0, 2.5]]),
        source_sizes=jax.numpy.array([0.5]),
        source_strength=5.0,
        # Grid (small for fast tests)
        nx=10,
        ny=10,
        nt=10,
        # Sensors
        sensor_rate=1.0,
        sensor_noise=0.0,  # No noise for deterministic tests
        sensor_locations=jax.numpy.array([[5.0, 2.5], [0.0, 0.0]]),
        # Training (small for fast tests)
        layer_sizes=[3, 8, 8, 1],
        learning_rate=0.001,
        num_epochs=10,
        seed=42,
        lambda_physics=1.0,
        lambda_ic=1.0,
        lambda_bc=1.0,
        lambda_data=1.0,
        num_collocation=64,
        num_ic=16,
        num_bc=16,
    )


@pytest.fixture
def nn_params(cfg):
    """Initialize neural network parameters."""
    from project.model import init_nn_params

    return init_nn_params(cfg, seed=42)


@pytest.fixture
def pinn_params(cfg):
    """Initialize PINN parameters (NN + scalars)."""
    from project.model import init_pinn_params

    return init_pinn_params(cfg, seed=42)


@pytest.fixture
def sensor_data(cfg):
    """Generate sensor data from FDM solution."""
    from project.data import generate_training_data

    _, _, _, _, sensor_data = generate_training_data(cfg)
    return sensor_data


@pytest.fixture
def ic_points(cfg):
    """Sample initial condition points."""
    key = jax.random.key(42)
    from project.sampling import sample_ic

    ic_pts, _ = sample_ic(key, cfg)
    return ic_pts


@pytest.fixture
def colloc_points(cfg):
    """Sample collocation points in the interior."""
    key = jax.random.key(42)
    from project.sampling import sample_interior

    colloc, _ = sample_interior(key, cfg)
    return colloc


@pytest.fixture
def bc_points(cfg):
    """Sample boundary condition points."""
    key = jax.random.key(42)
    from project.sampling import sample_bc

    bc, _ = sample_bc(key, cfg)
    return bc

"""Tests for data.py."""

import jax.numpy as jnp
import numpy as np

from project.data import generate_training_data


def test_returns_all_outputs(cfg):
    """Check that all five return values are not None."""
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    assert x is not None, (
        "x is None. Did you forget to call solve_heat_equation() "
        "and assign its outputs?"
    )
    assert y is not None, (
        "y is None. Did you forget to call solve_heat_equation() "
        "and assign its outputs?"
    )
    assert t is not None, (
        "t is None. Did you forget to call solve_heat_equation() "
        "and assign its outputs?"
    )
    assert T_fdm is not None, (
        "T_fdm is None. Did you forget to call solve_heat_equation() "
        "and assign its outputs?"
    )
    assert sensor_data is not None, (
        "sensor_data is None. Did you forget to call _generate_sensor_data() "
        "and assign its output?"
    )


def test_fdm_solution_shapes(cfg):
    """Check that FDM outputs have correct shapes."""
    x, y, t, T_fdm, _ = generate_training_data(cfg)

    assert x.shape == (cfg.nx,), (
        f"x should have shape ({cfg.nx},), got {x.shape}. "
        "Make sure you're returning the x array from solve_heat_equation."
    )
    assert y.shape == (cfg.ny,), (
        f"y should have shape ({cfg.ny},), got {y.shape}. "
        "Make sure you're returning the y array from solve_heat_equation."
    )
    assert t.shape == (cfg.nt,), (
        f"t should have shape ({cfg.nt},), got {t.shape}. "
        "Make sure you're returning the t array from solve_heat_equation."
    )
    assert T_fdm.shape == (cfg.nt, cfg.nx, cfg.ny), (
        f"T_fdm should have shape ({cfg.nt}, {cfg.nx}, {cfg.ny}), got {T_fdm.shape}."
    )


def test_sensor_data_shape(cfg):
    """Check that sensor_data has correct shape (N, 4)."""
    _, _, _, _, sensor_data = generate_training_data(cfg)

    assert len(sensor_data.shape) == 2, (
        f"sensor_data should be 2D array, got shape {sensor_data.shape}. "
        "Sensor data should be an array of [x, y, t, T] measurements."
    )
    assert sensor_data.shape[1] == 4, (
        f"sensor_data should have 4 columns [x, y, t, T], got {sensor_data.shape[1]}. "
        "Each row should contain [x_coord, y_coord, time, temperature]."
    )


def test_sensor_data_not_empty(cfg):
    """Check that sensor data contains measurements."""
    _, _, _, _, sensor_data = generate_training_data(cfg)

    assert sensor_data.shape[0] > 0, (
        "sensor_data is empty (no measurements). "
        "Make sure _generate_sensor_data is called and returns data."
    )


def test_sensor_coordinates_within_domain(cfg):
    """Check that all sensor measurements are within the domain."""
    _, _, _, _, sensor_data = generate_training_data(cfg)

    x_sensors = np.array(sensor_data[:, 0])
    y_sensors = np.array(sensor_data[:, 1])
    t_sensors = np.array(sensor_data[:, 2])

    assert np.all(x_sensors >= cfg.x_min) and np.all(x_sensors <= cfg.x_max), (
        f"Some x coordinates are outside domain [{cfg.x_min}, {cfg.x_max}]. "
        f"Got range [{x_sensors.min()}, {x_sensors.max()}]."
    )
    assert np.all(y_sensors >= cfg.y_min) and np.all(y_sensors <= cfg.y_max), (
        f"Some y coordinates are outside domain [{cfg.y_min}, {cfg.y_max}]. "
        f"Got range [{y_sensors.min()}, {y_sensors.max()}]."
    )
    assert np.all(t_sensors >= cfg.t_min) and np.all(t_sensors <= cfg.t_max), (
        f"Some time values are outside domain [{cfg.t_min}, {cfg.t_max}]. "
        f"Got range [{t_sensors.min()}, {t_sensors.max()}]."
    )


def test_sensor_data_is_jax_array(cfg):
    """Check that sensor_data is returned as a JAX array."""
    _, _, _, _, sensor_data = generate_training_data(cfg)

    # Check it's a JAX array (for compatibility with training)
    assert isinstance(sensor_data, jnp.ndarray), (
        f"sensor_data should be a JAX array (jnp.ndarray), got {type(sensor_data)}. "
        "The _generate_sensor_data function should return jnp.array(...)."
    )

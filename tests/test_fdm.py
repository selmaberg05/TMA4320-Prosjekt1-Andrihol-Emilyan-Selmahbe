"""Tests for fdm.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from project.config import Config
from project.fdm import solve_heat_equation


@pytest.fixture
def fdm_cfg():
    """Configuration with finer grid to capture heat source properly."""
    return Config(
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=5.0,
        t_min=0.0,
        t_max=24.0,
        alpha=0.15,
        k=0.026,
        h=5.0,
        T_outside=0.0,
        source_locations=jnp.array([[5.0, 2.5]]),
        source_sizes=jnp.array([0.5]),
        source_strength=5.0,
        # Finer grid to ensure heat source is captured
        nx=25,
        ny=15,
        nt=15,
        sensor_rate=1.0,
        sensor_noise=0.0,
        sensor_locations=jnp.array([[5.0, 2.5]]),
        layer_sizes=[3, 8, 8, 1],
        learning_rate=0.001,
        num_epochs=10,
        seed=42,
        lambda_physics=1.0,
        lambda_ic=1.0,
        lambda_bc=1.0,
        lambda_data=1.0,
        num_collocation=50,
        num_ic=20,
        num_bc=20,
    )


"""Tests for the FDM heat equation solver (Oppgave 3.2)."""


def test_returns_correct_types(cfg):
    """Check that the function returns numpy arrays."""
    x, y, t, T = solve_heat_equation(cfg)

    assert isinstance(x, np.ndarray), (
        "x should be a numpy array. Did you forget to return the coordinate array?"
    )
    assert isinstance(y, np.ndarray), (
        "y should be a numpy array. Did you forget to return the coordinate array?"
    )
    assert isinstance(t, np.ndarray), (
        "t should be a numpy array. Did you forget to return the time array?"
    )
    assert isinstance(T, np.ndarray), (
        "T should be a numpy array. Did you forget to return the temperature solution?"
    )


def test_output_shapes(cfg):
    """Check that output arrays have correct shapes."""
    x, y, t, T = solve_heat_equation(cfg)

    assert x.shape == (cfg.nx,), (
        f"x should have shape ({cfg.nx},), got {x.shape}. "
        "Make sure x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)."
    )
    assert y.shape == (cfg.ny,), (
        f"y should have shape ({cfg.ny},), got {y.shape}. "
        "Make sure y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)."
    )
    assert t.shape == (cfg.nt,), (
        f"t should have shape ({cfg.nt},), got {t.shape}. "
        "Make sure t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)."
    )
    assert T.shape == (cfg.nt, cfg.nx, cfg.ny), (
        f"T should have shape (nt={cfg.nt}, nx={cfg.nx}, ny={cfg.ny}), "
        f"got {T.shape}. "
        "Make sure T is indexed as T[time_index, x_index, y_index]."
    )


def test_initial_condition(cfg):
    """Check that T(x, y, 0) = T_outside at all spatial points."""
    _, _, _, T = solve_heat_equation(cfg)

    T_initial = T[0, :, :]
    expected = cfg.T_outside

    assert np.allclose(T_initial, expected), (
        f"Initial condition should be T_outside={expected} everywhere. "
        f"Got min={T_initial.min():.4f}, max={T_initial.max():.4f}. "
        "Make sure you set T[0, :, :] = cfg.T_outside (or just T[0] = cfg.T_outside)."
    )


def test_temperature_increases_with_heat_source(fdm_cfg):
    """Check that temperature increases over time due to heat source."""
    x, y, _, T = solve_heat_equation(fdm_cfg)

    # Verify the point is inside the source is heating
    for sensor_location in fdm_cfg.sensor_locations:
        cx, cy = sensor_location
        # Find grid indices closest to sensor location
        i_center = np.argmin(np.abs(x - cx))
        j_center = np.argmin(np.abs(y - cy))

        # Verify the point is inside the source
        x_pt, y_pt = x[i_center], y[j_center]
        in_source = (abs(x_pt - cx) <= fdm_cfg.source_sizes[0]) and (
            abs(y_pt - cy) <= fdm_cfg.source_sizes[0]
        )

        # Temperature at source location should increase
        T_center_initial = T[0, i_center, j_center]
        T_center_final = T[-1, i_center, j_center]

        if in_source and fdm_cfg.source_strength > 0:
            assert T_center_final > T_center_initial, (
                f"Temperature at heat source should increase over time. "
                f"Initial: {T_center_initial:.4f}, Final: {T_center_final:.4f}. "
                f"(Source at ({cx}, {cy}), grid point ({x_pt:.2f}, {y_pt:.2f}), "
                f"in source: {in_source}). "
                "Make sure you are solving the system A * T_new = rhs at each time step "
                "and storing the result in T[n+1]."
            )


def test_no_nan_or_inf(cfg):
    """Check that solution contains no NaN or Inf values."""
    _, _, _, T = solve_heat_equation(cfg)

    assert not np.any(np.isnan(T)), (
        "Solution contains NaN values. "
        "This typically happens when the linear system is singular. "
        "Check that the matrix A is built correctly."
    )
    assert not np.all(np.isinf(T)), (
        "Solution contains Inf values. "
        "This typically happens when dividing by zero or numerical overflow."
    )


def test_solution_bounded(cfg):
    """Check that temperature stays within physically reasonable bounds."""
    _, _, _, T = solve_heat_equation(cfg)

    # Temperature should stay bounded (not explode)
    # With T_outside=0 and a heat source, T should stay positive but not too large
    assert T.min() >= cfg.T_outside - 1, (
        f"Temperature fell below T_outside={cfg.T_outside} significantly. "
        f"Min temperature: {T.min():.4f}. This suggests a sign error in the solver."
    )
    assert T.max() < 1000, (
        f"Temperature exploded to {T.max():.4f}. "
        "This suggests numerical instability in the time stepping."
    )


def test_time_stepping_uses_implicit_euler(cfg):
    """Verify that the solution evolves smoothly (implicit Euler is stable)."""
    _, _, _, T = solve_heat_equation(cfg)

    # Check that temperature changes smoothly between time steps
    # (implicit Euler shouldn't have wild oscillations)
    for n in range(1, cfg.nt):
        max_change = np.abs(T[n] - T[n - 1]).max()
        assert max_change < 50, (
            f"Temperature changed by {max_change:.4f} between steps {n - 1} and {n}. "
            "This suggests the solver may not be using implicit Euler correctly. "
            "Remember: you should solve A * T_new = rhs, not T_new = A^{-1} * T_old."
        )


def test_implicit_euler_uses_next_time_for_source():
    """Verify that _build_rhs is called with t[n+1], not t[n] (implicit Euler).

    For implicit Euler, the heat source should be evaluated at the NEW time t[n+1],
    not the old time t[n]. This test uses a time-dependent heat source that turns
    on at a specific time to verify correct behavior.
    """
    import jax.numpy as jnp

    # Create a config subclass with time-dependent heat source
    class TimeDependentConfig(Config):
        def __init__(self, turn_on_time, **kwargs):
            super().__init__(**kwargs)
            self._turn_on_time = turn_on_time

        def heat_source(self, x, y, t):
            # Source only active after turn_on_time
            is_active = t >= self._turn_on_time
            spatial_mask = self.is_source(x, y)
            return jnp.where(is_active & spatial_mask, self.source_strength, 0.0)

    # Create config where source turns on at t=12
    # With t going from 0 to 24 in 5 steps: t = [0, 6, 12, 18, 24]
    # dt = 6, so:
    #   - Step 0->1: solving for T[1] at t=6, source should be OFF (t=6 < 12)
    #   - Step 1->2: solving for T[2] at t=12, source should be ON (t=12 >= 12)
    #   - Step 2->3: solving for T[3] at t=18, source should be ON
    turn_on_time = 12.0
    cfg = TimeDependentConfig(
        turn_on_time=turn_on_time,
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=5.0,
        t_min=0.0,
        t_max=24.0,
        alpha=0.15,
        k=0.026,
        h=5.0,
        T_outside=0.0,
        source_locations=jnp.array([[5.0, 2.5]]),
        source_sizes=jnp.array([0.5]),
        source_strength=10.0,  # Strong source for clear effect
        nx=25,
        ny=15,
        nt=5,  # t = [0, 6, 12, 18, 24]
        sensor_rate=1.0,
        sensor_noise=0.0,
        sensor_locations=jnp.array([[5.0, 2.5], [0.0, 0.0]]),
        layer_sizes=[3, 8, 1],
        learning_rate=0.001,
        num_epochs=10,
        seed=42,
        lambda_physics=1.0,
        lambda_ic=1.0,
        lambda_bc=1.0,
        lambda_data=1.0,
        num_collocation=50,
        num_ic=20,
        num_bc=20,
    )

    x, y, t, T = solve_heat_equation(cfg)

    # Find grid indices closest to sensor location
    sensor_location = cfg.sensor_locations[0]
    cx, cy = sensor_location
    i_center = np.argmin(np.abs(x - cx))
    j_center = np.argmin(np.abs(y - cy))

    # Verify time grid
    assert np.allclose(t, [0, 6, 12, 18, 24]), f"Expected t=[0,6,12,18,24], got {t}"

    # With CORRECT implicit Euler (using t[n+1]):
    # - T[1] computed with source at t=6 -> OFF -> T[1] ≈ T[0] (small diffusion only)
    # - T[2] computed with source at t=12 -> ON -> T[2] > T[1] significantly
    #
    # With WRONG implementation (using t[n]):
    # - T[1] computed with source at t=0 -> OFF
    # - T[2] computed with source at t=6 -> OFF -> T[2] ≈ T[1] (no source heating!)

    T_at_1 = T[1, i_center, j_center]  # Should have no source (t=6 < 12)
    T_at_2 = T[2, i_center, j_center]  # Should have source if using t[n+1]=12

    # The temperature jump from T[1] to T[2] should be significant
    # because the source turns on at t=12 and implicit Euler evaluates at t[n+1]
    temp_increase = T_at_2 - T_at_1

    # With diffusion only (no source), temperature change would be minimal
    # With source, we expect a noticeable increase
    assert temp_increase > 1.0, (
        f"Temperature at center: T[1]={T_at_1:.4f}, T[2]={T_at_2:.4f}, "
        f"increase={temp_increase:.4f}. "
        f"Expected significant increase because source turns on at t={turn_on_time}. "
        "For implicit Euler, the RHS (including heat source) must be evaluated at "
        "t[n+1], not t[n]:\n"
        "  rhs = _build_rhs(cfg, T[n], X, Y, dx, dy, dt, t[n+1])  # CORRECT\n"
        "  rhs = _build_rhs(cfg, T[n], X, Y, dx, dy, dt, t[n])    # WRONG"
    )

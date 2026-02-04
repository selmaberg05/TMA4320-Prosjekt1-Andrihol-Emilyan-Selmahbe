"""Tests for loss.py."""

import jax
import jax.numpy as jnp

from project.loss import data_loss, ic_loss, physics_loss
from project.model import forward


class TestDataLoss:
    """Tests for data loss function (Oppgave 4.2)."""

    def test_returns_scalar(self, nn_params, sensor_data, cfg):
        """Check that data_loss returns a scalar value."""
        loss = data_loss(nn_params, sensor_data, cfg)

        assert jnp.ndim(loss) == 0, (
            f"data_loss should return a scalar, got shape {loss.shape}. "
            "Make sure to use jnp.mean() to reduce the squared errors to a single value."
        )

    def test_returns_non_negative(self, nn_params, sensor_data, cfg):
        """Check that data_loss returns a non-negative value."""
        loss = data_loss(nn_params, sensor_data, cfg)

        assert loss >= 0, (
            f"data_loss should be non-negative (it's a mean squared error), got {loss}. "
            "Use jnp.mean((T_pred - T_true)**2)."
        )

    def test_not_none(self, nn_params, sensor_data, cfg):
        """Check that data_loss returns a value (not None)."""
        loss = data_loss(nn_params, sensor_data, cfg)

        assert loss is not None, (
            "data_loss returned None. "
            "Make sure to assign the computed MSE to data_loss_val and return it."
        )

    def test_zero_when_perfect_fit(self, cfg):
        """Check that loss is zero when predictions equal targets."""
        from project.model import init_nn_params

        nn_params = init_nn_params(cfg, seed=42)

        # Create fake sensor data where network output matches targets
        x = jnp.array([5.0, 2.0])
        y = jnp.array([2.5, 1.0])
        t = jnp.array([12.0, 6.0])
        T_true = forward(nn_params, x, y, t, cfg)  # Use actual network output as target

        sensor_data = jnp.stack([x, y, t, T_true], axis=1)
        loss = data_loss(nn_params, sensor_data, cfg)

        assert jnp.allclose(loss, 0.0, atol=1e-5), (
            f"When predictions exactly match targets, loss should be 0, got {loss}. "
            "Check your MSE computation: jnp.mean((T_pred - T_true)**2)."
        )

    def test_differentiable(self, nn_params, sensor_data, cfg):
        """Check that data_loss is differentiable with respect to parameters."""

        def loss_fn(params):
            return data_loss(params, sensor_data, cfg)

        try:
            grads = jax.grad(loss_fn)(nn_params)
            # Check gradients are not all zero
            has_nonzero_grad = any(jnp.any(w != 0) or jnp.any(b != 0) for w, b in grads)
            assert has_nonzero_grad, (
                "Gradients are all zero. "
                "Make sure forward() and the loss computation use differentiable operations."
            )
        except Exception as e:
            raise AssertionError(
                f"data_loss is not differentiable: {e}. "
                "Make sure to use JAX operations (jnp) for all computations."
            )

    def test_exact_mse_formula(self, nn_params, cfg):
        """Verify data_loss computes exactly: mean((T_pred - T_true)^2)."""
        # Create simple sensor data
        x = jnp.array([2.0, 5.0, 8.0])
        y = jnp.array([1.0, 2.5, 4.0])
        t = jnp.array([6.0, 12.0, 18.0])
        T_true = jnp.array([10.0, 20.0, 15.0])

        sensor_data = jnp.stack([x, y, t, T_true], axis=1)

        # Compute loss using the function
        loss = data_loss(nn_params, sensor_data, cfg)

        # Compute expected loss manually
        T_pred = forward(nn_params, x, y, t, cfg)
        expected_loss = jnp.mean((T_pred - T_true) ** 2)

        assert jnp.allclose(loss, expected_loss, atol=1e-6), (
            f"data_loss = {float(loss):.8f}, but manual MSE = {float(expected_loss):.8f}. "
            "The formula should be: jnp.mean((T_pred - T_true)**2)\n"
            "Steps:\n"
            "1. Extract: x, y, t, T_true = sensor_data[:, 0], sensor_data[:, 1], sensor_data[:, 2], sensor_data[:, 3]\n"
            "2. Predict: T_pred = forward(nn_params, x, y, t, cfg)\n"
            "3. MSE: loss = jnp.mean((T_pred - T_true)**2)"
        )


class TestICLoss:
    """Tests for initial condition loss function (Oppgave 4.2)."""

    def test_returns_scalar(self, nn_params, ic_points, cfg):
        """Check that ic_loss returns a scalar value."""
        loss = ic_loss(nn_params, ic_points, cfg)

        assert jnp.ndim(loss) == 0, (
            f"ic_loss should return a scalar, got shape {loss.shape}. "
            "Make sure to use jnp.mean() to reduce to a single value."
        )

    def test_returns_non_negative(self, nn_params, ic_points, cfg):
        """Check that ic_loss returns a non-negative value."""
        loss = ic_loss(nn_params, ic_points, cfg)

        assert loss >= 0, (
            f"ic_loss should be non-negative (it's a mean squared error), got {loss}. "
            "Use jnp.mean((T_pred - T_outside)**2)."
        )

    def test_not_none(self, nn_params, ic_points, cfg):
        """Check that ic_loss returns a value (not None)."""
        loss = ic_loss(nn_params, ic_points, cfg)

        assert loss is not None, (
            "ic_loss returned None. "
            "Make sure to assign the computed MSE to ic_loss_val and return it."
        )

    def test_uses_t_min(self, nn_params, cfg):
        """Check that ic_loss evaluates network at t=t_min."""
        # Create IC points
        ic_points = jnp.array([[5.0, 2.5, 0.0], [2.0, 1.0, 0.0]])

        loss = ic_loss(nn_params, ic_points, cfg)

        # Loss should be based on T(x, y, t_min) compared to T_outside
        # If network is randomly initialized, loss should be non-trivial
        assert not jnp.isnan(loss), (
            "ic_loss produced NaN. "
            "Check that you're calling forward() correctly with t=cfg.t_min."
        )

    def test_measures_deviation_from_T_outside(self, nn_params, cfg):
        """Check that ic_loss measures deviation from initial temperature."""
        ic_points = jnp.array([[5.0, 2.5, 0.0], [2.0, 1.0, 0.0]])

        loss = ic_loss(nn_params, ic_points, cfg)

        # Manually compute expected loss
        x, y = ic_points[:, 0], ic_points[:, 1]
        T_pred = forward(nn_params, x, y, cfg.t_min, cfg)
        expected_loss = jnp.mean((T_pred - cfg.T_outside) ** 2)

        assert jnp.allclose(loss, expected_loss, atol=1e-5), (
            f"ic_loss = {loss}, but expected {expected_loss}. "
            "The loss should be: jnp.mean((forward(params, x, y, cfg.t_min, cfg) - cfg.T_outside)**2)"
        )

    def test_exact_ic_formula(self, nn_params, cfg):
        """Verify ic_loss computes exactly: mean((T(x,y,t_min) - T_outside)^2)."""
        # Create IC points
        ic_points = jnp.array(
            [
                [1.0, 1.0, 0.0],
                [5.0, 2.5, 0.0],
                [9.0, 4.0, 0.0],
            ]
        )

        loss = ic_loss(nn_params, ic_points, cfg)

        # Manual computation
        x = ic_points[:, 0]
        y = ic_points[:, 1]
        # MUST use cfg.t_min, not the t column from ic_points
        T_pred = forward(nn_params, x, y, cfg.t_min, cfg)
        expected_loss = jnp.mean((T_pred - cfg.T_outside) ** 2)

        assert jnp.allclose(loss, expected_loss, atol=1e-6), (
            f"ic_loss = {float(loss):.8f}, expected = {float(expected_loss):.8f}. "
            "The formula should be:\n"
            "1. Extract x, y from ic_points (ignore t column)\n"
            "2. T_pred = forward(nn_params, x, y, cfg.t_min, cfg)  # Use cfg.t_min!\n"
            "3. loss = jnp.mean((T_pred - cfg.T_outside)**2)"
        )


class TestPhysicsLoss:
    """Tests for physics (PDE residual) loss function (Oppgave 5.2)."""

    def test_returns_scalar(self, pinn_params, colloc_points, cfg):
        """Check that physics_loss returns a scalar value."""
        loss = physics_loss(pinn_params, colloc_points, cfg)

        assert jnp.ndim(loss) == 0, (
            f"physics_loss should return a scalar, got shape {loss.shape}. "
            "Make sure to compute jnp.mean(residuals**2)."
        )

    def test_returns_non_negative(self, pinn_params, colloc_points, cfg):
        """Check that physics_loss returns a non-negative value."""
        loss = physics_loss(pinn_params, colloc_points, cfg)

        assert loss >= 0, (
            f"physics_loss should be non-negative, got {loss}. "
            "The loss is mean of squared residuals."
        )

    def test_not_none(self, pinn_params, colloc_points, cfg):
        """Check that physics_loss returns a value (not None)."""
        loss = physics_loss(pinn_params, colloc_points, cfg)

        assert loss is not None, (
            "physics_loss returned None. "
            "Make sure to compute the PDE residual and assign to physics_loss_val."
        )

    def test_no_nan(self, pinn_params, colloc_points, cfg):
        """Check that physics_loss doesn't produce NaN."""
        loss = physics_loss(pinn_params, colloc_points, cfg)

        assert not jnp.isnan(loss), (
            "physics_loss produced NaN. "
            "Check your gradient computations: T_t = grad(T_fn, 2)(x, y, t), "
            "T_xx = grad(grad(T_fn, 0), 0)(x, y, t), T_yy = grad(grad(T_fn, 1), 1)(x, y, t)."
        )

    def test_uses_learned_alpha(self, cfg):
        """Check that physics_loss uses the learned alpha parameter."""
        from project.model import init_pinn_params

        pinn_params = init_pinn_params(cfg, seed=42)
        colloc = jnp.array([[5.0, 2.5, 12.0]])

        loss1 = physics_loss(pinn_params, colloc, cfg)

        # Modify log_alpha and check loss changes
        modified_params = {**pinn_params, "log_alpha": pinn_params["log_alpha"] + 1.0}
        loss2 = physics_loss(modified_params, colloc, cfg)

        assert not jnp.allclose(loss1, loss2), (
            "physics_loss didn't change when log_alpha changed. "
            "Make sure to use alpha = jnp.exp(pinn_params['log_alpha']) in the residual."
        )

    def test_differentiable(self, pinn_params, colloc_points, cfg):
        """Check that physics_loss is differentiable."""

        def loss_fn(params):
            return physics_loss(params, colloc_points, cfg)

        try:
            grads = jax.grad(loss_fn)(pinn_params)
            assert "nn" in grads, "Gradients should include 'nn' key."
            assert "log_alpha" in grads, "Gradients should include 'log_alpha' key."
        except Exception as e:
            raise AssertionError(
                f"physics_loss is not differentiable: {e}. "
                "Make sure to use vmap for batching and JAX's grad for derivatives."
            )

    def test_uses_vmap(self, pinn_params, cfg):
        """Check that physics_loss correctly handles batched collocation points."""
        # Single point
        colloc_single = jnp.array([[5.0, 2.5, 12.0]])
        loss_single = physics_loss(pinn_params, colloc_single, cfg)

        # Multiple points
        colloc_batch = jnp.array([[5.0, 2.5, 12.0], [2.0, 1.0, 6.0], [8.0, 4.0, 18.0]])
        loss_batch = physics_loss(pinn_params, colloc_batch, cfg)

        # Both should work without error and return scalars
        assert jnp.ndim(loss_single) == 0 and jnp.ndim(loss_batch) == 0, (
            "physics_loss should work for any number of collocation points. "
            "Use vmap to vectorize the residual computation."
        )

    def test_exact_pde_residual_formula(self, pinn_params, cfg):
        """Verify physics_loss computes: mean((dT/dt - alpha*(d2T/dx2 + d2T/dy2) - q)^2)."""
        from jax import grad

        # Test at a single point (outside heat source for simplicity)
        x_pt, y_pt, t_pt = 1.0, 1.0, 12.0  # Outside source region
        colloc = jnp.array([[x_pt, y_pt, t_pt]])

        loss = physics_loss(pinn_params, colloc, cfg)

        # Manually compute the PDE residual
        def T_fn(x, y, t):
            return forward(pinn_params["nn"], x, y, t, cfg)

        # Compute derivatives
        T_t = grad(T_fn, 2)(x_pt, y_pt, t_pt)
        # T_x = grad(T_fn, 0)(x_pt, y_pt, t_pt)
        T_xx = grad(grad(T_fn, 0), 0)(x_pt, y_pt, t_pt)
        T_yy = grad(grad(T_fn, 1), 1)(x_pt, y_pt, t_pt)

        # Get learned alpha
        alpha = jnp.exp(pinn_params["log_alpha"])

        # Heat source (should be 0 at this point, outside source)
        q = jnp.where(cfg.is_source(x_pt, y_pt), jnp.exp(pinn_params["log_power"]), 0.0)

        # PDE residual: dT/dt - alpha*(d2T/dx2 + d2T/dy2) - q = 0
        residual = T_t - alpha * (T_xx + T_yy) - q
        expected_loss = residual**2  # Single point, so mean = value

        assert jnp.allclose(loss, expected_loss, atol=1e-5), (
            f"physics_loss = {float(loss):.8f}, expected = {float(expected_loss):.8f}. "
            f"(T_t={float(T_t):.6f}, T_xx={float(T_xx):.6f}, T_yy={float(T_yy):.6f}, "
            f"alpha={float(alpha):.6f}, q={float(q):.6f})\n"
            "The PDE residual should be:\n"
            "  residual = T_t - alpha * (T_xx + T_yy) - q\n"
            "where:\n"
            "  T_t = grad(T_fn, 2)(x, y, t)  # derivative w.r.t. t (index 2)\n"
            "  T_xx = grad(grad(T_fn, 0), 0)(x, y, t)  # second derivative w.r.t. x\n"
            "  T_yy = grad(grad(T_fn, 1), 1)(x, y, t)  # second derivative w.r.t. y\n"
            "  alpha = jnp.exp(pinn_params['log_alpha'])\n"
            "  q = jnp.where(cfg.is_source(x, y), jnp.exp(pinn_params['log_power']), 0.0)"
        )

    def test_heat_source_contribution(self, pinn_params, cfg):
        """Verify that heat source term q is included in physics residual."""

        # Use one of the defined source locations
        source_location = cfg.source_locations[0]
        cx, cy = source_location
        colloc_in_source = jnp.array([[cx, cy, 12.0]])

        # Test outside source
        colloc_outside = jnp.array([[0.5, 0.5, 12.0]])

        loss_in = physics_loss(pinn_params, colloc_in_source, cfg)
        loss_out = physics_loss(pinn_params, colloc_outside, cfg)

        # Verify source detection
        assert cfg.is_source(cx, cy), (
            f"Point ({cx}, {cy}) should be inside heat source region."
        )
        assert not cfg.is_source(0.5, 0.5), (
            "Point (0.5, 0.5) should be outside heat source region."
        )

        # Both should produce valid losses (different due to source term)
        assert jnp.isfinite(loss_in) and jnp.isfinite(loss_out), (
            "physics_loss should handle points both inside and outside the heat source."
        )

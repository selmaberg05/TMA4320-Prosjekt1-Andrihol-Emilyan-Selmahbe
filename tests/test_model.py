"""Tests for model.py."""

import jax.numpy as jnp
import numpy as np

from project.model import forward, init_pinn_params, predict_grid


class TestForward:
    """Tests for the neural network forward pass (Oppgave 4.1)."""

    def test_returns_array(self, nn_params, cfg):
        """Check that forward returns a JAX array."""
        x, y, t = 5.0, 2.5, 12.0
        result = forward(nn_params, x, y, t, cfg)

        assert isinstance(result, (jnp.ndarray, np.ndarray, float)), (
            f"forward() should return a number or array, got {type(result)}. "
            "Make sure you return the output of the final layer."
        )

    def test_scalar_input_returns_scalar(self, nn_params, cfg):
        """Check that scalar inputs return a scalar output."""
        x, y, t = 5.0, 2.5, 12.0
        result = forward(nn_params, x, y, t, cfg)

        # Should return a scalar (0-dimensional)
        result_shape = jnp.atleast_1d(result).shape
        assert result_shape == (1,) or result_shape == (), (
            f"For scalar inputs, forward() should return a scalar, got shape {result_shape}. "
            "Make sure the output is squeezed: use .squeeze() on the final result."
        )

    def test_batched_input_returns_batch(self, nn_params, cfg):
        """Check that batched inputs return batched outputs."""
        n_points = 10
        x = jnp.linspace(cfg.x_min, cfg.x_max, n_points)
        y = jnp.linspace(cfg.y_min, cfg.y_max, n_points)
        t = jnp.ones(n_points) * 12.0

        result = forward(nn_params, x, y, t, cfg)

        assert result.shape == (n_points,), (
            f"For {n_points} input points, output should have shape ({n_points},), "
            f"got {result.shape}. Make sure you handle batched inputs correctly."
        )

    def test_normalization_applied(self, nn_params, cfg):
        """Check that input normalization is applied (affects output range)."""
        # Test at domain corners - normalized should be 0 and 1
        x_min_result = forward(nn_params, cfg.x_min, cfg.y_min, cfg.t_min, cfg)
        x_max_result = forward(nn_params, cfg.x_max, cfg.y_max, cfg.t_max, cfg)

        # Results should be different (network sees different normalized inputs)
        assert not jnp.allclose(x_min_result, x_max_result), (
            "Output at (x_min, y_min, t_min) equals output at (x_max, y_max, t_max). "
            "This suggests inputs are not being normalized. "
            "Use: x_norm = (x - cfg.x_min) / (cfg.x_max - cfg.x_min), etc."
        )

    def test_uses_tanh_activation(self, nn_params, cfg):
        """Check that hidden layers use tanh (bounded outputs during training)."""
        # Create simple params to verify tanh is used
        from project.model import init_nn_params

        # Initialize with seed for reproducibility
        test_params = init_nn_params(cfg, seed=123)

        # Multiple forward passes should give values in reasonable range
        # (tanh keeps hidden activations in [-1, 1])
        x = jnp.linspace(cfg.x_min, cfg.x_max, 20)
        y = jnp.linspace(cfg.y_min, cfg.y_max, 20)
        t = jnp.ones(20) * cfg.t_max / 2

        result = forward(test_params, x, y, t, cfg)

        # With tanh activations and initialized weights, outputs should be bounded
        assert not jnp.any(jnp.isnan(result)), (
            "forward() produced NaN values. "
            "Check the activation function and matrix multiplications."
        )
        assert not jnp.any(jnp.isinf(result)), (
            "forward() produced Inf values. "
            "This suggests missing activation function or numerical issues."
        )

    def test_broadcast_mixed_inputs(self, nn_params, cfg):
        """Check that scalar t broadcasts with array x, y."""
        n_points = 5
        x = jnp.linspace(cfg.x_min, cfg.x_max, n_points)
        y = jnp.linspace(cfg.y_min, cfg.y_max, n_points)
        t = 12.0  # scalar

        result = forward(nn_params, x, y, t, cfg)

        assert result.shape == (n_points,), (
            f"With array x, y and scalar t, output should have shape ({n_points},), "
            f"got {result.shape}. The broadcasting setup should handle this case."
        )

    def test_exact_normalization_values(self, nn_params, cfg):
        """Verify that normalization produces correct values at domain boundaries."""
        # Create minimal network: input(3) -> output(1) with no hidden layers
        # to isolate normalization testing
        from project.config import Config
        from project.model import init_nn_params

        simple_cfg = Config(
            x_min=0.0,
            x_max=10.0,
            y_min=0.0,
            y_max=5.0,
            t_min=0.0,
            t_max=20.0,
            alpha=0.15,
            k=0.026,
            h=5.0,
            T_outside=0.0,
            source_locations=jnp.array([[5.0, 2.5]]),
            source_sizes=jnp.array([0.5]),
            source_strength=5.0,
            nx=10,
            ny=10,
            nt=10,
            sensor_rate=1.0,
            sensor_noise=0.0,
            sensor_locations=jnp.array([[5.0, 2.5]]),
            layer_sizes=[3, 1],  # Direct linear mapping for easy verification
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

        # Initialize with known weights
        simple_params = init_nn_params(simple_cfg, seed=0)

        # Test at corners: normalized inputs should be (0,0,0) and (1,1,1)
        # At (x_min, y_min, t_min): normalized = [0, 0, 0]
        # At (x_max, y_max, t_max): normalized = [1, 1, 1]
        out_min = forward(
            simple_params,
            simple_cfg.x_min,
            simple_cfg.y_min,
            simple_cfg.t_min,
            simple_cfg,
        )
        out_max = forward(
            simple_params,
            simple_cfg.x_max,
            simple_cfg.y_max,
            simple_cfg.t_max,
            simple_cfg,
        )

        # Manually compute expected outputs
        W, b = simple_params[0]
        # At min: h = [0, 0, 0], out = [0,0,0] @ W + b = b
        expected_min = float(b[0])
        # At max: h = [1, 1, 1], out = [1,1,1] @ W + b = sum(W) + b
        expected_max = float(jnp.sum(W) + b[0])

        assert jnp.allclose(out_min, expected_min, atol=1e-5), (
            f"At domain minimum, output should be {expected_min:.6f} (just bias), "
            f"got {float(out_min):.6f}. "
            "Check normalization: x_norm = (x - x_min) / (x_max - x_min), "
            "which should give 0 at x_min."
        )
        assert jnp.allclose(out_max, expected_max, atol=1e-5), (
            f"At domain maximum, output should be {expected_max:.6f}, "
            f"got {float(out_max):.6f}. "
            "Check normalization: at x_max, x_norm should be 1."
        )

    def test_exact_forward_computation(self, cfg):
        """Verify forward pass matches manual matrix multiplication with tanh."""
        from project.model import init_nn_params

        nn_params = init_nn_params(cfg, seed=42)

        # Test point
        x, y, t = 5.0, 2.5, 12.0

        # Manual computation
        x_norm = (x - cfg.x_min) / (cfg.x_max - cfg.x_min)
        y_norm = (y - cfg.y_min) / (cfg.y_max - cfg.y_min)
        t_norm = (t - cfg.t_min) / (cfg.t_max - cfg.t_min)
        h = jnp.array([[x_norm, y_norm, t_norm]])  # Shape (1, 3)

        # Hidden layers with tanh
        for w, b in nn_params[:-1]:
            h = jnp.tanh(h @ w + b)

        # Output layer (linear, no activation)
        w_out, b_out = nn_params[-1]
        expected = (h @ w_out + b_out).squeeze()

        # Compare with forward()
        result = forward(nn_params, x, y, t, cfg)

        assert jnp.allclose(result, expected, atol=1e-5), (
            f"Forward pass output {float(result):.6f} doesn't match manual computation {float(expected):.6f}. "
            "Check your implementation:\n"
            "1. Normalize: x_norm = (x - cfg.x_min) / (cfg.x_max - cfg.x_min)\n"
            "2. Stack: h = jnp.stack([x_norm, y_norm, t_norm], axis=-1)\n"
            "3. Hidden layers: h = jnp.tanh(h @ w + b)\n"
            "4. Output layer: out = h @ w_out + b_out (no activation)"
        )


class TestPredictGrid:
    """Tests for grid prediction (Oppgave 4.4)."""

    def test_output_shape(self, nn_params, cfg):
        """Check that output has shape (nt, nx, ny)."""
        x = jnp.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        y = jnp.linspace(cfg.y_min, cfg.y_max, cfg.ny)
        t = jnp.linspace(cfg.t_min, cfg.t_max, cfg.nt)

        T_pred = predict_grid(nn_params, x, y, t, cfg)

        assert T_pred.shape == (cfg.nt, cfg.nx, cfg.ny), (
            f"predict_grid should return shape (nt={cfg.nt}, nx={cfg.nx}, ny={cfg.ny}), "
            f"got {T_pred.shape}. "
            "Make sure the output tensor is built correctly with T_pred = jnp.zeros((nt, nx, ny)) "
            "and filled using T_pred = T_pred.at[n].set(T_pred_t)."
        )

    def test_no_nan_values(self, nn_params, cfg):
        """Check that grid prediction contains no NaN values."""
        x = jnp.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        y = jnp.linspace(cfg.y_min, cfg.y_max, cfg.ny)
        t = jnp.linspace(cfg.t_min, cfg.t_max, cfg.nt)

        T_pred = predict_grid(nn_params, x, y, t, cfg)

        assert not jnp.any(jnp.isnan(T_pred)), (
            "predict_grid produced NaN values. "
            "Check that forward() is being called correctly with meshgrid coordinates."
        )

    def test_different_time_steps(self, nn_params, cfg):
        """Check that predictions vary across time steps."""
        x = jnp.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        y = jnp.linspace(cfg.y_min, cfg.y_max, cfg.ny)
        t = jnp.linspace(cfg.t_min, cfg.t_max, cfg.nt)

        T_pred = predict_grid(nn_params, x, y, t, cfg)

        # First and last time steps should have different predictions
        # (since t is an input to the network)
        diff = jnp.abs(T_pred[0] - T_pred[-1]).mean()
        assert diff > 1e-6, (
            "Predictions at t=0 and t=t_max are identical. "
            "Make sure you're passing t[n] (different for each time step) to forward(), "
            "not the entire t array or a constant."
        )

    def test_correct_indexing_ij(self, nn_params, cfg):
        """Check that meshgrid uses 'ij' indexing."""
        # Use small grid for clarity
        x = jnp.array([0.0, 5.0, 10.0])  # 3 x-values
        y = jnp.array([0.0, 2.5, 5.0])  # 3 y-values
        t = jnp.array([0.0])  # 1 time step

        T_pred = predict_grid(nn_params, x, y, t, cfg)

        assert T_pred.shape == (1, 3, 3), (
            f"With 3 x-values, 3 y-values, 1 time step, shape should be (1, 3, 3), "
            f"got {T_pred.shape}. "
            "Make sure you're using jnp.meshgrid(x, y, indexing='ij') "
            "so that output is (nx, ny) not (ny, nx)."
        )

    def test_grid_values_match_forward(self, nn_params, cfg):
        """Verify each grid point exactly matches calling forward() directly."""
        x = jnp.array([0.0, 5.0, 10.0])
        y = jnp.array([0.0, 2.5, 5.0])
        t = jnp.array([0.0, 12.0])

        T_pred = predict_grid(nn_params, x, y, t, cfg)

        # Check each grid point matches forward() call
        for ti, t_val in enumerate(t):
            for xi, x_val in enumerate(x):
                for yi, y_val in enumerate(y):
                    expected = forward(nn_params, x_val, y_val, t_val, cfg)
                    actual = T_pred[ti, xi, yi]

                    assert jnp.allclose(actual, expected, atol=1e-5), (
                        f"Grid value at T[{ti}, {xi}, {yi}] = {float(actual):.6f} "
                        f"doesn't match forward({x_val}, {y_val}, {t_val}) = {float(expected):.6f}. "
                        "Make sure you're correctly iterating over the grid and storing results:\n"
                        "1. Create meshgrid: X, Y = jnp.meshgrid(x, y, indexing='ij')\n"
                        "2. Flatten: x_flat, y_flat = X.ravel(), Y.ravel()\n"
                        "3. Call forward: T_flat = forward(nn_params, x_flat, y_flat, t[n], cfg)\n"
                        "4. Reshape and store: T_pred = T_pred.at[n].set(T_flat.reshape(nx, ny))"
                    )

    def test_meshgrid_ij_vs_xy_indexing(self, nn_params, cfg):
        """Verify correct indexing: T[t, i, j] should be at position (x[i], y[j])."""
        x = jnp.array([0.0, 10.0])  # x[0]=0, x[1]=10
        y = jnp.array([0.0, 5.0])  # y[0]=0, y[1]=5
        t = jnp.array([0.0])

        T_pred = predict_grid(nn_params, x, y, t, cfg)

        # T[0, 0, 1] should be at position (x[0]=0, y[1]=5)
        expected_01 = forward(nn_params, 0.0, 5.0, 0.0, cfg)
        # T[0, 1, 0] should be at position (x[1]=10, y[0]=0)
        expected_10 = forward(nn_params, 10.0, 0.0, 0.0, cfg)

        assert jnp.allclose(T_pred[0, 0, 1], expected_01, atol=1e-5), (
            f"T[0, 0, 1] should equal forward(x[0]=0, y[1]=5, t=0) = {float(expected_01):.6f}, "
            f"got {float(T_pred[0, 0, 1]):.6f}. "
            "This suggests wrong indexing. Use indexing='ij' in meshgrid."
        )
        assert jnp.allclose(T_pred[0, 1, 0], expected_10, atol=1e-5), (
            f"T[0, 1, 0] should equal forward(x[1]=10, y[0]=0, t=0) = {float(expected_10):.6f}, "
            f"got {float(T_pred[0, 1, 0]):.6f}. "
            "This suggests wrong indexing. Use indexing='ij' in meshgrid."
        )


class TestInitPinnParams:
    """Tests for PINN parameter initialization (Oppgave 5.1)."""

    def test_returns_dict(self, cfg):
        """Check that init_pinn_params returns a dictionary."""
        pinn_params = init_pinn_params(cfg, seed=42)

        assert isinstance(pinn_params, dict), (
            f"init_pinn_params should return a dict, got {type(pinn_params)}. "
            "The returned structure should be a dictionary with 'nn' and scalar keys."
        )

    def test_contains_nn_key(self, cfg):
        """Check that pinn_params contains the 'nn' key with network weights."""
        pinn_params = init_pinn_params(cfg, seed=42)

        assert "nn" in pinn_params, (
            "pinn_params should contain an 'nn' key for neural network weights. "
            "Use pinn_params['nn'] = init_nn_params(cfg, key=nn_key)."
        )
        assert isinstance(pinn_params["nn"], list), (
            f"pinn_params['nn'] should be a list of (W, b) tuples, "
            f"got {type(pinn_params['nn'])}."
        )

    def test_contains_log_alpha(self, cfg):
        """Check that pinn_params contains log_alpha scalar parameter."""
        pinn_params = init_pinn_params(cfg, seed=42)

        assert "log_alpha" in pinn_params, (
            "pinn_params should contain 'log_alpha' for learnable thermal diffusivity. "
            "Add it to the scalar_names list and initialize with random.normal()."
        )
        assert pinn_params["log_alpha"].shape == (1,), (
            f"log_alpha should have shape (1,), got {pinn_params['log_alpha'].shape}. "
            "Use jax.random.normal(key, (1,)) to initialize."
        )

    def test_contains_log_power(self, cfg):
        """Check that pinn_params contains log_power scalar parameter."""
        pinn_params = init_pinn_params(cfg, seed=42)

        assert "log_power" in pinn_params, (
            "pinn_params should contain 'log_power' for learnable source strength. "
            "Add it to the scalar_names list."
        )

    def test_contains_log_k(self, cfg):
        """Check that pinn_params contains log_k scalar parameter."""
        pinn_params = init_pinn_params(cfg, seed=42)

        assert "log_k" in pinn_params, (
            "pinn_params should contain 'log_k' for learnable thermal conductivity. "
            "Add it to the scalar_names list."
        )

    def test_contains_log_h(self, cfg):
        """Check that pinn_params contains log_h scalar parameter."""
        pinn_params = init_pinn_params(cfg, seed=42)

        assert "log_h" in pinn_params, (
            "pinn_params should contain 'log_h' for learnable heat transfer coefficient. "
            "Add it to the scalar_names list."
        )

    def test_all_scalar_keys_present(self, cfg):
        """Check that all required scalar parameters are present."""
        pinn_params = init_pinn_params(cfg, seed=42)

        required_scalars = ["log_alpha", "log_power", "log_k", "log_h"]
        for name in required_scalars:
            assert name in pinn_params, (
                f"Missing scalar parameter: '{name}'. "
                f"Required scalars are: {required_scalars}"
            )

    def test_reproducible_with_seed(self, cfg):
        """Check that same seed produces same parameters."""
        params1 = init_pinn_params(cfg, seed=42)
        params2 = init_pinn_params(cfg, seed=42)

        # Check NN weights are identical
        for (w1, b1), (w2, b2) in zip(params1["nn"], params2["nn"]):
            assert jnp.allclose(w1, w2), (
                "NN weights differ with same seed. "
                "Make sure you're using the provided seed/key for initialization."
            )

        # Check scalars are identical
        assert jnp.allclose(params1["log_alpha"], params2["log_alpha"]), (
            "log_alpha differs with same seed."
        )

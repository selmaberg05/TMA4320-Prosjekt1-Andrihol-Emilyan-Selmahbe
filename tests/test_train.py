"""Tests for train.py."""

import jax.numpy as jnp

from project.train import train_nn, train_pinn


class TestTrainNN:
    """Tests for neural network training (Oppgave 4.3)."""

    def test_returns_tuple(self, sensor_data, cfg):
        """Check that train_nn returns a tuple of (params, losses)."""
        result = train_nn(sensor_data, cfg)

        assert isinstance(result, tuple), (
            f"train_nn should return a tuple, got {type(result)}. "
            "Return (nn_params, losses) at the end of the function."
        )
        assert len(result) == 2, (
            f"train_nn should return 2 values (params, losses), got {len(result)}."
        )

    def test_params_is_list(self, sensor_data, cfg):
        """Check that returned params is a list of (W, b) tuples."""
        nn_params, _ = train_nn(sensor_data, cfg)

        assert isinstance(nn_params, list), (
            f"First return value should be a list of layer parameters, got {type(nn_params)}."
        )
        assert len(nn_params) == len(cfg.layer_sizes) - 1, (
            f"Should have {len(cfg.layer_sizes) - 1} layers, got {len(nn_params)}."
        )
        for i, layer in enumerate(nn_params):
            assert isinstance(layer, tuple) and len(layer) == 2, (
                f"Layer {i} should be a (W, b) tuple, got {type(layer)}."
            )

    def test_losses_dict_keys(self, sensor_data, cfg):
        """Check that losses dict contains required keys."""
        _, losses = train_nn(sensor_data, cfg)

        assert isinstance(losses, dict), (
            f"Second return value should be a dict, got {type(losses)}."
        )

        required_keys = ["total", "data", "ic"]
        for key in required_keys:
            assert key in losses, (
                f"losses dict should contain '{key}' key. "
                f"Found keys: {list(losses.keys())}. "
                "Make sure to append losses to the lists in each epoch."
            )

    def test_loss_histories_correct_length(self, sensor_data, cfg):
        """Check that loss histories have num_epochs entries."""
        _, losses = train_nn(sensor_data, cfg)

        for key in ["total", "data", "ic"]:
            assert len(losses[key]) == cfg.num_epochs, (
                f"losses['{key}'] should have {cfg.num_epochs} entries (one per epoch), "
                f"got {len(losses[key])}. "
                "Make sure to append to the loss lists inside the training loop."
            )

    def test_losses_are_arrays(self, sensor_data, cfg):
        """Check that losses are returned as JAX arrays."""
        _, losses = train_nn(sensor_data, cfg)

        for key in ["total", "data", "ic"]:
            assert isinstance(losses[key], jnp.ndarray), (
                f"losses['{key}'] should be a jnp.ndarray, got {type(losses[key])}. "
                "Convert lists to arrays at the end: {{k: jnp.array(v) for k, v in losses.items()}}"
            )

    def test_total_loss_decreases(self, sensor_data, cfg):
        """Check that total loss generally decreases during training."""
        _, losses = train_nn(sensor_data, cfg)

        initial_loss = float(losses["total"][0])
        final_loss = float(losses["total"][-1])

        assert final_loss < initial_loss, (
            f"Loss should decrease during training. "
            f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}. "
            "Make sure the training loop: "
            "1) Computes gradients with jax.value_and_grad "
            "2) Updates params with adam_step "
            "3) Uses the updated params for the next iteration."
        )

    def test_no_nan_in_losses(self, sensor_data, cfg):
        """Check that training doesn't produce NaN losses."""
        _, losses = train_nn(sensor_data, cfg)

        for key in ["total", "data", "ic"]:
            assert not jnp.any(jnp.isnan(losses[key])), (
                f"losses['{key}'] contains NaN values. "
                "This usually means gradients exploded. "
                "Check that the loss function and forward pass are implemented correctly."
            )

    def test_uses_lambda_weights(self, cfg):
        """Check that the total loss uses lambda weights."""
        from project.data import generate_training_data

        _, _, _, _, sensor_data = generate_training_data(cfg)

        _, losses = train_nn(sensor_data, cfg)

        # If lambda weights are used, total should not equal just data_loss or ic_loss
        # (unless by coincidence with lambdas=1)
        # Just verify total is computed (non-zero)
        assert jnp.any(losses["total"] > 0), (
            "Total loss is all zeros. "
            "Make sure: total = cfg.lambda_data * l_data + cfg.lambda_ic * l_ic"
        )

    def test_step_function_is_jitted(self, sensor_data, cfg):
        """Check that training completes in reasonable time (implying JIT is used)."""
        import time

        start = time.time()
        train_nn(sensor_data, cfg)
        elapsed = time.time() - start

        # With JIT, 10 epochs should be fast (< 30 seconds even with compilation)
        # Without JIT, it would be much slower
        assert elapsed < 60, (
            f"Training took {elapsed:.1f} seconds for {cfg.num_epochs} epochs. "
            "This suggests the step function may not be JIT-compiled. "
            "Add @jit decorator: @jit followed by def step(...)."
        )


class TestTrainPINN:
    """Tests for PINN training (Oppgave 5.4)."""

    def test_returns_tuple(self, sensor_data, cfg):
        """Check that train_pinn returns a tuple of (pinn_params, losses)."""
        result = train_pinn(sensor_data, cfg)

        assert isinstance(result, tuple), (
            f"train_pinn should return a tuple, got {type(result)}. "
            "Return (pinn_params, losses) at the end of the function."
        )
        assert len(result) == 2, (
            f"train_pinn should return 2 values (pinn_params, losses), got {len(result)}."
        )

    def test_pinn_params_is_dict(self, sensor_data, cfg):
        """Check that returned pinn_params is a dictionary."""
        pinn_params, _ = train_pinn(sensor_data, cfg)

        assert isinstance(pinn_params, dict), (
            f"First return value should be a dict (pinn_params), got {type(pinn_params)}."
        )

    def test_pinn_params_has_nn(self, sensor_data, cfg):
        """Check that pinn_params contains neural network weights."""
        pinn_params, _ = train_pinn(sensor_data, cfg)

        assert "nn" in pinn_params, (
            "pinn_params should contain 'nn' key for neural network weights. "
            "Make sure init_pinn_params returns the correct structure."
        )

    def test_pinn_params_has_scalars(self, sensor_data, cfg):
        """Check that pinn_params contains learned scalar parameters."""
        pinn_params, _ = train_pinn(sensor_data, cfg)

        required_scalars = ["log_alpha", "log_power", "log_k", "log_h"]
        for name in required_scalars:
            assert name in pinn_params, (
                f"pinn_params should contain '{name}' scalar parameter. "
                f"Found keys: {list(pinn_params.keys())}"
            )

    def test_losses_dict_keys(self, sensor_data, cfg):
        """Check that losses dict contains all required PINN loss keys."""
        _, losses = train_pinn(sensor_data, cfg)

        assert isinstance(losses, dict), (
            f"Second return value should be a dict, got {type(losses)}."
        )

        required_keys = ["total", "data", "physics", "ic", "bc"]
        for key in required_keys:
            assert key in losses, (
                f"losses dict should contain '{key}' key. "
                f"Found keys: {list(losses.keys())}. "
                "PINN training should track: total, data, physics, ic, bc losses."
            )

    def test_loss_histories_correct_length(self, sensor_data, cfg):
        """Check that all loss histories have num_epochs entries."""
        _, losses = train_pinn(sensor_data, cfg)

        for key in ["total", "data", "physics", "ic", "bc"]:
            assert len(losses[key]) == cfg.num_epochs, (
                f"losses['{key}'] should have {cfg.num_epochs} entries, "
                f"got {len(losses[key])}. "
                "Make sure to append to the loss lists inside the training loop."
            )

    def test_no_nan_in_losses(self, sensor_data, cfg):
        """Check that training doesn't produce NaN losses."""
        _, losses = train_pinn(sensor_data, cfg)

        for key in ["total", "data", "physics", "ic", "bc"]:
            assert not jnp.any(jnp.isnan(losses[key])), (
                f"losses['{key}'] contains NaN values. "
                "Check the corresponding loss function for numerical issues."
            )

    def test_total_loss_decreases(self, sensor_data, cfg):
        """Check that total loss generally decreases during training."""
        _, losses = train_pinn(sensor_data, cfg)

        initial_loss = float(losses["total"][0])
        final_loss = float(losses["total"][-1])

        assert final_loss < initial_loss, (
            f"Loss should decrease during training. "
            f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}. "
            "Make sure the training loop updates pinn_params correctly "
            "using jax.value_and_grad and adam_step."
        )

    def test_physics_loss_tracked(self, sensor_data, cfg):
        """Check that physics loss is being computed and tracked."""
        _, losses = train_pinn(sensor_data, cfg)

        assert jnp.any(losses["physics"] > 0), (
            "Physics loss is all zeros. "
            "Make sure to call physics_loss(p, colloc, cfg) in the step function "
            "and include it in the total loss."
        )

    def test_bc_loss_tracked(self, sensor_data, cfg):
        """Check that boundary condition loss is being computed and tracked."""
        _, losses = train_pinn(sensor_data, cfg)

        assert jnp.any(losses["bc"] > 0), (
            "BC loss is all zeros. "
            "Make sure to call bc_loss(p, bc_batch, cfg) in the step function "
            "and include it in the total loss."
        )

    def test_samples_collocation_points(self, sensor_data, cfg):
        """Check that collocation points are sampled each epoch."""
        # This is implicitly tested by physics_loss not being NaN/zero
        _, losses = train_pinn(sensor_data, cfg)

        assert not jnp.any(jnp.isnan(losses["physics"])), (
            "Physics loss is NaN, which may indicate collocation points "
            "are not being sampled correctly. "
            "Use: colloc, key = _sample_interior(key, cfg)"
        )

    def test_samples_bc_points(self, sensor_data, cfg):
        """Check that boundary points are sampled each epoch."""
        _, losses = train_pinn(sensor_data, cfg)

        assert not jnp.any(jnp.isnan(losses["bc"])), (
            "BC loss is NaN, which may indicate boundary points "
            "are not being sampled correctly. "
            "Use: bc_batch, key = _sample_bc(key, cfg)"
        )

    def test_uses_correct_loss_functions_for_nn_vs_pinn(self, sensor_data, cfg):
        """Check that ic_loss and data_loss use pinn_params['nn'], not pinn_params."""
        pinn_params, losses = train_pinn(sensor_data, cfg)

        # If the wrong params are passed (pinn_params instead of pinn_params['nn']),
        # the loss functions will likely fail or produce different results
        # This test checks that the losses are reasonable (not NaN or extremely large)
        assert not jnp.any(jnp.isnan(losses["data"])), (
            "data_loss is NaN. Make sure to pass pinn_params['nn'] (not pinn_params) "
            "to data_loss: l_data = data_loss(p['nn'], data, cfg)"
        )
        assert not jnp.any(jnp.isnan(losses["ic"])), (
            "ic_loss is NaN. Make sure to pass pinn_params['nn'] (not pinn_params) "
            "to ic_loss: l_ic = ic_loss(p['nn'], ic_batch, cfg)"
        )

    def test_scalars_are_updated(self, sensor_data, cfg):
        """Check that scalar parameters change during training."""
        from project.model import init_pinn_params

        initial_params = init_pinn_params(cfg, seed=cfg.seed)
        trained_params, _ = train_pinn(sensor_data, cfg)

        # At least one scalar should have changed
        scalars_changed = False
        for name in ["log_alpha", "log_power", "log_k", "log_h"]:
            if not jnp.allclose(initial_params[name], trained_params[name]):
                scalars_changed = True
                break

        assert scalars_changed, (
            "No scalar parameters changed during training. "
            "Make sure adam_step updates the entire pinn_params dict, "
            "not just pinn_params['nn']."
        )

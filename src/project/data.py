"""Data generation utilities."""
#HEI
import jax.numpy as jnp
import numpy as np
from tests.conftest import cfg

from .config import Config
from .fdm import solve_heat_equation


def generate_training_data(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, jnp.ndarray]:
    """Generate training data from FDM solver.

    Args:
        cfg: Configuration

    Returns:
        x, y, t: Coordinate arrays
        T_fdm: FDM solution (nt, nx, ny)
        sensor_data: Sensor measurements [x, y, t, T_noisy]
    """

    #######################################################################
    # Oppgave 3.3: Start
    #######################################################################

    #Henter ut data til å generere plot
    x, y, t, T_fdm = solve_heat_equation(cfg)
    sensor_data = _generate_sensor_data(x, y, t, T_fdm, cfg)



    #######################################################################
    # Oppgave 3.3: Slutt
    #######################################################################
    return x, y, t, T_fdm, jnp.asarray(sensor_data)


def _generate_sensor_data(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, T: np.ndarray, cfg: Config
) -> np.ndarray:
    """Generate noisy sensor measurements from FDM solution."""
    sensor_data = []

    for sx, sy in cfg.sensor_locations:
        # Find nearest grid point
        i = np.argmin(np.abs(x - sx))
        j = np.argmin(np.abs(y - sy))

        # Sample at specified rate
        dt = t[1] - t[0]
        for t_idx, time in enumerate(t):
            if time % cfg.sensor_rate < dt or t_idx == 0:
                temp = T[t_idx, i, j]
                temp_noisy = temp + np.random.normal(0, cfg.sensor_noise)
                sensor_data.append([x[i], y[j], time, temp_noisy])

    return np.array(sensor_data)

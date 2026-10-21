import jax, jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

N_VARS = 4  # S, E, I, R

class Parameters:
    def __init__(self, *, alpha: float, beta: float, gamma: float, delta: float, W: np.ndarray, quarantine_factor: float = 0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.n_cities = W.shape[0]
        self.W = jnp.array((1 - quarantine_factor) *  W)
        self.W_colsum = jnp.sum(self.W, axis=0)

@jax.jit(static_argnames=("p",))
def f(t, y: jnp.ndarray, p: Parameters):
    S, E, I, R = y.reshape((N_VARS, -1))

    N = S + E + I + R
    recovery_rate = p.alpha * I
    exposure_rate = p.beta * S * I / N
    infection_rate = p.delta * E
    vaccination_rate = p.gamma * S

    dS = -exposure_rate - vaccination_rate + p.W @ S - p.W_colsum * S
    dE = exposure_rate - infection_rate + p.W @ E - p.W_colsum * E
    dI = infection_rate - recovery_rate + p.W @ I - p.W_colsum * I
    dR = vaccination_rate + recovery_rate + p.W @ R - p.W_colsum * R

    return jnp.concat((dS, dE, dI, dR))

import jax, jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

N_VARS = 3  # S, I, R

class Parameters:
    def __init__(self, *, alpha: float, beta: float, gamma: float, W: np.ndarray, quarantine_factor: float = 0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_cities = W.shape[0]
        self.W = jnp.array((1 - quarantine_factor) *  W)
        self.W_colsum = jnp.sum(self.W, axis=0)

@jax.jit(static_argnames=("p",))
def f(t, y: jnp.ndarray, p: Parameters):
    S, I, R = y.reshape((N_VARS, -1))

    N = S + I + R
    recovery_rate = p.alpha * I
    infection_rate = p.beta * S * I / N
    vaccination_rate = p.gamma * S

    dS = -infection_rate - vaccination_rate + p.W @ S - p.W_colsum * S
    dI = infection_rate - recovery_rate + p.W @ I - p.W_colsum * I
    dR = vaccination_rate + recovery_rate + p.W @ R - p.W_colsum * R

    return jnp.concat((dS, dI, dR))

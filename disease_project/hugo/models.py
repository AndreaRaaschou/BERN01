from diffrax import ODETerm
import jax, jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_NAN = float("NaN")


class Parameters:
    def __init__(
        self,
        *,
        alpha: float = _NAN,
        beta: float = _NAN,
        gamma: float = _NAN,
        delta: float = _NAN,
        W: np.ndarray = np.zeros((1, 1)),
        quarantine_factor: float = 0
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.n_cities = W.shape[0]
        self.W = jnp.array((1 - quarantine_factor) * W)
        self.W_colsum = jnp.sum(self.W, axis=0)


@jax.jit(static_argnames=("p",))
def _rhs_SIS(t, y: jnp.ndarray, p: Parameters):
    S, I = y.reshape((2, -1))

    N = S + I
    recovery_rate = p.alpha * I
    infection_rate = p.beta * S * I / N

    dS = -infection_rate + recovery_rate + p.W @ S - p.W_colsum * S
    dI = infection_rate - recovery_rate + p.W @ I - p.W_colsum * I

    return jnp.concat((dS, dI))


@jax.jit(static_argnames=("p",))
def _rhs_SIR(t, y: jnp.ndarray, p: Parameters):
    S, I, R = y.reshape((3, -1))

    N = S + I + R
    recovery_rate = p.alpha * I
    infection_rate = p.beta * S * I / N
    vaccination_rate = p.gamma * S

    dS = -infection_rate - vaccination_rate + p.W @ S - p.W_colsum * S
    dI = infection_rate - recovery_rate + p.W @ I - p.W_colsum * I
    dR = vaccination_rate + recovery_rate + p.W @ R - p.W_colsum * R

    return jnp.concat((dS, dI, dR))


@jax.jit(static_argnames=("p",))
def _rhs_SEIR(t, y: jnp.ndarray, p: Parameters):
    S, E, I, R = y.reshape((4, -1))

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

@jax.jit(static_argnames=("p",))
def _rhs_SEIRD(t, y: jnp.ndarray, p: Parameters):
    S, E, I, R, D = y.reshape((5, -1))

    N = S + E + I + R
    recovery_rate = p.alpha * (1 - p.cfr) * I
    death_rate = p.alpha * p.cfr * I
    exposure_rate = p.beta * S * I / N
    infection_rate = p.delta * E
    vaccination_rate = p.gamma * S

    dS = -exposure_rate - vaccination_rate + p.W @ S - p.W_colsum * S
    dE = exposure_rate - infection_rate + p.W @ E - p.W_colsum * E
    dI = infection_rate - recovery_rate - death_rate + p.W @ I - p.W_colsum * I
    dR = vaccination_rate + recovery_rate + p.W @ R - p.W_colsum * R
    dD = death_rate  # the dead don't travel

    return jnp.concat((dS, dE, dI, dR, dD))


SIS = ODETerm(_rhs_SIS)
SIR = ODETerm(_rhs_SIR)
SEIR = ODETerm(_rhs_SEIR)
SEIRD = ODETerm(_rhs_SEIRD)

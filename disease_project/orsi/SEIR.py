import jax, jax.numpy as jnp
from hugo.SIR import Parameters

jax.config.update("jax_enable_x64", True)

N_VARS = 4  # S, E, I, R

class ExtraParameters(Parameters):
    def __init__(self, *, alpha, beta, gamma, sigma, W, quarantine_factor = 0):
        super().__init__(alpha=alpha, beta=beta, gamma=gamma, W=W, quarantine_factor=quarantine_factor)
        self.sigma = sigma

@jax.jit(static_argnames=("p",))
def f_e(t, y: jnp.ndarray, p: Parameters):
    S, E, I, R = y.reshape((N_VARS, -1))

    N = S+E+I+R

    recovery_rate = p.alpha * I
    infection_rate = p.beta * S * I / N
    vaccination_rate = p.gamma * S
    incubation_rate = p.sigma * E

    dS = -infection_rate - vaccination_rate + p.W @ S - p.W_colsum * S
    dI = infection_rate - recovery_rate + p.W @ I - p.W_colsum * I
    dR = vaccination_rate + recovery_rate + p.W @ R - p.W_colsum * R
    dE = infection_rate - incubation_rate + p.W @ E - p.W_colsum * E

    return jnp.concat((dS, dE, dI, dR))
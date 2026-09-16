# Python standard library imports
import time
from multiprocessing import Pool, get_context

# External library imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pgf.mplstyle')
plt.rcParams.update({
    "pgf.preamble": "\n".join([
        r"\usepackage{newpxmath}",
        r"\usepackage{newpxtext}",
    ]),
})

# Internal library imports
from qpotts import PottsModel, Start

# Constant definitions
FIG_PATH = '../results/'

# Common function definitions
def curie_temp(q: int):
    return 1 / (np.log(1 + np.sqrt(q)))

def batch_means_method(arr: np.ndarray, min_num_blocks: int = 8):
    sigma_list = []
    while True:
        sigma = arr.std(ddof=1) / np.sqrt(arr.size)
        sigma_list.append(sigma)

        if (arr.size <= min_num_blocks):
            break

        arr = arr.reshape(-1, 2).mean(axis=1)

    return np.max(sigma_list)

def iterate_until_equilibrium(pm: PottsModel, expect_increasing: bool, block_size = 1_000_000):
    while True:
        mu_1 = np.mean(pm.sample_metropolis(block_size, 10))
        mu_2 = np.mean(pm.sample_metropolis(block_size, 10))
        if (expect_increasing and mu_1 >= mu_2):
            return

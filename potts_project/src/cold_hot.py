import numpy as np
import matplotlib.pyplot as plt

from qpotts import PottsModel, Start

L=16
T=3
q=2
pm_cold = PottsModel(L, T, q, Start.Cold)
pm_hot = PottsModel(L, T, q, Start.Hot)


num_samples = 10_000
energies_cold = pm_cold.sample_metropolis(num_samples, 0)
energies_hot = pm_hot.sample_metropolis(num_samples, 0)

step = 10
xs = range(0,num_samples, step)
plt.plot(xs, energies_cold[::step], label="Cold")
plt.plot(xs, energies_hot[::step], label="Warm")
plt.xlabel("sample")
plt.ylabel("E")
plt.legend()
plt.show()

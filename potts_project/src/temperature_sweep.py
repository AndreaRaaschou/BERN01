import time

import numpy as np
import matplotlib.pyplot as plt

from qpotts import PottsModel, Start

L = 100

def run(q: int) -> None:
    temp_curie = 1 / (np.log(1 + np.sqrt(q)))
    temp_range = np.linspace(0.2, 2, 100)
    temp_range_lower = temp_range[temp_range<=temp_curie]
    temp_range_upper = temp_range[temp_range>temp_curie]

    pm_hot = PottsModel(L, temp_range_upper[-1], q, Start.Hot)
    pm_hot.sample_metropolis(0, 100_000_000)

    means_hot = []
    for temp in reversed(temp_range_upper):
        pm_hot.set_temperature(temp)
        energies = pm_hot.sample_metropolis(500_000, 100_000)
        means_hot.append(np.mean(energies))
    means_hot.reverse()

    pm_cold = PottsModel(L, temp_range_lower[0], q, Start.Cold)
    pm_cold.sample_metropolis(0, 100_000_000)

    means_cold = []
    for temp in temp_range_lower:
        pm_cold.set_temperature(temp)
        energies = pm_cold.sample_metropolis(500_000, 100_000)
        means_cold.append(np.mean(energies))

    means = means_cold + means_hot

    line, = plt.plot(temp_range, means, '.', label=f"q={q}")
    plt.axvline(x=temp_curie, c=line.get_color(), ls='--', alpha=0.5)

start = time.perf_counter()
run(2)
run(10)
end = time.perf_counter()
print(f"Elapsed time: {end - start} s")

plt.xlabel('T')
plt.ylabel('⟨E⟩')
plt.legend()
plt.show()


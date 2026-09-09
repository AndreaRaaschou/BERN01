import numpy as np
import matplotlib.pyplot as plt

from qpotts import PottsModel, Start

L = 100
T = 0.5
q = 3

pm = PottsModel(L, T, q, Start.Hot)
pm.sample_metropolis(1_000_000)
plt.imshow(pm.state_view())
plt.show()

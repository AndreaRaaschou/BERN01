import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(L, q, state_type):

    if state_type == 'cold':
        L_grid = np.ones((L, L))
    elif state_type == 'hot':
        L_grid = np.random.randint(1, q+1, size=(L, L), dtype=int)
    else:
        raise ValueError("Invalid state_type. Choose 'cold' or 'hot'.")

    return L_grid

def metropolis_step(L_grid, beta, q):
    L = L_grid.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        current_state = L_grid[i, j]
        new_state = np.random.randint(1, q + 1)

        if new_state != current_state:
            delta_E = calculate_energy_change(L_grid, i, j, new_state, current_state)
            if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
                L_grid[i, j] = new_state

def calculate_energy_change(L_grid, i, j, new_state, current_state):
    L = L_grid.shape[0]
    neighbors = [
        ((i - 1) % L, j),  # Up
        ((i + 1) % L, j),  # Down
        (i, (j - 1) % L),  # Left
        (i, (j + 1) % L)   # Right
    ]

    delta_E = 0
    for ni, nj in neighbors:
        if L_grid[ni, nj] == current_state:
            delta_E += 1
        if L_grid[ni, nj] == new_state:
            delta_E -= 1

    return delta_E

def calculate_total_energy(L_grid):
    L = L_grid.shape[0]
    E = 0
    for i in range(L):
        for j in range(L):
            current_state = L_grid[i, j]
            if current_state == L_grid[(i + 1) % L, j]:
                E -= 1
            if current_state == L_grid[i, (j + 1) % L]:
                E -= 1
    return E

def simulation(L, q, beta, num_steps, state_type):
    L_grid = initialize_grid(L, q, state_type)
    L_squared = L * L
    energies = []

    for step in range(num_steps):
        metropolis_step(L_grid, beta, q)
        total_energy = calculate_total_energy(L_grid)
        total_energy = total_energy / L_squared  # Normalize energy per spin
        energies.append(total_energy)

    return energies

def block_error(energies, block_size = 100):
    num_blocks = len(energies) // block_size
    block_means = np.array([np.mean(energies[i * block_size:(i + 1) * block_size]) for i in range(num_blocks)])
    mean_energy = np.mean(block_means)
    error = np.sqrt(np.var(block_means, ddof = 1) / num_blocks)
    return mean_energy, error

energies_hot = simulation(16, 2, 1.0/3.0, 500, "hot")
energies_cold = simulation(16, 2, 1.0/3.0, 500, "cold")


plt.plot(energies_hot, label="Hot Start")
plt.plot(energies_cold, label="Cold Start")
plt.xlabel("Monte Carlo Sweeps")
plt.ylabel("Energy per spin (E/N)")
plt.legend()
plt.show()

T_c_q2 = np.linspace(0.2, 2, 50)
energies_q2 = []
current_grid_q2 = initialize_grid(16, 2, "cold")

for T in T_c_q2:
    beta = 1.0 / T
    energies_step = []
    for step in range(2000):
        metropolis_step(current_grid_q2, beta, 2)
        if step >= 500:
            total_energy = calculate_total_energy(current_grid_q2)
            energies_step.append(total_energy)

    energies_q2.append(np.mean(energies_step))

T_c_q10 = np.linspace(0.2, 2, 100)
energies_q10 = []
current_grid_q10 = initialize_grid(16, 10, "cold")

for T in T_c_q10:
    beta = 1.0 / T
    energies_step = []
    for step in range(2000):
        metropolis_step(current_grid_q10, beta, 10)
        if step >= 500:
            total_energy = calculate_total_energy(current_grid_q10)
            energies_step.append(total_energy)

    energies_q10.append(np.mean(energies_step))

L_squared = 16 * 16
#energies_q2 = np.array(energies_q2) / L_squared
#energies_q10 = np.array(energies_q10) / L_squared

plt.plot(T_c_q2, energies_q2, label="q=2", fmt='.')
#plt.errorbar(T_c_q2, energies_q2, yerr=errors_q2, fmt='o-', label="q=2", capsize=3)
plt.plot(T_c_q10, energies_q10, label="q=10", fmt='.')
plt.xlabel("Temperature (T)")
plt.ylabel("Average Energy per spin (E/N)")
plt.legend()
plt.show()

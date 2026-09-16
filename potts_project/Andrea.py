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

# Changed this function to perform a single Metropolis step instead of L^2 steps
def metropolis_step(L_grid, L, beta, q):
    # Randomly select a spin to flip
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)

    current_state = L_grid[i, j]
    new_state = np.random.randint(1, q + 1)

    if new_state != current_state:
        delta_E = calculate_energy_change(L_grid, i, j, new_state, current_state)
        if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
            L_grid[i, j] = new_state

def metropolis_sweep(L_grid, L,beta, q):
    for _ in range(L * L):
        metropolis_step(L_grid, L, beta, q)

def simulation(L, q, beta, num_steps, state_type, filename):
    L_grid = initialize_grid(L, q, state_type)
    L_squared = L * L
    energies = []

    for step in range(num_steps):
        metropolis_sweep(L_grid, L, beta, q)
        total_energy = calculate_total_energy(L_grid)
        average_energy = total_energy / L_squared  # Normalize energy per spin
        energies.append(average_energy)

    np.savetxt(filename, energies) # Save energies to a text file

def phase_transition_simulation(L, q, num_steps, state_type, filename):
    T_c = np.linspace(0.2, 2, 50)
    energies = []
    current_grid = initialize_grid(L, q, state_type)

    for T in T_c:
        beta = 1.0 / T
        energies_step = []
        for step in range(num_steps): # for each temperature, perform num_steps sweeps
            metropolis_sweep(current_grid, L, beta, q)
            if step >= 50:  # Discard initial steps 
                total_energy = calculate_total_energy(current_grid)
                average_energy = total_energy / (L * L)  # Normalize energy per spin
                #energies_step.append(average_energy) # Dont do this here anymore, keep the full energy simulation
                energies.append([T, average_energy])

        #energies.append([T, np.mean(energies_step)]) # Skip this step
        np.savetxt(filename, energies) # Save energies to a text file


def calculate_mean_and_error(filename, block_sizes):
    data = np.loadtxt(filename)
    temperatures = data[:, 0]
    energies = data[:, 1]
    unique_temperatures = np.unique(temperatures)

    output = []
    for T in unique_temperatures:
        energy_at_T = energies[temperatures == T]
        mean_energy = np.mean(energy_at_T)

        standard_errors = [] # stores the SE for each block size
        for block_size in block_sizes: # calculate SE for different each block size
            num_blocks = len(energy_at_T) // block_size
            block_means = np.array([np.mean(energy_at_T[i * block_size:(i + 1) * block_size]) for i in range(num_blocks)])
            SE = np.sqrt(np.var(block_means, ddof = 1) / num_blocks)
            standard_errors.append(SE)
        output.append([T, mean_energy] + standard_errors)
    return np.array(output)

"""def try_different_block_sizes(filename, block_sizes):
    # block_sizes: array with the block sizes one wants to calculate error for
    for i in range(num_block_sizes):
        calculate_mean_and_error()"""

# Use file with simulated phase transition data to print possible block sizes where all blocks will have the same size
def print_possible_block_sizes(filename):
    data = np.loadtxt(filename)
    unique_T, counts = np.unique(data[:, 0], return_counts=True) # Get the amount of simulated energies for each temperature
    num_energies = counts[0]
    for n in range(1, num_energies + 1):
        if num_energies % n == 0:
            print(n)





"""def block_error(filename, block_size):
    energies = np.loadtxt(filename)
    # for T in energies[0]: # For each temperature in the file
    # calculate the mean energy (overall)
    # divide the samples into blocks - make sure that the blocks have the same size (think about this later)
    # return array (3 columns) with T, mean, df
    num_blocks = len(energies) // block_size
    block_means = np.array([np.mean(energies[i * block_size:(i + 1) * block_size]) for i in range(num_blocks)])
    mean_energy = np.mean(block_means)
    error = np.sqrt(np.var(block_means, ddof = 1) / num_blocks)
    print(f"Mean energy (hot start): {mean_energy:.4f} ± {error:.4f}")
    return mean_energy, error"""

def make_time_evolution_plot(filename_hot, filename_cold):
    energies_hot = np.loadtxt(filename_hot)
    energies_cold = np.loadtxt(filename_cold)

    plt.plot(energies_hot, label="Hot Start")
    plt.plot(energies_cold, label="Cold Start")
    plt.xlabel("Monte Carlo Sweeps")
    plt.ylabel("Energy per spin (E/N)")
    plt.title("Time Evolution of Energy per Spin")
    plt.legend()
    plt.show()

def make_phase_transition_plot(filename_q2, filename_q10):
    data_q2 = np.loadtxt(filename_q2)
    data_q10 = np.loadtxt(filename_q10)

    Tc_q2 = 1 / np.log(1 + np.sqrt(2))  # Critical temperature for q=2
    Tc_q10 = 1 / np.log(1 + np.sqrt(10))

    plt.plot(data_q2[:, 0], data_q2[:, 1], '.', color = 'green', label="q=2")
    plt.plot(data_q10[:, 0], data_q10[:, 1], '.', color = 'blue', label="q=10")
    plt.axvline(Tc_q2, color = 'green', linestyle="--", label=f"$T_c$ (q=2)")
    plt.axvline(Tc_q10, color = 'blue', linestyle="--", label=f"$T_c$ (q=10)")
    plt.title("Phase Transition for q=2 and q=10")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Energy per spin (E/N)")
    plt.legend()
    plt.show()

# Make phase transition plots with errorbars using different block sizes
def make_phase_transition_plot_with_errorbars(data_q2, data_q10, block = 2):
    # the argument block determines which block to look at
    Tc_q2 = 1 / np.log(1 + np.sqrt(2))  # Critical temperature for q=2
    #Tc_q10 = 1 / np.log(1 + np.sqrt(10))

    #plt.plot(data_q2[:, 0], data_q2[:, 1], '.', color = 'green', label="q=2")
    #plt.plot(data_q10[:, 0], data_q10[:, 1], '.', color = 'blue', label="q=10")
    plt.errorbar(data_q2[:, 0], data_q2[:, 1], data[:, block], fmt = ".")
    plt.axvline(Tc_q2, color = 'green', linestyle="--", label=f"$T_c$ (q=2)")
    #plt.axvline(Tc_q10, color = 'blue', linestyle="--", label=f"$T_c$ (q=10)")
    plt.title("Phase Transition for q=2 and q=10")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Energy per spin (E/N)")
    plt.legend()
    plt.show()

def produce_block_plots(data):
    T = data[:, 0]
    mean = data[:, 1]

    for i in range(len(data[0])-2): # loop through SE for the different block sizes
        print(i)

    


   
# Run simulation 
#simulation(16, 2, 1.0/3.0, 500, "hot", filename="hot_start.txt") # Temp = 3
#simulation(16, 2, 1.0/3.0, 500, "cold", filename="cold_start.txt")
#phase_transition_simulation(10, 2, 100, "cold", filename="phase_transition_q2.txt")
#phase_transition_simulation(10, 10, 100, "cold", filename="phase_transition_q10.txt")

# Run some caluclations on phase transition simulation data
#print_possible_block_sizes("phase_transition_q10.txt")
#output = calculate_mean_and_error("phase_transition_q10.txt", 5)
#try_different_block_sizes("phase_transition_q10.txt", 5)

data = calculate_mean_and_error("phase_transition_q10.txt", block_sizes = [5, 10])
#for i in range(50):
    #print(data[i])

#produce_block_plots(data)
make_phase_transition_plot_with_errorbars(data, data, block = 3)

# Make plots
#make_time_evolution_plot("hot_start.txt", "cold_start.txt")
#make_phase_transition_plot("phase_transition_q2.txt", "phase_transition_q10.txt")   








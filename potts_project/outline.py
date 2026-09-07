# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 08:33:43 2026
@author: andrearaaschou

Outline of the different parameters needed in the project
"""
import numpy as np
import  matplotlib.pyplot as plt


q = 2 # possible spin values
T_low = -10 # temperature
T_high = 100
Tc = 1 / np.log(1 + np.sqrt(q)) # critical temperature
beta = 1/T_low
J = 1 # coupling parameter

# Assume a square lattice with N = L*L sites and periodic boundary conditions
# state space/spin configuration, cold start, L = 10
L = 10 # number of rows and columns in lattice matrix
N = L**2 # Total number of spins
s = np.zeros((L, L)) 



# Energy (E) is given by
E = -J*np.sum(1) # in the sum, ad neighboring pairs function
energy_per_spin = E/N # use this to plot the time evolution with cold and hot starts

# Boltzmann distrubution
Z = np.sum(np.exp(-beta * E)) # Z: total weight of all possible configurations
p = np.exp(-beta * E) / Z # p: probability of a particular configuration

# One elementary update:
    # Pick one spin and attempt to change it using the Metropolis algorithm.

# Write some values to a file at regular intervals
N_meas = 10 # Choose this parameter to control how far apart measurements should be
recording_interval = N_meas * L**2


# Introduce a navigator that takes the periodic boundary conditions into account




# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:11:51 2026
@author: andrearaaschou

Sketch of some of the functions that could be useful in the project
"""
import numpy as np
import  matplotlib.pyplot as plt

def cold_start(L, q):
    """
    Creates a cold start initial state space. Randomly selects one of the 
    possible values for spins and fills the whole lattice with that value.

    Parameters
    ----------
    L : int
        Dictates size of configuration matrix (number of rows/columns).
    q : int
        possible values for spins.

    Returns
    -------
    np.array
        s.

    """
    value = np.random.choice(q) # choose random value to start with
    return np.full((L, L), value) 


def hot_start(L, q):
    """
    Creates a hot start initial state space. Randomly selects values from q
    and fills the lattice (LxL matrix)

    Parameters
    ----------
    L : int
        Dictates size of configuration matrix (number of rows/columns).
    q : int
        possible values for spins.

    Returns
    -------
    np.array
        s.

    """
    return np.random.randint(0, q, size=(L, L))


def plot_time_evolution(E, N):
    # Plot time evolution of E/N in runs with hot and cold starts
    # See if E/N converges to a common level
    plt.show()

# Working
def compute_E(s):
    matching_pairs = 0 # used to keep track of number of matching neighboring points
    L = s.shape[0]
    
    # go through s and look at all neighorin gpoints that 
        # 1) are to the right for each point (including the last point in each row)
        # 2) are below the current point including the lowest point
    # When a match is found, add 1 to sum
        
    for i in range(L):
        for j in range(L):
            point = s[i,j]
            point_to_the_right = s[i, (j+1) % L]
            point_below = s[(i+1) % L, j]
            
            #print(f"{point}, {point_to_the_right}")
            #print(f"{point}, {point_below}")
            
            if point == point_to_the_right:
                matching_pairs += 1 
                #print("number of matching pairs:")
                #print(matching_pairs)
            
            if point == point_below:
                matching_pairs += 1 
                #print("number of matching pairs:")
                #print(matching_pairs)
    
    E = - matching_pairs
    return E
    
def compute_delta_E(s, i, j, new_spin):
    """
    This function computes delta E by only looking at number of matching neighboring
    points to the spin that might change.

    Parameters
    ----------
    s : 2D array
        current state space
    i : int
        row-coordinate for selected spin 
    j : int
        column-coordinate for selected spin
    new_spin : int
        new possible value for selected spin

    Returns
    -------
    delta_E : int
        change in energy (E) if the proposed change is accepted

    """
    L = s.shape[0]
    
    old_matching_pairs = 0
    new_matching_pairs = 0
    
    if s[i, j] == s[i, s[i, (j+1) % L]]: 
        old_matching_pairs += 1
    if s[i, j] == s[i, s[i, (j-1) % L]]: 
        old_matching_pairs += 1
    if s[i, j] == s[i, s[(i+1) % L, j]]: 
        old_matching_pairs += 1
    if s[i, j] == s[i, s[(i-1) % L, j]]: 
        old_matching_pairs += 1
        
    if new_spin == s[i, s[i, (j+1) % L]]: 
        new_matching_pairs += 1
    if new_spin == s[i, s[i, (j-1) % L]]: 
        new_matching_pairs += 1
    if new_spin == s[i, s[(i+1) % L, j]]: 
        new_matching_pairs += 1
    if new_spin == s[i, s[(i-1) % L, j]]: 
        new_matching_pairs += 1
    
    delta_E = old_matching_pairs - new_matching_pairs

    # remember to update E somewhere else after using this function
    return delta_E



s_cold = cold_start(10, 4)
s_hot = hot_start(10, 4)

compute_E(s_hot)
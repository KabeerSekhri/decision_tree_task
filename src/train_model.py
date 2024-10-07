import numpy as np
import pandas as pd 


def potential_splits(data):
    splits = {}
    n_col = data.shape[1]
    for i in range(n_col - 1):  # Exclude last column (target)
        splits[data.columns[i]] = np.unique(data[data.columns[i]])  # Store column names
    
    return splits

# Function to split data based on column name and value
def split_data(data, split_col, split_val):
    # Ensure split_col is treated as a string (column name)
    left_data = data[data[split_col] <= split_val]
    right_data = data[data[split_col] > split_val]
    
    return left_data, right_data

def entropy(data): # Takes dataset instead of column
    target_col = dataset[dataset.columns[-1]]  # Makes last column of dataset the target column
    counts = []
    target_dict = target.value_counts().to_dict()
    for k,v in target_dict.items():
        counts.append(v)
    counts = np.array(counts)

    p = counts / counts.sum() # Does calculation at once, so faster
    ent = -np.sum(p * np.log2(p))

    return ent

def total_entropy(left_child, right_child):
    total = len(left_child) + len(right_child)

    p_left = len(left_child)/total
    p_right = len(right_child)/total

    ent_tot = (p_left * entropy(left_child)) + (p_right * entropy(right_child)) # Can use asame ent func. this since entropy now takes subset of dataset

    return ent_tot


def get_best_split(data, pot_splits):
    best_ent = np.inf # Initialise entropy

    for k in pot_splits:
        for v in pot_splits[k]:
            left_child, right_child = split_data(data,k,v)
            current_ent = total_entropy(left_child,right_child)

            if current_ent < best_ent:
                best_ent = current_ent
                best_col = k
                best_val = v

    return best_col, best_val
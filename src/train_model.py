import numpy as np
import pandas as pd 


# Find column indices and all potential splits at said indies
def potential_splits(data):
    splits = {}
    n_col = data.shape[1]
    for i in range(n_col - 1):  # Exclude last column (target)
        splits[data.columns[i]] = np.unique(data[data.columns[i]])  # Store column names
    
    return splits

# Function to split data based on column name and value
def split_data(data, split_col, split_val):
    left_data = data[data[split_col] <= split_val]
    right_data = data[data[split_col] > split_val]
    
    return left_data, right_data

def entropy(data): # Takes dataset instead of column
    target_col = data[data.columns[-1]]  # Makes last column of dataset the target column
    counts = []
    target_dict = target_col.value_counts().to_dict()
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


# Get best column and attribute in that column to split at
def get_best_split(data, pot_splits):
    best_ent = np.inf  # Initialize entropy
    best_col = None    # Initialize best column
    best_val = None    # Initialize best value

    for k in pot_splits:
        for v in pot_splits[k]:
            left_child, right_child = split_data(data, k, v)
            current_ent = total_entropy(left_child, right_child)

            if current_ent < best_ent:
                best_ent = current_ent
                best_col = k
                best_val = v

    return best_col, best_val

def check_pure(data):
    if len(np.unique(data[data.columns[-1]]))==1:
        return True
    else:
        return False

def decision_tree(data, min_samples, max_depth):
    
    depth = 0
    if (check_pure(data)==True) or (len(data)<=min_samples) or (depth >= max_depth):
        return target.unique()[0]
    
    else: 
        depth +=1
        # Determining the child nodes
        pot_splits = potential_splits(data)
        split_col, split_val = get_best_split(data, pot_splits)
        left_child, right_child = split_data(data, split_col, split_val)

        # Making the tree
        cond = f"{split_col} <= {split_val}"
        sub_tree = {cond: []} # Tree is in form of dictionary

        node_yes = decision_tree(left_child, min_samples,max_depth)
        node_no = decision_tree(right_child, min_samples,max_depth)

        sub_tree[cond].append[node_yes]
        sub_tree[cond].append[node_no]

        return sub_tree
    
# Classify from an example
def classify(example,tree):
  cond = list(tree.keys())[0]
  feature, compare, val = cond.split()

  # Split condition
  if example[feature] <= float(val):
    answer = tree[cond][0]
  else:
    answer = tree[cond][1]

  # See if label reached
  if type(answer) != dict:
   return answer
  else:
   return classify(example,answer) # Answer is the remaining part of the tree


# Accuracy of classifictions
def accuracy(data, tree):
    data["classification"] = data.apply(classify, axis=1, args=(tree,)) # Classify each row using the decision tree
    data["correctness"] = data["classification"] == data['isFraud'] # Compare classification with the actual labels
    
    accuracy_score = data["correctness"].mean() # Calculate accuracy as the mean of correctness

    return accuracy_score
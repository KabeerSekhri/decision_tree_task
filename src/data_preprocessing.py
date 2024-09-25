import numpy as np 
import pandas as pd 

dataset = pd.read_csv("../data/training_data.csv")

dataset.columns = dataset.columns.str.strip()  # Removes leading and trailing spaces in titles
#dataset.isnull().sum() # Check for missing values
#dataset.duplicated().sum() # Check for duplicate values

dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)

#dataset.corr()['FUEL CONSUMPTION'] # To find correlation

X = dataset[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']] # Features (independent)
Y = dataset['isFraud'].values.reshape(-1, 1) # Target (dependent)
X = np.c_[np.ones(X.shape[0]), X] # Adding a column of 1s for constant in matrix multiplication

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

RANDOM_STATE = 12345

# Load Data (NLSY)
df = pd.read_stata('http://www.stata-press.com/data/r10/nlswork.dta')
df = df.dropna()

Y = df['ln_wage']
df.drop('ln_wage', axis='columns', inplace=True)
D = df['union']
df.drop('union', axis='columns', inplace=True)
X = df

# convert to pytorch tensor#
Y = torch.tensor(Y.values.astype(np.float32)).unsqueeze(-1)
D = torch.tensor(D.values.astype(np.float32)).unsqueeze(-1)
X = torch.tensor(X.values.astype(np.float32))

# split into A and B set
dataset_size = Y.size()[0]
a_index, b_index = train_test_split(range(dataset_size), test_size=0.5, random_state=RANDOM_STATE)
a_index_train, a_index_test, b_index_train, b_index_test = train_test_split(a_index, b_index, test_size=0.2, random_state=RANDOM_STATE)
Y_a_train, D_a_train, X_a_train = Y[a_index_train], D[a_index_train], X[a_index_train]
Y_a_test, D_a_test, X_a_test = Y[a_index_test], D[a_index_test], X[a_index_test]
Y_b_train, D_b_train, X_b_train = Y[b_index_train], D[b_index_train], X[b_index_train]
Y_b_test, D_b_test, X_b_test = Y[b_index_test], D[b_index_test], X[b_index_test]


#save preprocessed data
torch.save(Y_a_train, 'Data/Preprocessed/NLSY/Y_a_train')
torch.save(Y_a_test, 'Data/Preprocessed/NLSY/Y_a_test')
torch.save(Y_b_train, 'Data/Preprocessed/NLSY/Y_b_train')
torch.save(Y_b_test, 'Data/Preprocessed/NLSY/Y_b_test')

torch.save(D_a_train, 'Data/Preprocessed/NLSY/D_a_train')
torch.save(D_a_test, 'Data/Preprocessed/NLSY/D_a_test')
torch.save(D_b_train, 'Data/Preprocessed/NLSY/D_b_train')
torch.save(D_b_test, 'Data/Preprocessed/NLSY/D_b_test')

torch.save(X_a_train, 'Data/Preprocessed/NLSY/X_a_train')
torch.save(X_a_test, 'Data/Preprocessed/NLSY/X_a_test')
torch.save(X_b_train, 'Data/Preprocessed/NLSY/X_b_train')
torch.save(X_b_test, 'Data/Preprocessed/NLSY/X_b_test')



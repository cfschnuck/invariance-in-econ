import pandas as pd
import numpy as np
import torch

from helper_functions_preprocessing import save_train_test_data

PATH = 'Data/Preprocessed/NLSY/'

RANDOM_STATE = 12345

# Load Data (NLSY)
df = pd.read_stata('http://www.stata-press.com/data/r10/nlswork.dta')
df = df.dropna()

Y = df['ln_wage']
df.drop('ln_wage', axis='columns', inplace=True)
D = df['union']
df.drop('union', axis='columns', inplace=True)
X = df[['age', 'year', 'race', 'msp','collgrad', 'nev_mar', 'grade', 'not_smsa', 'south' ]]

# convert to pytorch tensor#
Y = torch.tensor(Y.values.astype(np.float32)).unsqueeze(-1)
D = torch.tensor(D.values.astype(np.float32)).unsqueeze(-1)
X = torch.tensor(X.values.astype(np.float32))

save_train_test_data(Y, D, X, PATH, RANDOM_STATE)



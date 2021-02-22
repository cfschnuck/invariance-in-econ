import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

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

plt.figure()
plt.scatter(D, Y, s=2, alpha=0.6)
plt.xlim(-0.5, 1.5)
plt.title("Log wage Y on union status D")
plt.ylabel("Log wage Y")
plt.xlabel("Union status D")
plt.tight_layout()
plt.savefig(PATH + 'DY_scatter.png')

save_train_test_data(Y, D, X, PATH, RANDOM_STATE)



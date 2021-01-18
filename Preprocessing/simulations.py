import torch
from torch.distributions.uniform import Uniform
from helper_functions_preprocessing import save_train_test_data

PATH = 'Data/Preprocessed/Simulation/'

# simulate data set
RANDOM_STATE = 12345
N_F = int(4)
N_X = int(5)
N_SAMPLES = int(1e5)

# sample ground truth factors for D
u = Uniform(0, 1) # for tanh activation function in encoder
f_factors = u.sample((N_SAMPLES, N_F))
x_factors = u.sample((N_SAMPLES, N_X))

# make d some nonlinear function of f and x, then treshold it
d_f = f_factors[:,0]**2 - f_factors[:,1] + torch.log(f_factors[:,2] + 1)
d_x = x_factors[:,0]**2 - 4 * x_factors[:,1] - x_factors[:,2] - x_factors[:,3]**2 + torch.log(x_factors[:,4] + 1)

d = d_x + d_f

y_x = 3 * x_factors[:,1] + (x_factors[:,2])**2 - torch.log(x_factors[:,0] + 5)

y = 4 * d + y_x

save_train_test_data(y.unsqueeze(-1), d.unsqueeze(-1), x_factors, PATH, RANDOM_STATE)

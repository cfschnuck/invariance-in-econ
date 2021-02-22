import torch
from torch.distributions.uniform import Uniform
from helper_functions_preprocessing import save_train_test_data
import matplotlib.pyplot as plt

PATH = 'Data/Preprocessed/Simulation/'

# simulate data set
torch.manual_seed(12345)
RANDOM_STATE = 12345
N_F = int(3)
N_X = int(5)
N_SAMPLES = int(1e4)

# sample ground truth factors for D
u = Uniform(0, 1) # for tanh activation function in encoder
f_factors = u.sample((N_SAMPLES, N_F))
x_factors = u.sample((N_SAMPLES, N_X))

# sample noise for D with standard deviation 0.5
e_d = torch.randn(N_SAMPLES) * 0.2
# sample noise for Y with standard deviation 0.5
e_y = torch.randn(N_SAMPLES) * 0.2

# make d some nonlinear function of f and x, then treshold it
d_f = f_factors[:,0]**2 - f_factors[:,1] + torch.log(f_factors[:,2] + 1)
d_x = x_factors[:,0]**2 - 4 * x_factors[:,1] - x_factors[:,2] - x_factors[:,3]**2 + torch.log(x_factors[:,4] + 2)

d = d_x + d_f + e_d
d = (d - torch.mean(d)) / torch.std(d)
d = (torch.sign(d) + 1) / 2

y_x = 3 * x_factors[:,1] + (x_factors[:,2])**2 - torch.log(x_factors[:,0] + 5)

y = 4 * d + y_x + e_y

# plot D and Y data in scatter plot
plt.figure()
plt.scatter(d, y, s=2, alpha=0.6)
plt.xlim(-0.5, 1.5)
plt.title("Outcome variable Y on  treatment D")
plt.ylabel("Outcome variable Y")
plt.xlabel("Treatment D")
plt.tight_layout()
plt.savefig(PATH + 'DY_scatter.png')

save_train_test_data(y.unsqueeze(-1), d.unsqueeze(-1), x_factors, PATH, RANDOM_STATE)

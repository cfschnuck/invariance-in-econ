import torch
from torch.distributions.uniform import Uniform
import math

def sample_from_latent(size_1, size_2):
    u = Uniform(-1, 1) # for tanh activation function in encoder
    return u.sample(size_1), u.sample(size_2)

def prior_loss_f(z_mean, z_log_var):
    loss = (-0.5) * (1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)).mean(dim=0)
    return loss.sum()

# from https://github.com/dcmoyer/invariance-tutorial/blob/master/src/kl_tools.py
def kl_qzx_qz_loss_f(z_mean, z_log_var):
    z_sigma = torch.exp( 0.5 * z_log_var )
    z_dim = z_mean.size()[1]
    all_pairs_GKL = all_pairs_gaussian_kl(z_mean, z_sigma, True) - 0.5 * z_dim
    return torch.mean(all_pairs_GKL)
    

def all_pairs_gaussian_kl(mu, sigma, add_third_term = False):
    sigma_sq = sigma ** 2 + 1e-8
    sigma_sq_inv = torch.reciprocal(sigma_sq)
    first_term = torch.matmul(sigma_sq, torch.transpose(sigma_sq_inv, 0, 1))
    r = torch.matmul(mu * mu, torch.transpose(sigma_sq_inv, 0, 1))
    r2 = mu * mu * sigma_sq_inv
    r2 = torch.sum(r2, 1)
    second_term = 2 * torch.matmul(mu, torch.transpose(mu*sigma_sq_inv, 0, 1))
    second_term = r - second_term + torch.transpose(r2.unsqueeze(-1), 0, 1)
    if add_third_term:
        r = torch.sum(torch.log(sigma_sq), 1)
        r = r.reshape(-1, 1)
        third_term = r - torch.transpose(r, 0, 1)
    else:
        third_term = 0
    return 0.5 * ( first_term + second_term + third_term )


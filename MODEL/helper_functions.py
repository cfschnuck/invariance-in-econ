from torch.distributions.uniform import Uniform

def sample_from_latent(size_1, size_2):
    u = Uniform(-1, 1) # for tanh activation function in encoder
    return u.sample(size_1), u.sample(size_2) 
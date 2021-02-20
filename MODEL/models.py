from torch import nn

class InvarSennM1(nn.Module):
    def __init__(self, autoencoder, predictor):
        super(InvarSennM1, self).__init__()
        self.autoencoder = autoencoder
        self.predictor = predictor

    def forward(self, x):
        e1, e2, x_reconstructed = self.autoencoder(x)
        pred = self.predictor(e1)
        return pred, (e1, e2), x_reconstructed

class InvarSennM2(nn.Module):
    def __init__(self, disentangler1, disentangler2):
        super(InvarSennM2, self).__init__()
        self.disentangler1 = disentangler1 # predict e2 from e1
        self.disentangler2 = disentangler2 # predict e1 from e2

    def forward(self, e1, e2):
        e2_reconstructed = self.disentangler1(e1)
        e1_reconstructed = self.disentangler2(e2)
        return e1_reconstructed, e2_reconstructed

# class InvarSennM(nn.Module):
#     def __init__(self, m1, m2):
#         super(InvarSennM, self).__init__()
#         self.m1 = m1
#         self.m2 = m2

#     def forward(self, x):
#         pred, (e1, e2), x_reconstructed = m1(x)
#         e1_reconstructed, e2_reconstructed = m2(e1, e2)
#         return (pred, (e1, e2), x_reconstructed), (e1_reconstructed, e2_reconstructed)

class VAEModel(nn.Module):
    def __init__(self, autoencoder):
        super(VAEModel, self).__init__()
        self.autoencoder = autoencoder

    def forward(self, targets, x = None):
        z_mean, z_log_var, target_reconstructed = self.autoencoder(targets, x)
        return z_mean, z_log_var, target_reconstructed
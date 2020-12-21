import torch
import torch.nn.functional as F

def disentangle_loss(e1, pred1, e2, pred2):
    return F.mse_loss(e1, pred1) + F.mse_loss(e2, pred2) 

# def accuracy(pred, y):
#         return torch.mean((pred.argmax(axis=1) == y).float()).item()
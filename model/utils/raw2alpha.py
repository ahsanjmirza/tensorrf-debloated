import torch

def raw2alpha(sigma, dist):
    alpha = 1. - torch.exp(-sigma*dist)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
    weights = alpha * T[:, :-1]
    return alpha, weights, T[:,-1:]
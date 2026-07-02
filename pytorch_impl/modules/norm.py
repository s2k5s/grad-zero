import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.dim = dim
        self.epsilon = eps
        # we should initialize gamma to 1 creating an identity transform
        # we want the data to flow through the network as smoothly and 
        # neutrally as possible so the gradients don't explode or vanish
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        x : dim --> (B, L, D)
        x * x : dim --> (B, L, D)
        mean_sq --> (B, L, 1)
        rsqrt --> (B, L, 1) --> (B, L, D) broadcasted 
        """
        mean_sq = torch.mean((x*x), dim=-1, keepdim=True)
        res = x * torch.rsqrt(mean_sq + self.epsilon) * self.gamma #element wise multiplication
        return res


class LayerNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.epsilon = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)

        res = self.gamma * (x - mean) * torch.rsqrt(var + self.epsilon) + self.beta
        return res

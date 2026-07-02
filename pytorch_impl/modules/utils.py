import torch
import torch.nn as nn
# Just to demonstrate the math behind these functions

def sigmoid(x):
    res = 1 / (1 + torch.exp(-x))
    return res

def silu(x):
    return x * sigmoid(x)

def relu(x):
    res = torch.where(x >= 0, x, 0)
    return res


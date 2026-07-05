import torch
import torch.nn as nn
import torch.nn.functional as F
# Why SwiGLU: Replaces the static thresholding of ReLU/GELU with a dynamic, multiplicative gate.
# By using one projection to conditionally scale another, it acts as feature-level attention.
# The smooth, non-monotonic SiLU activation improves gradient flow and consistently lowers perplexity.


class swigluFFN(nn.Module):
    """
    Llama 3 style feed-forward block: SwiGLU-gated MLP.
    output --> down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, d_model, multiple_of=256, ffn_dim_multiplier=None):
        """
        start from the standard 4x expansion, then shrink by 2/3 since SwiGLU
        has 3 weight matrices instead of 2, keeping total params comparable
        to a plain (non-gated) FFN of the same width
        
        total params in traditional FFN = 2(tensors) * 4 * d^2 = 8d^2
        total params with swiglu = 3 * 8d^2 / 3 = 8d^2 as here we have an
        extra gating tensor
        """
        
        super().__init__()
        hidden_dim = 4 * d_model
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        remainder = -hidden_dim % multiple_of #how much room is left to complete the next multiple of 256
        hidden_dim = hidden_dim + remainder

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))



class reluFFN(nn.Module):
    def __init__(self, d_model, multiple_of=256, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = 4 * d_model
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        remainder = -hidden_dim % multiple_of #how much room is left to complete the next multiple of 256
        hidden_dim = hidden_dim + remainder
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)))
    
class geluFFN(nn.Module):
    def __init__(self, d_model, multiple_of=256, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = 4 * d_model
        if ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * ffn_dim_multiplier)
        hidden_dim = (-hidden_dim % multiple_of) + hidden_dim
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))

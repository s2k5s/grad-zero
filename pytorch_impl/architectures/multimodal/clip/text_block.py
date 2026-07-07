import torch
import torch.nn as nn
from pytorch_impl.modules.attention import NonCausalMHSA
from pytorch_impl.modules.feedforward import swigluFFN
from pytorch_impl.modules.norm import RMSNorm

class TextBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = NonCausalMHSA(config.attention)
        self.ffn = swigluFFN(config.attention.d_model, config.ffn_multiple_of, config.ffn_dim_multiplier)
        self.pre_attn_RMSNorm = RMSNorm(config.attention.d_model, config.norm_eps)
        self.pre_ffn_RMSNorm = RMSNorm(config.attention.d_model, config.norm_eps)

    def forward(self, x, freq_cis, padding_mask=None):
        h = x + self.attn(self.pre_attn_RMSNorm(x), freq_cis, padding_mask)
        out = h + self.ffn(self.pre_ffn_RMSNorm(h))
        return out
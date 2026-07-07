import torch
import torch.nn as nn
from pytorch_impl.architectures.vision.vit.config import ViTConfig
from pytorch_impl.modules.attention import NonCausalMHSA
from pytorch_impl.modules.feedforward import geluFFN
from pytorch_impl.modules.norm import LayerNorm


class Block(nn.Module):
    def __init__(self, config : ViTConfig):
        super().__init__()
        self.attn = NonCausalMHSA(config.attention)
        self.ffn = geluFFN(config.attention.d_model, config.ffn_multiple_of, config.ffn_dim_multiplier)
        self.pre_attn_LayerNorm = LayerNorm(config.attention.d_model, config.norm_eps)
        self.pre_ffn_LayerNorm = LayerNorm(config.attention.d_model, config.norm_eps)

    def forward(self, x, freq_cis, padding_mask=None):
        h = x + self.attn(self.pre_attn_LayerNorm(x), freq_cis, padding_mask)
        out = h + self.ffn(self.pre_ffn_LayerNorm(h))
        return out
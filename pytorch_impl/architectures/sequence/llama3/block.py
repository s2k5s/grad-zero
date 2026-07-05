import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_impl.modules.attention import GQA
from pytorch_impl.modules.feedforward import swigluFFN
from pytorch_impl.modules.norm import RMSNorm
from pytorch_impl.architectures.sequence.llama3.config import LMConfig

class Block(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.attn = GQA(config.attention)
        self.attn_norm = RMSNorm(config.attention.d_model, config.norm_eps)
        self.ffn = swigluFFN(config.attention.d_model, config.ffn_multiple_of, config.ffn_dim_multiplier)
        self.ffn_norm = RMSNorm(config.attention.d_model, config.norm_eps)

    def forward(self, x, layer_num, freq_cis, padding_mask=None, KV_cache=None):
        """
        pre_attn_norm --> attn(with residual connection) --> pre_ffn_norm --> ffn(with residual connection)
        """
        h = x + self.attn(self.attn_norm(x), layer_num, freq_cis, padding_mask, KV_cache)
        out = h + self.ffn(self.ffn_norm(h))
        return out

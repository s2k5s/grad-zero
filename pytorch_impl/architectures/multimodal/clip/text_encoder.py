import torch
import torch.nn as nn
from pytorch_impl.modules.attention import NonCausalMHSA
from pytorch_impl.modules.feedforward import swigluFFN
from pytorch_impl.modules.norm import RMSNorm
from pytorch_impl.architectures.multimodal.clip.config import TextEncoderConfig
from pytorch_impl.architectures.multimodal.clip.text_block import TextBlock
from pytorch_impl.modules.positional_embeddings import compute_freq_cis

class TextEncoder(nn.Module):
    def __init__(self, config: TextEncoderConfig, joint_embed_dim: int):
        super().__init__()
        self.text_embedding = nn.Embedding(config.vocab_size, config.attention.d_model)
        self.blocks = nn.ModuleList([TextBlock(config) for _ in range(config.attention.num_layers)])
        self.final_norm = RMSNorm(config.attention.d_model, config.norm_eps)
        self.projection = nn.Linear(config.attention.d_model, joint_embed_dim, bias=False)
        head_dim = config.attention.d_model // config.attention.num_heads
        freq_cis = compute_freq_cis(head_dim, config.attention.max_seq_len, config.rope_theta)
        self.register_buffer("freq_cis", freq_cis, persistent=False)

    def forward(self, tokens, padding_mask=None):
        """
        original clip paper didn't use meanpooling but the End of text(EOT) token
        as they were using causal attention. Here causality is not required hence 
        we use Bidirectional attention, each token has interacted with every other
        token, hence we do meanpooling over all the tokens in the sequence
        """
        x = self.text_embedding(tokens)
        for block in self.blocks:
            x = block(x, self.freq_cis, padding_mask)

        x = self.final_norm(x)
        x = self.projection(x)
        # handle the padding tokens first
        mask = torch.ones_like(x).to(x.dtype)
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).to(x.dtype) #padding mask=true means we need to skip that pos
        x = x * mask # element wise multiplication --> padded token embeddings sent to 0
        norm = torch.sum(mask, dim=1).clamp(min=1e-9)
        pooled_tensor = torch.sum(x, dim=1) / norm #calculates the mean pooled tensor

        return pooled_tensor

        

        


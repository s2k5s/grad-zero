import torch
import torch.nn as nn
from pytorch_impl.architectures.sequence.llama3.block import Block
from pytorch_impl.architectures.sequence.llama3.config import LMConfig
from pytorch_impl.modules.positional_embeddings import compute_freq_cis
from pytorch_impl.modules.norm import RMSNorm


class LM(nn.Module):
    """LM style decoder-only transformer: token embedding, stacked GQA+SwiGLU blocks, final norm, lm_head."""
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.attention.d_model) # sets trainable embeddings
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.attention.num_layers)])
        self.final_norm = RMSNorm(config.attention.d_model, config.norm_eps)
        # projecting to vocab dimension to form logits
        self.lm_head = nn.Linear(config.attention.d_model, config.vocab_size, bias=False)

        head_dim = config.attention.d_model // config.attention.num_q_heads
        freq_cis = compute_freq_cis(head_dim, config.attention.max_seq_len, config.rope_theta)
        # pytorch moves all the registered parameters and registered buffers to new device
        # if we don't register freq_cis as a buffer (i.e. leave it as a plain attribute),
        # .to(device) won't move it, so our attention matrices will be on GPU while
        # freq_cis stays on CPU, and multiplying these two tensors would crash as they are on different devices
        # registered buffer acts as a middle layer between nn.parameters and normal torch tensor
        self.register_buffer("freq_cis", freq_cis, persistent=False)# part of model state but not learnable

    def forward(self, tokens, padding_mask=None, kv_cache=None):
        """
        tokens: (B, L) token ids for this step — L == 1 during decode, L > 1 during prefill.
        padding_mask: (B, L_total) bool mask, True at key positions to ignore.
        kv_cache: optional KV_cache shared across all layers/steps of one generation.
        """
        x = self.tok_embed(tokens)
        for layer_num, block in enumerate(self.blocks):
            x = block(x, layer_num, self.freq_cis, padding_mask, kv_cache)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

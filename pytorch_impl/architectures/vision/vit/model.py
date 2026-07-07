import torch
import torch.nn as nn
from pytorch_impl.architectures.vision.vit.block import Block
from pytorch_impl.architectures.vision.vit.config import ViTConfig
from pytorch_impl.modules.positional_embeddings import compute_2D_freq_cis
from pytorch_impl.modules.norm import LayerNorm
from pytorch_impl.architectures.vision.vit.patch_embed import PatchEmbedding

class ViT(nn.Module):
    """Vision Transformer encoder: patch embedding, stacked bidirectional attention+FFN blocks, final norm."""
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config.patch_size, config.num_channels, config.attention.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.attention.num_layers)])
        self.final_norm = LayerNorm(config.attention.d_model, config.norm_eps)
        head_dim = config.attention.d_model // config.attention.num_heads
        freq_cis = compute_2D_freq_cis(head_dim, config.max_num_patches_h, config.max_num_patches_w, config.rope_theta)
        self.register_buffer("freq_cis", freq_cis, persistent=False)

    def forward(self, img, padding_mask=None):
        """
        img: (B, C, H, W), H and W divisible by patch_size.
        returns: (B, num_patches, d_model) — one refined embedding per patch, in row-major order
        (num_patches = (H//patch_size) * (W//patch_size)). No pooling/CLS token, no output head —
        this is a general-purpose encoder; classification heads or vision-language projectors
        consume this sequence, not part of ViT itself.
        """

        _, _, H, W = img.shape
        num_h = H // self.config.patch_size
        num_w = W // self.config.patch_size

        # RoPE angles only depend on absolute (row, col) index, not on grid size, so the
        # top-left (num_h, num_w) sub-grid of the precomputed max grid is exactly correct —
        # reshape back to 2D, slice both axes (not a flat prefix), flatten again
        head_dim_half = self.freq_cis.shape[-1]
        freq_grid = self.freq_cis.reshape(self.config.max_num_patches_h, self.config.max_num_patches_w, head_dim_half)
        freq_cis = freq_grid[:num_h, :num_w, :].reshape(num_h * num_w, head_dim_half)

        x = self.patch_embedding(img)
        for block in self.blocks:
            x = block(x, freq_cis, padding_mask)
        out = self.final_norm(x)
        return out
from dataclasses import dataclass
from typing import Optional
from pytorch_impl.modules.attention import AttentionConfig


@dataclass
class ViTConfig:
    """ViT specific config: wraps the generic AttentionConfig plus patch/norm/FFN/RoPE hyperparameters."""
    attention: AttentionConfig
    patch_size: int
    max_image_height: int
    max_image_width: int
    num_channels: int = 3
    norm_eps: float = 1e-5
    ffn_multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_theta: float = 10000.0

    def __post_init__(self):
        assert self.max_image_height % self.patch_size == 0
        assert self.max_image_width % self.patch_size == 0
        self.max_num_patches_h = self.max_image_height // self.patch_size
        self.max_num_patches_w = self.max_image_width // self.patch_size

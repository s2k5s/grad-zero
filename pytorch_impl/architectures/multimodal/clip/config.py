import math
from dataclasses import dataclass, field
from typing import Optional
from pytorch_impl.modules.attention import AttentionConfig
from pytorch_impl.architectures.vision.vit.config import ViTConfig


@dataclass
class TextEncoderConfig:
    """CLIP text tower config: wraps AttentionConfig plus vocab/norm/FFN/RoPE hyperparameters. Sized independently of the vision tower."""
    attention: AttentionConfig
    vocab_size: int
    norm_eps: float = 1e-5
    ffn_multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_theta: float = 10000.0


@dataclass
class CLIPConfig:
    """Top-level CLIP config: independently-sized text and vision towers, projected into a shared joint embedding space."""
    text: TextEncoderConfig
    vision: ViTConfig
    joint_embed_dim: int
    logit_scale_init: float = field(default_factory=lambda: math.log(1 / 0.07))

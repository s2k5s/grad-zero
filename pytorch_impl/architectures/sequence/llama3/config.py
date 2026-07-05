from dataclasses import dataclass
from typing import Optional
from pytorch_impl.modules.attention import AttentionConfig


@dataclass
class LMConfig:
    """LM specific config: wraps the generic AttentionConfig plus norm/FFN/RoPE hyperparameters."""
    attention: AttentionConfig
    vocab_size: int
    norm_eps: float = 1e-5
    ffn_multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_theta: float = 500000.0

import torch
import torch.nn as nn
from pytorch_impl.architectures.multimodal.clip.config import CLIPConfig
from pytorch_impl.architectures.multimodal.clip.text_encoder import TextEncoder
from pytorch_impl.architectures.multimodal.clip.image_encoder import ImageEncoder


class CLIP(nn.Module):
    """Dual-encoder CLIP: independently-sized text and image towers projected into a shared joint embedding space."""
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.text_encoder = TextEncoder(config.text, config.joint_embed_dim)
        self.image_encoder = ImageEncoder(config.vision, config.joint_embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init))

    def forward(self, text_tokens, image, padding_mask=None):
        """
        text_tokens: (B, L) token ids. image: (B, C, H, W). padding_mask: (B, L) bool, True at positions to ignore.
        returns: (image_embeds, text_embeds, logit_scale) — both embeds are (B, joint_embed_dim), L2-normalized.
        logit_scale is the exponentiated learnable temperature; loss computation (e.g. contrastive_loss
        in loss.py) is left to the caller, not baked into the model itself.
        """
        text_embeds = self.text_encoder(text_tokens, padding_mask)
        image_embeds = self.image_encoder(image)

        eps = 1e-8
        text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + eps)
        image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + eps)

        return image_embeds, text_embeds, self.logit_scale.exp()

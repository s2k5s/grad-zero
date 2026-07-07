import torch.nn as nn
from pytorch_impl.architectures.vision.vit.model import ViT
from pytorch_impl.architectures.vision.vit.config import ViTConfig

class ImageEncoder(nn.Module):
    def __init__(self, config: ViTConfig, joint_embedding_dim):
        super().__init__()
        self.config = config
        self.joint_embedding_dim = joint_embedding_dim
        self.encoder = ViT(config)
        self.out = nn.Linear(config.attention.d_model, joint_embedding_dim, bias=False)

    def forward(self, img):
        encoder_out = self.encoder(img)  # (B, num_patches, d_model) — every image in the batch
        pooled = encoder_out.mean(dim=1)  # has the same H, W, so all patches are real, no padding to mask
        return self.out(pooled)

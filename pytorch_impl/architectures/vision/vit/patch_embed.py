import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, num_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.d_model = d_model
        self.patch_dim = num_channels*patch_size*patch_size
        self.flatten = nn.Linear(self.patch_dim, d_model)


    def forward(self, img):
        B, C, H, W = img.shape
        p = self.patch_size
        
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        num_h = H//self.patch_size
        num_w = W//self.patch_size
        # division is performed by the second dimension hence (., ., num_h, p,. ,.) and not (., ., p, num_h,. ,.)
        transformed = img.view(B, C, num_h, p, num_w, p)
        x = torch.permute(transformed, (0, 2, 4, 1, 3, 5))
        x = x.reshape(B, num_h*num_w, self.patch_dim)
        x = self.flatten(x)
        return x




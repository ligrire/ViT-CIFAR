import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# This model is modified from https://github.com/lucidrains/vit-pytorch

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, transformer, num_classes, channels=3, joint=True):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.joint = joint
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.transformer = transformer
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img1, img2=None):
        if not self.joint:
            x1 = self.to_patch_embedding(img1)
            b, n, _ = x1.shape
            x1 += self.pos_embedding[:, :n]   
            x1 = self.transformer(x1)
            x1 = x1.mean(dim=1)
            return self.mlp_head(x1)
        x1 = self.to_patch_embedding(img1)
        x2 = self.to_patch_embedding(img2)
        b, n, _ = x1.shape
        x1 += self.pos_embedding[:, :n]
        x2 += self.pos_embedding[:, :n]
        x = torch.cat((x1, x2), dim=1)
        x = self.transformer(x)
        x = self.mlp_head(x)
        if self.training:
            return x.reshape(x.shape[0] * x.shape[1], -1)
        else:
            return x.mean(dim=1)

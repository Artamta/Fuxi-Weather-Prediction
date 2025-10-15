import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple

class CubeEmbedding3D(nn.Module):
    def __init__(self, in_channels: int = 70, embed_dim: int = 384, patch_size: Tuple[int, int, int] = (2, 4, 4)):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.projection = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,T,H,W); got {tuple(x.shape)}")
        B, C, T, H, W = x.shape
        if T != self.patch_size[0]:
            raise ValueError(f"Expected {self.patch_size[0]} timesteps; got {T}")
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels; got {C}")
        x = self.projection(x).squeeze(2)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

    def get_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        h, w = input_size
        return h // self.patch_size[1], w // self.patch_size[2]
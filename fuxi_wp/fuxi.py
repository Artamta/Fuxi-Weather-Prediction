import torch
import torch.nn as nn
from einops import rearrange

from cube_embedding import CubeEmbedding3D
from swin_blocks import SwinBlockStack


class UpsampleDecoder(nn.Module):
    def __init__(self, in_dim: int, out_channels: int, target_shape=(16, 8)):
        super().__init__()
        self.target_shape = target_shape
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(in_dim // 4, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return nn.functional.interpolate(
            x, size=self.target_shape, mode="bilinear", align_corners=False
        )


class FuXiTinyBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        depths=(6, 6),
        num_heads=(6, 6),
        input_shape=(720, 1440),
        window_size: int = 8,
        in_channels: int = 70,
        output_channels: int = 70,
    ):
        super().__init__()
        self.embedding = CubeEmbedding3D(in_channels=in_channels, embed_dim=embed_dim)
        self.target_shape = input_shape

        spatial_size = (
            input_shape[0] // self.embedding.patch_size[1],
            input_shape[1] // self.embedding.patch_size[2],
        )
        self.spatial_size = spatial_size

        win = min(window_size, spatial_size[0], spatial_size[1])

        self.encoder = SwinBlockStack(
            embed_dim,
            depth=depths[0],
            heads=num_heads[0],
            input_resolution=spatial_size,
            window_size=win,
        )
        self.bottleneck = SwinBlockStack(
            embed_dim,
            depth=depths[1],
            heads=num_heads[1],
            input_resolution=spatial_size,
            window_size=win,
        )
        self.decoder = UpsampleDecoder(
            in_dim=embed_dim,
            out_channels=output_channels,
            target_shape=input_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        h, w = x.shape[-2:]
        if (h, w) != self.spatial_size:
            raise ValueError(f"Expected spatial {self.spatial_size}, got {(h, w)}")
        x = rearrange(x, "b c h w -> b h w c")
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = rearrange(x, "b h w c -> b c h w")
        return self.decoder(x)
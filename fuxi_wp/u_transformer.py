import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence

from swin import SwinTransformerBlock


def _build_swin_stage(
    dim: int,
    resolution: Tuple[int, int],
    depth: int,
    num_heads: int,
    window_size: int,
    drop_path: Sequence[float],
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
) -> nn.ModuleList:
    """Creates a list of Swin blocks that operate on B×C×H×W tensors."""
    blocks = nn.ModuleList()
    H, W = resolution
    for i in range(depth):
        blocks.append(
            SwinTransformerBlock(
                dim=dim,
                input_resolution=(H, W),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path[i],
            )
        )
    return blocks


class ResidualBlock(nn.Module):
    """3×3 conv → GN → SiLU repeated twice with a residual connection."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = x + residual
        x = F.silu(x)
        return x


class DownBlock(nn.Module):
    """3×3 stride-2 conv followed by the Residual block (FuXi Down Block)."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.residual = ResidualBlock(channels, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.residual(x)
        return x


class UpBlock(nn.Module):
    """Residual block + 2×2 transpose conv with skip concatenation (FuXi Up Block)."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.residual = ResidualBlock(channels, groups)
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.residual(x)
        x = self.up(x)
        return x


class SwinStage(nn.Module):
    """Wraps a list of Swin Transformer blocks to operate on feature maps."""

    def __init__(
        self,
        dim: int,
        resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        drop_path: Sequence[float],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.blocks = _build_swin_stage(
            dim=dim,
            resolution=resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (H, W) == self.resolution, f"Expected resolution {self.resolution}, got {(H, W)}"
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class UTransformer(nn.Module):
    """
    FuXi U-Transformer built from 48 Swin Transformer V2 blocks with down/up sampling.

    Layout:
        ┌─ SwinStage (24 blocks @ 48×96) ─┐
        │                                 ├─ skip_high
        └─ DownBlock ─ SwinStage (24 blocks @ 24×48) ─┐
                                                      └─ UpBlock → fuse with skip_high
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        input_resolution: Tuple[int, int] = (48, 96),
        down_resolution: Tuple[int, int] = (24, 48),
        depths: Tuple[int, int] = (24, 24),
        num_heads: Tuple[int, int] = (12, 12),
        window_sizes: Tuple[int, int] = (8, 6),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        groups: int = 32,
    ):
        super().__init__()
        assert sum(depths) == 48, "FuXi U-Transformer uses 48 Swin blocks in total."

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        ptr = 0

        self.encoder = SwinStage(
            dim=embed_dim,
            resolution=input_resolution,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_sizes[0],
            drop_path=dpr[ptr:ptr + depths[0]],
            mlp_ratio=mlp_ratio,
        )
        ptr += depths[0]

        self.down = DownBlock(embed_dim, groups=groups)

        self.bottleneck = SwinStage(
            dim=embed_dim,
            resolution=down_resolution,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_sizes[1],
            drop_path=dpr[ptr:ptr + depths[1]],
            mlp_ratio=mlp_ratio,
        )

        self.up = UpBlock(embed_dim, groups=groups)
        self.high_fuse = nn.Sequential(
            nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=1, bias=False),
            ResidualBlock(embed_dim, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) with H×W padded to match input_resolution (e.g., 48×96)

        Returns:
            (B, C, H, W) processed feature map.
        """
        skip_high = self.encoder(x)              # (B, C, H, W)
        low = self.down(skip_high)               # (B, C, H/2, W/2)
        skip_low = low                           # store skip before bottleneck
        low = self.bottleneck(low)               # (B, C, H/2, W/2)
        up = self.up(low, skip_low)              # upsample to H, W
        out = self.high_fuse(torch.cat([up, skip_high], dim=1))
        return out
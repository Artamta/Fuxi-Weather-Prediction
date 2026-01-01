import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Optional

from cube_embedding import CubeEmbedding3D
from swin import SwinTransformerV2


def pad_to_window(x: torch.Tensor, window_size: int, num_downsamples: int, swin_stages: int) -> torch.Tensor:
    """Pad features so they remain divisible by the Swin window size after all downsamples."""
    h, w = x.shape[-2:]
    factor = (2 ** (num_downsamples + swin_stages - 1)) * window_size
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


class DropPath(nn.Module):
    """Stochastic depth regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class ResidualBlock(nn.Module):
    """GroupNorm + SiLU residual block with optional DropPath."""
    def __init__(self, channels: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.drop_path(x)
        return self.act(x + residual)


class DownBlock(nn.Module):
    """Stride-2 downsampling followed by residual refinement."""
    def __init__(self, in_channels: int, out_channels: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.residual = ResidualBlock(out_channels, drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.residual(x)
        return x


class UpBlock(nn.Module):
    """Transposed-conv upsample, skip fusion, residual refinement."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        self.residual = ResidualBlock(out_channels, drop_path_rate)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.residual(x)
        return x


class FuXiModel(nn.Module):
    """
    FuXi-style encoder → Swin → decoder backbone.
    Scales cleanly with encoder_dims / swin_depths / heads.
    """
    def __init__(
        
        self,
        in_channels: int = 20,
        out_channels: int = 20,
        embed_dim: int = 512,
        swin_window_size: int = 8,
        input_height: int = 32,
        input_width: int = 64,
        encoder_dims: Iterable[int] = (384, 448, 512, 576),
        swin_depths: Iterable[int] = (8, 8, 12, 12),
        swin_heads: Iterable[int] = (8, 16, 32, 32),
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.swin_window_size = swin_window_size
        self.encoder_dims = list(encoder_dims)
        self.swin_depths = list(swin_depths)
        self.num_swin_stages = len(self.swin_depths)

        # Embed spatio-temporal cube.
        self.cube_embedding = CubeEmbedding3D(in_channels=in_channels, embed_dim=embed_dim)

        # Encoder blocks with progressive drop-path.
        num_down_blocks = len(self.encoder_dims)
        drop_rates = (
            torch.linspace(0.0, drop_path_rate, num_down_blocks, dtype=torch.float32).tolist()
            if num_down_blocks > 0
            else []
        )

        prev_channels = embed_dim
        self.down_blocks: List[DownBlock] = nn.ModuleList()
        self.skip_channels: List[int] = []

        for idx, dim in enumerate(self.encoder_dims):
            self.skip_channels.append(prev_channels)
            rate = drop_rates[idx] if idx < len(drop_rates) else 0.0
            self.down_blocks.append(DownBlock(prev_channels, dim, rate))
            prev_channels = dim

        # Determine Swin spatial size.
        dummy = torch.zeros(1, in_channels, 2, input_height, input_width)
        with torch.no_grad():
            feat = self.cube_embedding(dummy)
            feat = pad_to_window(
                feat,
                window_size=self.swin_window_size,
                num_downsamples=len(self.down_blocks),
                swin_stages=self.num_swin_stages,
            )
            for down_block in self.down_blocks:
                feat = down_block(feat)
            swin_h, swin_w = feat.shape[-2:]

        # Swin transformer in the bottleneck.
        swin_base_dim = prev_channels // 2
        self.swin_out_channels = swin_base_dim * (2 ** (self.num_swin_stages - 1))
        self.swin = SwinTransformerV2(
            img_size=(swin_h, swin_w),
            patch_size=1,
            in_chans=prev_channels,
            num_classes=0,
            embed_dim=swin_base_dim,
            depths=self.swin_depths,
            num_heads=list(swin_heads),
            window_size=self.swin_window_size,
            drop_path_rate=drop_path_rate,
        )
        self.swin_proj = nn.Conv2d(self.swin_out_channels, prev_channels, kernel_size=1)

        # Decoder mirroring the encoder.
        decoder_drop_rates = (
            torch.linspace(0.0, drop_path_rate, num_down_blocks, dtype=torch.float32).flip(0).tolist()
            if num_down_blocks > 0
            else []
        )

        self.up_blocks: List[UpBlock] = nn.ModuleList()
        for idx, skip_ch in enumerate(reversed(self.skip_channels)):
            rate = decoder_drop_rates[idx] if idx < len(decoder_drop_rates) else 0.0
            self.up_blocks.append(
                UpBlock(
                    in_channels=prev_channels,
                    skip_channels=skip_ch,
                    out_channels=skip_ch,
                    drop_path_rate=rate,
                )
            )
            prev_channels = skip_ch

        self.output_projection = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if target_shape is None:
            target_shape = x.shape[-2:]

        # Encoder
        x = self.cube_embedding(x)
        x = pad_to_window(
            x,
            window_size=self.swin_window_size,
            num_downsamples=len(self.down_blocks),
            swin_stages=self.num_swin_stages,
        )

        skips: List[torch.Tensor] = []
        for down_block in self.down_blocks:
            skips.append(x)
            x = down_block(x)

        # Swin core
        x = self.swin(x)
        if x.dim() == 3:
            b, n, c = x.shape
            h = w = int(n ** 0.5)
            x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        elif x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.swin_proj(x)

        # Decoder with skip connections
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip)

        # Output projection
        if x.shape[-2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=False)
        return self.output_projection(x)
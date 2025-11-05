import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from cube_embedding import CubeEmbedding3D
from swin import SwinTransformerV2

def pad_to_window(x, window_size, num_downsamples=1, swin_stages=3):
    h, w = x.shape[-2:]
    factor = (2 ** (num_downsamples + swin_stages - 1)) * window_size
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        # print(f"[pad_to_window] Padding: height +{pad_h}, width +{pad_w}")
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x

class ResidualBlock(nn.Module):
    def __init__(self, chaennels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.block(x) + x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res = ResidualBlock(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResidualBlock(out_channels)
    def forward(self, x, skip=None):
        x = self.tconv(x)
        x = self.res(x)
        return x

class FuXiModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 70,
        embed_dim: int = 1536,
        swin_window_size: int = 8,
        input_height=721,
        input_width=1440
    ):
        super().__init__()
        self.swin_window_size = swin_window_size
        self.cube_embedding = CubeEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.down = DownBlock(embed_dim, embed_dim)

        # Calculate Swin input shape (after padding and downsampling)
        dummy = torch.zeros(1, in_channels, 2, input_height, input_width)
        with torch.no_grad():
            dummy_emb = self.cube_embedding(dummy)
            dummy_emb = pad_to_window(dummy_emb, self.swin_window_size, num_downsamples=1)
            down_dummy = self.down(dummy_emb)
            _, _, swin_h, swin_w = down_dummy.shape

        self.swin_out_channels = embed_dim // 16 * 2 ** 2  # 384 for embed_dim=1536

        self.swin = SwinTransformerV2(
            img_size=(swin_h, swin_w),
            patch_size=1,
            in_chans=embed_dim,
            num_classes=0,
            global_pool=None,
            embed_dim=embed_dim // 16,
            depths=[12, 12, 24],  # Increase for full model
            num_heads=[2, 4, 8],
            window_size=self.swin_window_size,
            drop_path_rate=0.2
        )

        self.up = UpBlock(self.swin_out_channels + embed_dim, embed_dim)
        self.fc = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x, target_shape=None):
        if target_shape is None:
            target_shape = x.shape[-2:]
        # Only print if shape mismatch
        if x.shape[-2:] != target_shape:
            print(f"[FuXiModel] Input shape: {x.shape[-2:]}, Target shape: {target_shape}")
        x = self.cube_embedding(x)
        x = pad_to_window(x, self.swin_window_size, num_downsamples=1)
        skip = x
        x = self.down(x)
        x = self.swin(x)
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            B, N, C = x.shape
            H, W = skip.shape[-2:]
            x = x.transpose(1, 2).contiguous().view(B, C, H//2, W//2)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up(x, None)
        x_fc = self.fc(x)
        x_out = F.interpolate(x_fc, size=target_shape, mode='bilinear', align_corners=False)
        return x_out

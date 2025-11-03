import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange

# --- CubeEmbedding3D ---
class CubeEmbedding3D(nn.Module):
    def __init__(self, in_channels: int = 70, embed_dim: int = 192, patch_size: Tuple[int, int, int] = (2, 4, 4)):
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

# --- Helper ---
def pad_to_window(x, window_size, num_downsamples=1, swin_stages=2):
    h, w = x.shape[-2:]
    factor = (2 ** (num_downsamples + swin_stages - 1)) * window_size
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.2, num_blocks=2):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x) + x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res = ResidualBlock(out_channels, num_blocks=num_res_blocks)
    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResidualBlock(out_channels, num_blocks=num_res_blocks)
    def forward(self, x, skip=None):
        x = self.tconv(x)
        x = self.res(x)
        return x

class FuXiModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 70,
        embed_dim: int = 192,
        swin_window_size: int = 8,
        input_height=32,
        input_width=64,
        depths=[2, 4, 4],
        num_heads=None,
    ):
        super().__init__()
        self.swin_window_size = swin_window_size
        self.cube_embedding = CubeEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Deeper encoder: 2 DownBlocks
        self.down1 = DownBlock(embed_dim, embed_dim, num_res_blocks=2)
        self.down2 = DownBlock(embed_dim, embed_dim, num_res_blocks=2)

        # Calculate Swin input shape (after padding and downsampling)
        dummy = torch.zeros(1, in_channels, 2, input_height, input_width)
        with torch.no_grad():
            dummy_emb = self.cube_embedding(dummy)
            dummy_emb = pad_to_window(dummy_emb, self.swin_window_size, num_downsamples=2, swin_stages=3)
            down1_dummy = self.down1(dummy_emb)
            down2_dummy = self.down2(down1_dummy)
            _, _, swin_h, swin_w = down2_dummy.shape

        swin_embed_dim = embed_dim // 4  # 192 // 4 = 48

        from swin import SwinTransformerV2

        # Deeper Swin: 3 stages, more depth
        self.swin = SwinTransformerV2(
            img_size=(swin_h, swin_w),
            patch_size=1,
            in_chans=embed_dim,
            num_classes=0,
            embed_dim=swin_embed_dim,
            depths=depths,           # Deeper Swin
            num_heads=num_heads,
            window_size=self.swin_window_size,
            drop_path_rate=0.2
        )

        swin_out_channels = swin_embed_dim * 4  # Swin output is 4x embed_dim after last stage (for 3 stages)

        # Deeper decoder: 2 UpBlocks
        self.up1 = UpBlock(swin_out_channels + embed_dim, embed_dim, num_res_blocks=2)
        self.up2 = UpBlock(embed_dim, embed_dim, num_res_blocks=2)
        self.fc = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x, target_shape=None):
        if target_shape is None:
            target_shape = x.shape[-2:]
        x = self.cube_embedding(x)
        x = pad_to_window(x, self.swin_window_size, num_downsamples=2, swin_stages=3)
        skip = x
        x = self.down1(x)
        x = self.down2(x)
        x = self.swin(x)
        if x.dim() == 3:
            B, N, C = x.shape
            H_patch = W_patch = int(N ** 0.5)
            assert H_patch * W_patch == N, f"Cannot reshape: N={N} is not a square number"
            x = x.transpose(1, 2).contiguous().view(B, C, H_patch, W_patch)
        elif x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up1(x)
        x = self.up2(x)
        if x.shape[-2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        x_fc = self.fc(x)
        return x_fc
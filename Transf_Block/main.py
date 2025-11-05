import torch
import torch.nn as nn
from cube_embedding import CubeEmbedding3D
from swin_transf import BasicLayer

class FuXiUNet(nn.Module):
    def __init__(
        self,
        in_channels=70,
        out_channels=70,
        embed_dim=1536,
        depths=(12, 12, 12, 12),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        input_shape=(2, 70, 721, 1440),  # (T, C, H, W)
    ):
        super().__init__()
        # Cube embedding: (B, 2, 70, 721, 1440) -> (B, embed_dim, 180, 360)
        self.embedding = CubeEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4),
            padding=(0, 0, 0),
        )
        # Encoder
        self.enc1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(180, 360),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
        )
        self.down1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)  # 180x360 -> 90x180

        self.enc2 = BasicLayer(
            dim=embed_dim,
            input_resolution=(90, 180),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
        )
        self.down2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)  # 90x180 -> 45x90

        # Bottleneck
        self.bottleneck = BasicLayer(
            dim=embed_dim,
            input_resolution=(45, 90),
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)  # 45x90 -> 90x180
        self.dec2 = BasicLayer(
            dim=embed_dim,
            input_resolution=(90, 180),
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
        )

        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)  # 90x180 -> 180x360
        self.dec1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(180, 360),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
        )

        # Output projection
        self.out_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, 2, 70, 721, 1440)
        x = self.embedding(x)  # (B, embed_dim, 180, 360)
        # Encoder
        skip1 = self.enc1(x)  # (B, embed_dim, 180, 360)
        x = self.down1(skip1)  # (B, embed_dim, 90, 180)
        skip2 = self.enc2(x)   # (B, embed_dim, 90, 180)
        x = self.down2(skip2)  # (B, embed_dim, 45, 90)
        # Bottleneck
        x = self.bottleneck(x) # (B, embed_dim, 45, 90)
        # Decoder
        x = self.up2(x)        # (B, embed_dim, 90, 180)
        # Align skip2 spatial dims if needed
        if x.shape[-2:] != skip2.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = x + skip2          # skip connection
        x = self.dec2(x)       # (B, embed_dim, 90, 180)
        x = self.up1(x)        # (B, embed_dim, 180, 360)
        if x.shape[-2:] != skip1.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = x + skip1          # skip connection
        x = self.dec1(x)       # (B, embed_dim, 180, 360)
        x = self.out_conv(x)   # (B, out_channels, 180, 360)
        # Final upsampling to original shape if needed (e.g., 721x1440)
        x = nn.functional.interpolate(x, size=(721, 1440), mode="bilinear", align_corners=False)
        return x
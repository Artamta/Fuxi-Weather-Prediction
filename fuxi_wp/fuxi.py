import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cube_embedding import CubeEmbedding3D
from swin import SwinTransformerV2

def pad_to_window(x, window_size, num_downsamples=1, swin_stages=3):
    h, w = x.shape[-2:]
    factor = (2 ** (num_downsamples + swin_stages - 1)) * window_size
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        print(f"Padding: height +{pad_h}, width +{pad_w}")
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
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
    def forward(self, x):
        x = self.tconv(x)
        x = self.res(x)
        return x

class FuXiModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 70,
        embed_dim: int = 512,
        swin_window_size: int = 4
    ):
        super().__init__()
        self.swin_window_size = swin_window_size
        self.embed_dim = embed_dim

        self.cube_embedding = CubeEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.down = DownBlock(embed_dim, embed_dim)

        # Calculate Swin input shape (after padding and downsampling)
        dummy = torch.zeros(1, in_channels, 2, 721, 1440)
        with torch.no_grad():
            dummy_emb = self.cube_embedding(dummy)
            dummy_emb = pad_to_window(dummy_emb, self.swin_window_size, num_downsamples=1)
            down_dummy = self.down(dummy_emb)
            _, _, swin_h, swin_w = down_dummy.shape

        # Swin output channels: embed_dim // 16 * 2**2 = 384 for embed_dim=1536, 3 stages
        self.swin_out_channels = embed_dim // 16 * 2 ** 2  # 384

        self.swin = SwinTransformerV2(
            img_size=(swin_h, swin_w),
            patch_size=1,
            in_chans=embed_dim,
            num_classes=0,
            global_pool=None,
            embed_dim=embed_dim // 16,  # 1536//16=96, so last stage is 96*4=384
            depths=[2, 2, 6],
            num_heads=[3, 6, 12],
            window_size=self.swin_window_size
        )

        # Up Block: Swin output + skip connection
        self.up = UpBlock(self.swin_out_channels + embed_dim, embed_dim)

        # FC Layer: 1x1 Conv2d, as in the FuXi paper
        self.fc = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x, target_shape=(721, 1440)):
        print("Input x:", x.shape)
        x = self.cube_embedding(x)
        print("After embedding:", x.shape)
        x = pad_to_window(x, self.swin_window_size, num_downsamples=1)
        print("After pad_to_window:", x.shape)
        skip = x
        x = self.down(x)
        print("After DownBlock:", x.shape)
        x = self.swin(x)
        print("After SwinTransformer:", x.shape)
        # Reshape Swin output if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            B, N, C = x.shape
            H, W = skip.shape[-2:]
            x = x.transpose(1, 2).contiguous().view(B, C, H//2, W//2)
        print("After Swin reshape:", x.shape)
        # Upsample Swin output to match skip's spatial size
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            print(f"After upsampling Swin output to skip spatial size: {x.shape}")
        x = torch.cat([x, skip], dim=1)
        print("After concat Swin and skip:", x.shape)
        x = self.up(x)
        print("After UpBlock:", x.shape)
        x_fc = self.fc(x)
        print("After FC layer:", x_fc.shape)
        x_out = F.interpolate(x_fc, size=target_shape, mode='bilinear', align_corners=False)
        print("After upsampling:", x_out.shape)
        return x_out

    def predict_autoregressive(
        self,
        x: torch.Tensor,
        steps: int,
        target_shape: Tuple[int, int] = (721, 1440)
    ) -> torch.Tensor:
        device = x.device
        B, C, T, H, W = x.shape

        predictions = []
        current_input = x.clone()
        print(f"Autoregressive prediction: input {x.shape}")

        for step in range(steps):
            print(f"\n--- Step {step+1} ---")
            with torch.no_grad():
                next_pred = self.forward(current_input, target_shape)
                print(f"Prediction at step {step+1}: {next_pred.shape}")
                predictions.append(next_pred)

                last_timestep = current_input[:, :, -1:, :, :]  # (B, C, 1, H, W)
                next_pred_expanded = next_pred.unsqueeze(2)      # (B, C, 1, H, W)
                current_input = torch.cat([last_timestep, next_pred_expanded], dim=2)

                if current_input.shape[-2:] != (H, W):
                    current_input = F.interpolate(
                        current_input.flatten(0, 1),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).view(B, C, 2, H, W)
                print(f"Current input for next step: {current_input.shape}")

        predictions = torch.stack(predictions, dim=1)
        print(f"\nFinal stacked predictions: {predictions.shape}")
        return predictions

if __name__ == "__main__":
    print("=== Complete FuXi Model Test ===")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 1
    in_channels = 70
    timesteps = 2
    height = 721
    width = 1440

    print(f"\n[INFO] Creating synthetic weather data of shape: ({batch_size}, {in_channels}, {timesteps}, {height}, {width})")
    x = torch.randn(batch_size, in_channels, timesteps, height, width, device=device)
    print(f"[DEBUG] Input tensor shape: {x.shape}")

    print("\n[INFO] Instantiating FuXi model...")
    model = FuXiModel(
        in_channels=in_channels,
        out_channels=in_channels,
        embed_dim=1536,
        swin_window_size=4
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Model size: ~{total_params * 4 / 1e9:.2f} GB (fp32)")

    print("\n--- Single-step prediction ---")
    with torch.no_grad():
        output = model(x)
        print(f"[DEBUG] Output tensor shape: {output.shape}")

    print(f"[RESULT] Input->Output: {x.shape} -> {output.shape}")

    print("\n--- Multi-step prediction (5 steps) ---")
    with torch.no_grad():
        multi_output = model.predict_autoregressive(x, steps=5)
        print(f"[DEBUG] Multi-step output tensor shape: {multi_output.shape}")

    print(f"[RESULT] Multi-step output: {multi_output.shape}")
    print(f"[SUCCESS] 5-step forecast generated successfully!")

    print("\n✅ Complete FuXi model working correctly!")

    def estimate_memory_gb(tensor_shape, dtype_bytes=4):
        elements = 1
        for dim in tensor_shape:
            elements *= dim
        return elements * dtype_bytes / 1e9

    input_memory = estimate_memory_gb(x.shape)
    output_memory = estimate_memory_gb(output.shape)
    multi_memory = estimate_memory_gb(multi_output.shape)

    print(f"\n--- Memory Usage ---")
    print(f"[MEMORY] Input tensor: {input_memory:.2f} GB")
    print(f"[MEMORY] Single output: {output_memory:.2f} GB")
    print(f"[MEMORY] 5-step output: {multi_memory:.2f} GB")
    #EOFv1
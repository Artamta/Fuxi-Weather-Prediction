import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from cube_embedding import CubeEmbedding3D
from u_transformer import UTransformer


class FuXiModel(nn.Module):
    """
    Complete FuXi Weather Forecasting Model
    
    Pipeline:
        Input: (B, 70, 2, 721, 1440) - Two timesteps of 70 weather variables
        ↓
        CubeEmbedding: (B, 1536, 180, 360)
        ↓
        U-Transformer: (B, 1536, 180, 360)
        ↓
        Output Head: (B, 70, 180, 360)
        ↓
        Bilinear Upsample: (B, 70, 721, 1440)
    """
    
    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 70,
        embed_dim: int = 1536,
        depths: Tuple[int, int] = (24, 24),
        num_heads: Tuple[int, int] = (12, 12),
        window_sizes: Tuple[int, int] = (10, 10),  # Changed for 180×360
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        
        # Step 1: Cube Embedding (2×70×721×1440 → 1536×180×360)
        self.cube_embedding = CubeEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=(2, 4, 4)
        )
        
        # After cube embedding: 721/4 = 180.25 → 180, 1440/4 = 360
        # Step 2: U-Transformer (1536×180×360 → 1536×180×360)
        self.u_transformer = UTransformer(
            embed_dim=embed_dim,
            input_resolution=(180, 360),          # Changed from (48, 96)
            down_resolution=(90, 180),            # Changed from (24, 48)
            depths=depths,
            num_heads=num_heads,
            window_sizes=window_sizes,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )
        
        # Step 3: Output Head (FC layer)
        self.output_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int] = (721, 1440)) -> torch.Tensor:
        B = x.shape[0]
        
        # Step 1: Cube Embedding
        embedded = self.cube_embedding(x)  # (B, 1536, 180, 360)
        
        # Step 2: U-Transformer processing
        processed = self.u_transformer(embedded)  # (B, 1536, 180, 360)
        
        # Step 3: Output head
        output = self.output_head(processed)  # (B, 70, 180, 360)
        
        # Step 4: Upsample to original resolution
        output = F.interpolate(output, size=target_shape, 
                              mode='bilinear', align_corners=False)  # (B, 70, 721, 1440)
        
        return output
    
    def predict_autoregressive(self, x: torch.Tensor, steps: int = 20) -> torch.Tensor:
        predictions = []
        current = x
        
        for _ in range(steps):
            pred = self.forward(current)
            predictions.append(pred)
            
            current = torch.cat([
                current[:, :, 1:2, :, :],
                pred.unsqueeze(2)
            ], dim=2)
        
        return torch.stack(predictions, dim=0)
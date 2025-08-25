import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import our components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cube_embedding import CubeEmbedding3D
from models.u_transformer import UTransformer

class FuXiModel(nn.Module):
    """
    Complete FuXi weather forecasting model
    
    Pipeline:
    1. 3D Cube Embedding: (B, 70, 2, 721, 1440) -> (B, 1536, 180, 360)
    2. U-Transformer: Process embedded features with attention
    3. Output Head: Predict next timestep weather variables
    4. Upsampling: Restore original resolution
    """
    
    def __init__(
        self,
        in_channels: int = 70,           # ERA5 variables
        out_channels: int = 70,          # Same as input
        embed_dim: int = 1536,           # Transformer dimension
        num_heads: int = 8,              # Attention heads
        depth: int = 12,                 # Transformer depth
        mlp_ratio: float = 4.0,          # MLP expansion ratio
        dropout: float = 0.1             # Dropout rate
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        
        # Core components
        self.cube_embedding = CubeEmbedding3D(
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.u_transformer = UTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Output projection
        self.output_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )
        
        # For upsampling back to original resolution
        self.upsample_factor = 4  # 721//180 ≈ 4, 1440//360 = 4
        
    def forward(
        self, 
        x: torch.Tensor, 
        target_shape: Tuple[int, int] = (721, 1440)
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, 2, H, W) - Two timesteps of weather data
            target_shape: (H, W) - Target output shape
            
        Returns:
            (B, out_channels, H, W) - Predicted next timestep
        """
        B = x.shape[0]
        
        # 1. Cube embedding: reduce spatiotemporal dimensions
        embedded = self.cube_embedding(x)  # (B, embed_dim, H//4, W//4)
        
        # 2. U-Transformer: process with attention
        processed = self.u_transformer(embedded)  # (B, embed_dim, H//4, W//4)
        
        # 3. Output head: predict weather variables
        output = self.output_head(processed)  # (B, out_channels, H//4, W//4)
        
        # 4. Upsample to target resolution
        output = F.interpolate(
            output,
            size=target_shape,
            mode='bilinear',
            align_corners=False
        )  # (B, out_channels, H, W)
        
        return output
    
    def predict_autoregressive(
        self, 
        x: torch.Tensor, 
        steps: int,
        target_shape: Tuple[int, int] = (721, 1440)
    ) -> torch.Tensor:
        """
        Generate multi-step forecasts autoregressively
        
        Args:
            x: (B, in_channels, 2, H, W) - Initial two timesteps
            steps: Number of forecasting steps
            target_shape: Target spatial resolution
            
        Returns:
            (B, steps, out_channels, H, W) - Multi-step forecasts
        """
        device = x.device
        B, C, T, H, W = x.shape
        
        # Storage for predictions
        predictions = []
        
        # Current input (last 2 timesteps)
        current_input = x.clone()
        
        for step in range(steps):
            # Predict next timestep
            with torch.no_grad():
                next_pred = self.forward(current_input, target_shape)
                predictions.append(next_pred)
                
                # Update input: slide window forward
                # Take last timestep and current prediction as new input
                last_timestep = current_input[:, :, -1:, :, :]  # (B, C, 1, H, W)
                next_pred_expanded = next_pred.unsqueeze(2)      # (B, C, 1, H, W)
                
                # Create new input for next iteration
                current_input = torch.cat([last_timestep, next_pred_expanded], dim=2)
                
                # Downsample if needed to match original input resolution
                if current_input.shape[-2:] != (H, W):
                    current_input = F.interpolate(
                        current_input.flatten(0, 1),  # Combine B and T dims
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).view(B, C, 2, H, W)
        
        # Stack predictions: (B, steps, C, H, W)
        predictions = torch.stack(predictions, dim=1)
        
        return predictions

# Test the complete FuXi model
if __name__ == "__main__":
    print("=== Complete FuXi Model Test ===")
    
    # ERA5-like dimensions
    batch_size = 1
    in_channels = 70
    timesteps = 2
    height = 721
    width = 1440
    
    print(f"Input: {batch_size}×{in_channels}×{timesteps}×{height}×{width}")
    
    # Create synthetic weather data
    x = torch.randn(batch_size, in_channels, timesteps, height, width)
    
    # Create FuXi model
    model = FuXiModel(
        in_channels=in_channels,
        out_channels=in_channels,
        embed_dim=1536,
        num_heads=8,
        depth=12
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1e9:.2f} GB (fp32)")
    
    # Test single-step prediction
    print("\n--- Single-step prediction ---")
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Input->Output: {x.shape} -> {output.shape}")
    
    # Test multi-step prediction
    print("\n--- Multi-step prediction (5 steps) ---")
    with torch.no_grad():
        multi_output = model.predict_autoregressive(x, steps=5)
    
    print(f"Multi-step output: {multi_output.shape}")
    print(f"5-step forecast generated successfully!")
    
    print("\n✅ Complete FuXi model working correctly!")
    
    # Memory usage estimation
    def estimate_memory_gb(tensor_shape, dtype_bytes=4):
        elements = 1
        for dim in tensor_shape:
            elements *= dim
        return elements * dtype_bytes / 1e9
    
    input_memory = estimate_memory_gb(x.shape)
    output_memory = estimate_memory_gb(output.shape)
    multi_memory = estimate_memory_gb(multi_output.shape)
    
    print(f"\n--- Memory Usage ---")
    print(f"Input tensor: {input_memory:.2f} GB")
    print(f"Single output: {output_memory:.2f} GB") 
    print(f"5-step output: {multi_memory:.2f} GB")
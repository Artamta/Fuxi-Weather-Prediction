import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Tuple, Optional

class CubeEmbedding3D(nn.Module):
    """
    FuXi's 3D Cube Embedding - the key innovation
    
    Transforms: (B, C, T=2, H, W) -> (B, embed_dim, H//4, W//4)
    
    This is what makes FuXi work:
    1. Processes 2 timesteps jointly (captures temporal dynamics)
    2. Spatial downsampling (reduces computation 16x)
    3. Feature expansion (70 -> 1536 channels)
    """
    
    def __init__(
        self, 
        in_channels: int = 70,           # ERA5 variables
        embed_dim: int = 1536,           # Transformer dimension
        patch_size: Tuple[int,int,int] = (2, 4, 4)  # T, H, W
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # The magic 3D convolution
        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        
        # Normalization (crucial for training stability)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) - Two timesteps of weather data
            
        Returns:
            (B, embed_dim, H', W') where H'=H//4, W'=W//4
        """
        B, C, T, H, W = x.shape
        assert T == 2, f"Expected 2 timesteps, got {T}"
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        
        # 3D convolution: fuse time and downsample space
        x = self.projection(x)  # (B, embed_dim, 1, H//4, W//4)
        
        # Remove time dimension (now 1)
        x = x.squeeze(2)        # (B, embed_dim, H//4, W//4)
        
        # Apply layer normalization per spatial location
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        
        return x
    
    def get_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate output spatial dimensions"""
        H, W = input_size
        return H // self.patch_size[1], W // self.patch_size[2]

# Test the cube embedding
if __name__ == "__main__":
    print("=== FuXi 3D Cube Embedding Test ===")
    
    # Simulate ERA5 data dimensions
    batch_size = 2
    channels = 70      # 5 surface + 65 upper-air variables  
    timesteps = 2      # (t-1, t) -> predict (t+1)
    height = 721       # 90°S to 90°N at 0.25°
    width = 1440       # 0° to 360° at 0.25°
    
    print(f"Input: {batch_size}×{channels}×{timesteps}×{height}×{width}")
    print(f"Total elements: {batch_size*channels*timesteps*height*width:,}")
    
    # Create synthetic weather data
    x = torch.randn(batch_size, channels, timesteps, height, width)
    
    # Apply cube embedding
    cube_embed = CubeEmbedding3D(in_channels=channels, embed_dim=1536)
    
    print(f"\nParameters: {sum(p.numel() for p in cube_embed.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        embedded = cube_embed(x)
    
    out_h, out_w = embedded.shape[2], embedded.shape[3]
    print(f"\nOutput: {embedded.shape}")
    print(f"Spatial compression: {height}×{width} -> {out_h}×{out_w}")
    print(f"Compression ratio: {(height*width)/(out_h*out_w):.1f}x")
    print(f"Memory reduction: {x.numel()/embedded.numel():.1f}x")
    
    print("\n✅ Cube embedding working correctly!")
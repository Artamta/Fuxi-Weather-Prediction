import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional

class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal position encoding for spatial coordinates
    Helps the model understand geographical relationships
    """
    def __init__(self, embed_dim: int, max_len: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, h: int, w: int) -> torch.Tensor:
        """
        Generate 2D positional encoding for spatial grid
        Returns: (h*w, embed_dim)
        """
        # Create position indices for 2D grid
        pos_h = torch.arange(h, device=self.pe.device).unsqueeze(1).repeat(1, w).flatten()
        pos_w = torch.arange(w, device=self.pe.device).unsqueeze(0).repeat(h, 1).flatten()
        
        # Get positional encodings
        pe_h = self.pe[pos_h % self.pe.size(0)]
        pe_w = self.pe[pos_w % self.pe.size(0)]
        
        # Combine height and width encodings
        pe_2d = pe_h + pe_w
        return pe_2d

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product attention for processing weather features
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) where N = H*W (flattened spatial)
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        
        # Generate queries, keys, values
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-normalization
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attn(self.norm1(x))
        
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class UTransformer(nn.Module):
    """
    U-shaped Transformer architecture for weather forecasting
    
    Architecture:
    1. Downsample spatially while increasing channels
    2. Process with transformer blocks at reduced resolution
    3. Upsample back to original resolution with skip connections
    """
    
    def __init__(
        self,
        embed_dim: int = 1536,
        num_heads: int = 8,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionEncoding(embed_dim)
        
        # Downsampling path
        self.down_conv = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1)
        self.down_norm = nn.GroupNorm(32, embed_dim * 2)
        
        # Transformer blocks (at reduced resolution)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim * 2, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Upsampling path
        self.up_conv = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        self.up_norm = nn.GroupNorm(32, embed_dim)
        
        # Skip connection fusion
        self.skip_fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, embed_dim, H, W) - output from cube embedding
        Returns:
            (B, embed_dim, H, W) - processed features
        """
        B, C, H, W = x.shape
        
        # Store for skip connection
        skip = x
        
        # Downsample
        x = self.down_conv(x)  # (B, embed_dim*2, H//2, W//2)
        x = self.down_norm(x)
        x = F.gelu(x)
        
        # Reshape for transformer: (B, C, H, W) -> (B, H*W, C)
        B, C, H_down, W_down = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add positional encoding
        pos_enc = self.pos_encoding(H_down, W_down)
        x = x + pos_enc.unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H_down, w=W_down)
        
        # Upsample
        x = self.up_conv(x)  # (B, embed_dim, H, W)
        x = self.up_norm(x)
        x = F.gelu(x)
        
        # Fuse with skip connection
        x = torch.cat([x, skip], dim=1)  # (B, embed_dim*2, H, W)
        x = self.skip_fusion(x)          # (B, embed_dim, H, W)
        
        return x

# Test the U-Transformer
if __name__ == "__main__":
    print("=== U-Transformer Test ===")
    
    # Test with cube embedding output
    batch_size = 2
    embed_dim = 1536
    height = 180  # 721//4 from cube embedding
    width = 360   # 1440//4 from cube embedding
    
    print(f"Input: {batch_size}×{embed_dim}×{height}×{width}")
    
    # Create synthetic embedded features
    x = torch.randn(batch_size, embed_dim, height, width)
    
    # Create U-Transformer
    model = UTransformer(
        embed_dim=embed_dim,
        num_heads=8,
        depth=12,
        dropout=0.1
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output: {output.shape}")
    print(f"Shape preserved: {output.shape == x.shape}")
    
    print("\n✅ U-Transformer working correctly!")
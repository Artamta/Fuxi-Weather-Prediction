import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# Simple multi‑head self attention (no masking, no extras)
# --------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):          # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dh).permute(2,0,3,1,4)
        Q, K, V = qkv[0], qkv[1], qkv[2]      # (B, h, N, dh)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dh)   # (B,h,N,N)
        attn = scores.softmax(dim=-1)
        out = attn @ V                         # (B,h,N,dh)
        out = out.transpose(1,2).contiguous().view(B, N, D)
        out = self.proj(self.drop(out))
        return out

# --------------------------------------------------------
# Basic Transformer block: Attention + MLP + skips + norms
# --------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

# --------------------------------------------------------
# Patch embedding: split 28x28 into 4x4 patches (7x7 = 49)
# Each patch flattened and projected to embedding dim
# --------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, patch=4, in_ch=1, dim=128):
        super().__init__()
        self.patch = patch
        self.num_patches_side = 28 // patch          # 7
        self.num_patches = self.num_patches_side ** 2 # 49
        self.proj = nn.Linear(in_ch * patch * patch, dim)

    def forward(self, x):            # x: (B,1,28,28)
        B, C, H, W = x.shape
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)        # (B,C,7,7,p,p)
        x = x.contiguous().view(B, -1, C*p*p)        # (B,49,16)
        return self.proj(x)                          # (B,49,D)

# --------------------------------------------------------
# Simple U-Transformer for MNIST classification
# Two scales: 7x7 -> 3x3 -> up to 7x7
# --------------------------------------------------------
class USimpleTransformerMNIST(nn.Module):
    def __init__(self, dim=128, heads=4,
                 depth_level0=2, depth_level1=2,
                 patch=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch=patch, dim=dim)
        P0 = self.patch_embed.num_patches

        # Positional embeddings per level
        self.pos0 = nn.Parameter(torch.zeros(1, P0, dim))
        self.pos1 = nn.Parameter(torch.zeros(1, 16, dim))  # enough for 3x3 or 4x4

        # Level 0 (7x7 => 49 tokens)
        self.blocks0 = nn.ModuleList([
            TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth_level0)
        ])

        # Level 1 (after downsample to 3x3 => 9 tokens)
        self.blocks1 = nn.ModuleList([
            TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth_level1)
        ])

        # Refinement after up + skip
        self.refine = TransformerBlock(dim, heads, dropout=dropout)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos0, std=0.02)
        nn.init.trunc_normal_(self.pos1, std=0.02)

    def forward(self, x):                 # x: (B,1,28,28)
        B = x.size(0)

        # ---- Level 0 ----
        tok0 = self.patch_embed(x) + self.pos0      # (B,49,D)
        for blk in self.blocks0:
            tok0 = blk(tok0)

        # Reshape to grid 7x7 for pooling
        H0 = W0 = self.patch_embed.num_patches_side
        g0 = tok0.view(B, H0, W0, -1).permute(0,3,1,2)    # (B,D,7,7)

        # ---- Downsample to Level 1 (avg pool -> 3x3) ----
        g1 = F.avg_pool2d(g0, kernel_size=2, stride=2)    # (B,D,3,3)
        tok1 = g1.permute(0,2,3,1).reshape(B, -1, g1.size(1))  # (B,9,D)
        tok1 = tok1 + self.pos1[:, :tok1.size(1)]
        for blk in self.blocks1:
            tok1 = blk(tok1)

        # ---- Upsample back ----
        g1b = tok1.view(B, 3, 3, -1).permute(0,3,1,2)     # (B,D,3,3)
        g1_up = F.interpolate(g1b, size=(H0, W0), mode='nearest')  # (B,D,7,7)

        # ---- Fuse skip (add) ----
        fused = g0 + g1_up                                # (B,D,7,7)

        # ---- Refinement ----
        fused_tok = fused.permute(0,2,3,1).reshape(B, -1, fused.size(1))  # (B,49,D)
        fused_tok = self.refine(fused_tok)

        # ---- Classification ----
        fused_tok = self.norm(fused_tok)
        cls_vec = fused_tok.mean(dim=1)                   # mean over tokens
        return self.head(cls_vec)                         # (B,10)

# --------------------------------------------------------
# Quick test (few steps) so you can show it runs
# --------------------------------------------------------
def quick_train(steps=200):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = USimpleTransformerMNIST().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    ds = datasets.MNIST("data", train=True, download=True,
                        transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    model.train()
    for i,(img,lab) in enumerate(loader):
        img,lab = img.to(device), lab.to(device)
        opt.zero_grad()
        out = model(img)
        loss = loss_fn(out, lab)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            pred = out.argmax(1)
            acc = (pred == lab).float().mean().item()
            print(f"step {i} loss {loss.item():.4f} acc {acc*100:.1f}%")
        if i >= steps:
            break

if __name__ == "__main__":
    # Run with (mac temporary fix if needed):
    # KMP_DUPLICATE_LIB_OK=TRUE python u_transformer_simple.py
    quick_train()
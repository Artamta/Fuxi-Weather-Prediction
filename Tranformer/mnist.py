import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# ---- Minimal Attention + Encoder Block ----
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.0):
        super().__init__()
        assert embed_size % heads == 0
        self.heads = heads
        self.head_dim = embed_size // heads
        self.qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.proj = nn.Linear(embed_size, embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, E)
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2,0,3,1,4)
        Q, K, V = qkv[0], qkv[1], qkv[2]                    # (B,h,N,hd)
        attn = (Q @ K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = attn @ V                                      # (B,h,N,hd)
        out = out.transpose(1,2).contiguous().view(B,N,E)
        return self.proj(self.drop(out))

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

# ---- Patch Embedding ----
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, embed_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size*patch_size, embed_size)

    def forward(self, x):
        # x: (B,1,28,28)
        B,_,H,W = x.shape
        p = self.patch_size
        x = x.unfold(2,p,p).unfold(3,p,p)          # (B,1,H/p,W/p,p,p)
        x = x.contiguous().view(B, -1, p*p)        # (B, num_patches, p*p)
        return self.proj(x)                        # (B, P, E)

# ---- MNIST Transformer (Encoder-only ViT style) ----
class MNISTTransformer(nn.Module):
    def __init__(self, embed_size=256, num_layers=4, heads=8,
                 ff_mult=4, dropout=0.1, patch=4, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbed(patch, embed_size)
        num_patches = (28//patch)*(28//patch)
        self.cls = nn.Parameter(torch.zeros(1,1,embed_size))
        self.pos = nn.Parameter(torch.zeros(1, num_patches+1, embed_size))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, forward_expansion=ff_mult, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        tok = self.patch_embed(x)                  # (B,P,E)
        B = tok.size(0)
        cls_tok = self.cls.expand(B, -1, -1)       # (B,1,E)
        x = torch.cat([cls_tok, tok], dim=1)       # (B,P+1,E)
        x = x + self.pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x[:,0])                   # (B,num_classes)

# ---- Quick Train Loop ----
def quick_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MNISTTransformer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    ds = datasets.MNIST(root="data", train=True, download=True,
                        transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    model.train()
    for i,(img,label) in enumerate(loader):
        img,label = img.to(device), label.to(device)
        opt.zero_grad()
        out = model(img)
        loss = loss_fn(out,label)
        loss.backward()
        opt.step()
        if i % 100 == 0:
            print(f"Step {i} Loss {loss.item():.4f}")
        if i == 300:
            break

if __name__ == "__main__":
    quick_train()
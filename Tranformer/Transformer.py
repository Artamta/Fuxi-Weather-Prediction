import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Self Attention
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.0):
        super().__init__()
        assert embed_size % heads == 0, "embed_size must divide heads"
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.q_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Shapes: query/key/value: (N, seq_len, embed)
        N, q_len, _ = query.shape
        k_len = key.shape[1]
        v_len = value.shape[1]

        Q = self.q_proj(query).view(N, q_len, self.heads, self.head_dim).transpose(1, 2)  # (N,h,q,hd)
        K = self.k_proj(key).view(N, k_len, self.heads, self.head_dim).transpose(1, 2)    # (N,h,k,hd)
        V = self.v_proj(value).view(N, v_len, self.heads, self.head_dim).transpose(1, 2)  # (N,h,v,hd)

        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)          # (N,h,q,k)

        if mask is not None:
            # mask: broadcastable to (N, 1, q, k)
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(energy, dim=-1)                                              # (N,h,q,k)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                                       # (N,h,q,hd)
        out = out.transpose(1, 2).contiguous().view(N, q_len, self.embed_size)            # (N,q,embed)
        return self.out_proj(out)                                                         # (N,q,embed)
    
# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.attn(x, x, x, src_mask)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x

# Encoder
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads,
                 forward_expansion, dropout, max_length, device):
        super().__init__()
        self.device = device
        self.token_embed = nn.Embedding(src_vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: (N, src_len)
        N, src_len = src.shape
        positions = torch.arange(0, src_len, device=self.device).unsqueeze(0).expand(N, src_len)
        x = self.drop(self.token_embed(src) + self.pos_embed(positions))  # (N,src_len,embed)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x  # (N,src_len,embed)

# Decoder Block (masked self-attn + cross-attn)
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.cross_attn = SelfAttention(embed_size, heads, dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, trg_mask=None, src_mask=None):
        # Masked self-attention
        _x = self.self_attn(x, x, x, trg_mask)
        x = self.norm1(x + self.drop(_x))
        # Cross attention (queries = decoder states, keys/values = encoder output)
        _x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop(_x))
        # Feed forward
        _x = self.ff(x)
        x = self.norm3(x + self.drop(_x))
        return x

# Decoder

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads,
                 forward_expansion, dropout, max_length, device):
        super().__init__()
        self.device = device
        self.token_embed = nn.Embedding(trg_vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, trg, enc_out, src_mask=None, trg_mask=None):
        N, trg_len = trg.shape
        positions = torch.arange(0, trg_len, device=self.device).unsqueeze(0).expand(N, trg_len)
        x = self.drop(self.token_embed(trg) + self.pos_embed(positions))  # (N,trg_len,embed)
        for layer in self.layers:
            x = layer(x, enc_out, trg_mask, src_mask)
        return self.fc_out(x)  # (N,trg_len,vocab)


# Full Transformer

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers=4,
                 heads=8,
                 forward_expansion=4,
                 dropout=0.1,
                 max_length=100,
                 device="cpu"):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads,
                               forward_expansion, dropout, max_length, device)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads,
                               forward_expansion, dropout, max_length, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src: (N, src_len)
        mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N,1,1,src_len)
        return mask  # broadcast to (N,1,q,src_len)

    def make_trg_mask(self, trg):
        # trg: (N, trg_len)
        N, trg_len = trg.shape
        causal = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()  # (trg_len,trg_len)
        pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)                  # (N,1,1,trg_len)
        # Combine: broadcast causal over batch; later masking zeros -> -inf
        return pad_mask & causal.unsqueeze(0)  # (N,1,trg_len,trg_len)

    def forward(self, src, trg):
        # trg given without last token for teacher forcing
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return out  # (N,trg_len,trg_vocab)


# Test

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = torch.tensor([[1,5,6,4,0,0],[1,8,7,3,4,2]], device=device)  # (N, src_len)
    trg = torch.tensor([[1,7,4,3,0],[1,5,6,2,0]], device=device)      # (N, trg_len)
    model = Transformer(src_vocab_size=10,
                        trg_vocab_size=10,
                        src_pad_idx=0,
                        trg_pad_idx=0,
                        device=device).to(device)
    # Teacher forcing input excludes last token when predicting next
    out = model(src, trg[:, :-1])
    print("Output shape:", out.shape)  # (N, trg_len-1, vocab)
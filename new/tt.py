import torch.nn as nn
from einops import rearrange
from swin import BasicLayer

class SwinBlockStack(nn.Module):
    def __init__(self, dim, depth, heads, input_resolution, window_size=8, use_checkpoint=False):
        super().__init__()
        h, w = input_resolution
        win = min(window_size, h, w)

        self.layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=heads,
            window_size=win,
            downsample=None,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        h, w = x.shape[1:3]
        tokens = rearrange(x, "b h w c -> b (h w) c")
        encoded = self.layer(tokens)
        return rearrange(encoded, "b (h w) c -> b h w c", h=h, w=w)
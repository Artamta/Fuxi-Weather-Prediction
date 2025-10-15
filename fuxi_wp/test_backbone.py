import torch
from fuxi import FuXiTinyBackbone

model = FuXiTinyBackbone(
    input_shape=(16, 8),
    window_size=2,
    depths=(2, 2),
    num_heads=(4, 4),
)

x = torch.randn(1, 70, 2, 16, 8)
print(model(x).shape)
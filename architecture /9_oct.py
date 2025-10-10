from textwrap import indent

sections = {
    "Current Stack": [
        "Input: (B, 70, T=2, H=16, W=8)",
        "CubeEmbedding3D → (B, 384, 4, 2)",
        "SwinBlockStack x2 (depth=2, heads=4, window=2)",
        "UpsampleDecoder → (B, 70, 16, 8)",
    ],
    "Data Flow": [
        "Conv3D kernel (T,4,4), stride (1,4,4) reduces spatial grid.",
        "Swin layers operate on tokens with alternating shifted windows.",
        "Transpose-conv stack upsamples back to forecast grid.",
    ],
    "Future Work": [
        "Add missing humidity channels to reach full 70-input.",
        "Wire dataloaders + training loop with loss logging.",
        "Scale depths/heads and window sizes for full-resolution FuXi.",
    ],
}


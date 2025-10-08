import graphviz

dot = graphviz.Digraph("FuXiCascade", format="png")
dot.attr(rankdir="LR", fontsize="11", splines="spline", nodesep="1", ranksep="1")

# === Legend / Title ===
dot.attr(label=(
    "FuXi Cascade Weather Forecasting System\n"
    "Resolution: 0.25° (721×1440) • Temporal Step: 6 h • Variables: 70\n"
    "FuXi-Tiny backbone per module: embed_dim=384, depths=(6,6), num_heads=(6,6)"
), labelloc="t", fontsize="14")

# === Modules ===
with dot.subgraph(name="cluster_pipeline") as pipe:
    pipe.attr(label="Cascade Pipeline (Autoregressive, 3 × FuXi-Tiny)", fontsize="12", color="#424242")

    pipe.node("ICs", "Initial Conditions\nERA5 6-hourly • Two time slices (t-1,t)\nShape: (1,70,2,721,1440)", shape="box",
              style="filled", fillcolor="#E3F2FD")

    # FuXi-Short
    pipe.node("Short", (
        "FuXi-Short (0–5 days / 20 steps)\n"
        "• CubeEmbedding3D 70→384\n"
        "• U-Transformer (Encoder/Bottleneck 6 SwinV2 blocks each)\n"
        "• DownBlock + UpBlock + residual fusion\n"
        "• OutputHead 384→192→70 + bilinear upscale\n"
        "Outputs cached @ t=5d for FuXi-Medium"
    ), shape="box", style="filled", fillcolor="#C8E6C9")

    # FuXi-Medium
    pipe.node("Medium", (
        "FuXi-Medium (5–10 days / steps 21–40)\n"
        "Input: FuXi-Short forecast @ t=5d + last obs history\n"
        "Fine-tuned with autoregressive steps 2→12\n"
        "Same FuXi-Tiny backbone"
    ), shape="box", style="filled", fillcolor="#FFE082")

    # FuXi-Long
    pipe.node("Long", (
        "FuXi-Long (10–15 days / steps 41–60)\n"
        "Input: FuXi-Medium forecast @ t=10d\n"
        "Targets longest horizon, uses cached medium outputs\n"
        "Same FuXi-Tiny backbone"
    ), shape="box", style="filled", fillcolor="#FFCCBC")

    pipe.node("Forecast", (
        "Final Forecast\n15-day, 6-hourly steps (60 frames)\n"
        "Shape per step: (1,70,721,1440)"
    ), shape="box", style="filled", fillcolor="#EF9A9A")

    # Connections
    pipe.edge("ICs", "Short", label="(t-1,t) ⇒ 20 forward steps", fontsize="9")
    pipe.edge("Short", "Medium", label="handoff @ step 20 (5d)", fontsize="9")
    pipe.edge("Medium", "Long", label="handoff @ step 40 (10d)", fontsize="9")
    pipe.edge("Long", "Forecast", label="collect steps 41–60", fontsize="9")

# === Detailed FuXi-Tiny Backbone ===
with dot.subgraph(name="cluster_backbone") as backbone:
    backbone.attr(label="FuXi-Tiny Backbone (used in all three cascade modules)", fontsize="12", color="#616161")

    backbone.node("InputNode", "Input Tensor\n(1,70,2,721,1440)", shape="box", style="filled", fillcolor="#E3F2FD")

    backbone.node("Cube", (
        "CubeEmbedding3D\nConv3D (k=2×4×4, stride=1×4×4)\nChannels: 70→384 (≈3.4M params)\n"
        "GroupNorm(32) + SiLU\nReshapes ⇒ (1,384,180,360)"
    ), shape="box", style="filled", fillcolor="#AED581")

    backbone.node("Encoder", (
        "Encoder Stage\n6 × SwinTransformerBlock V2\nWindow 10×10, shift alternate\n"
        "qkv_bias, scaled cosine attention, DropPath 0→0.1\n"
        "Tokens: (180×360)=64800 • Heads: 6"
    ), shape="box", style="filled", fillcolor="#FFF59D")

    backbone.node("Down", (
        "DownBlock\nConv2d 3×3 stride 2 (384 channels)\nGroupNorm(32) + SiLU\nResidualBlock ×1\n"
        "Spatial: 180×360 → 90×180"
    ), shape="box", style="filled", fillcolor="#FFE082")

    backbone.node("Bottleneck", (
        "Bottleneck Stage\n6 × SwinTransformerBlock V2\nWindow 10×10 on 90×180 grid\n"
        "Maintains 384 channels\nDropPath 0.1→0.2"
    ), shape="box", style="filled", fillcolor="#FFECB3")

    backbone.node("Up", (
        "UpBlock\nConcat with skip_low (after DownBlock)\n1×1 Conv fuse 768→384\n"
        "ResidualBlock ×1 • ConvTranspose2d stride 2\nRestores 180×360"
    ), shape="box", style="filled", fillcolor="#FFE082")

    backbone.node("HighFuse", (
        "High-level Fusion\nConcat skip_high (Encoder output)\n1×1 Conv 768→384\nResidualBlock ×1"
    ), shape="box", style="filled", fillcolor="#FFE082")

    backbone.node("Head", (
        "Output Head\nConv2d 384→192 (1×1) + GELU\nConv2d 192→70 (1×1)\n"
        "Bilinear Interpolate ⇒ (721,1440)"
    ), shape="box", style="filled", fillcolor="#EF9A9A")

    backbone.node("BackboneOut", "Forecast Frame\n(1,70,721,1440)", shape="box", style="filled", fillcolor="#FFCDD2")

    # Backbone edges
    backbone.edges([
        ("InputNode", "Cube"),
        ("Cube", "Encoder"),
        ("Encoder", "Down"),
        ("Down", "Bottleneck"),
        ("Bottleneck", "Up"),
        ("Up", "HighFuse"),
        ("HighFuse", "Head"),
        ("Head", "BackboneOut"),
    ])

    backbone.edge("Encoder", "HighFuse", label="skip_high (384,180,360)", style="dashed", color="#616161", fontsize="9")
    backbone.edge("Down", "Up", label="skip_low (384,90,180)", style="dashed", color="#616161", fontsize="9")

# === Ensemble Head ===
with dot.subgraph(name="cluster_ensemble") as ens:
    ens.attr(label="Ensemble Generation (optional)", fontsize="12", color="#757575")
    ens.node("Perturb", (
        "Initial Perturbations\n49 × Perlin noise (flow-independent)\nMC Dropout (p=0.2)"
    ), shape="box", style="filled", fillcolor="#B39DDB")
    ens.node("Members", "50-member FuXi Cascade\nParallel forward passes\nCRPS / SSR evaluation", shape="box",
             style="filled", fillcolor="#D1C4E9")
    ens.edge("Perturb", "Members")

# === Notes ===
dot.node("Notes", (
    "Training: Pre-train 40k iters (batch 1 × 8×A100) • AdamW lr 2.5e-4 → 1e-7 fine-tune\n"
    "Fine-tune schedule: autoregressive steps ramp (2→12)\n"
    "DropPath schedule per block • bfloat16 + FSDP + checkpointing\n"
    "Loss: latitude-weighted L1 • Evaluation: RMSE, ACC, CRPS, SSR"
), shape="note", fontsize="9")

# === Layout connections ===
dot.edge("ICs", "InputNode", style="dotted", arrowhead="none")
dot.edge("BackboneOut", "Short", style="dotted", arrowhead="none")
dot.edge("Long", "Members", style="dotted", arrowhead="none")

dot.render("fuxi_cascade_detailed", cleanup=True)
print("Created fuxi_cascade_detailed.png")
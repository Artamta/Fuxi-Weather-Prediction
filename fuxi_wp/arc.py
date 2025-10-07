import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from fuxi import create_fuxi_tiny, FuXiModel
from u_transformer import UTransformer
from cube_embedding import CubeEmbedding3D

def analyze_actual_implementation():
    """Analyze what's actually in your code"""
    print("="*60)
    print("ANALYZING YOUR ACTUAL IMPLEMENTATION")
    print("="*60)
    
    # Check cube embedding
    cube_emb = CubeEmbedding3D(in_channels=70, embed_dim=1536, patch_size=(2, 4, 4))
    test_input = torch.randn(1, 70, 2, 721, 1440)
    cube_output = cube_emb(test_input)
    print(f"\n1. CUBE EMBEDDING:")
    print(f"   Input:  {tuple(test_input.shape)}")
    print(f"   Output: {tuple(cube_output.shape)}")
    print(f"   Params: {sum(p.numel() for p in cube_emb.parameters()):,}")
    
    # Check U-Transformer
    u_trans = UTransformer(
        embed_dim=1536,
        input_resolution=(180, 360),
        down_resolution=(90, 180),
        depths=(24, 24),
        num_heads=(12, 12),
        window_sizes=(10, 10),
    )
    print(f"\n2. U-TRANSFORMER:")
    print(f"   Input resolution:  {u_trans.encoder.resolution}")
    print(f"   Down resolution:   {u_trans.bottleneck.resolution}")
    print(f"   Encoder blocks:    {len(u_trans.encoder.blocks)}")
    print(f"   Bottleneck blocks: {len(u_trans.bottleneck.blocks)}")
    print(f"   Total Swin blocks: {len(u_trans.encoder.blocks) + len(u_trans.bottleneck.blocks)}")
    print(f"   Params: {sum(p.numel() for p in u_trans.parameters()):,}")
    
    # Check full model
    model_full = FuXiModel(embed_dim=1536, depths=(24, 24), num_heads=(12, 12))
    print(f"\n3. FULL FUXI MODEL:")
    print(f"   Total params: {sum(p.numel() for p in model_full.parameters()):,}")
    
    # Check tiny model
    model_tiny = create_fuxi_tiny()
    print(f"\n4. FUXI-TINY MODEL:")
    print(f"   Embed dim: {model_tiny.u_transformer.encoder.dim}")
    print(f"   Encoder blocks: {len(model_tiny.u_transformer.encoder.blocks)}")
    print(f"   Bottleneck blocks: {len(model_tiny.u_transformer.bottleneck.blocks)}")
    print(f"   Total params: {sum(p.numel() for p in model_tiny.parameters()):,}")
    
    print("\n" + "="*60)
    return {
        'cube_input': test_input.shape,
        'cube_output': cube_output.shape,
        'encoder_blocks': len(u_trans.encoder.blocks),
        'bottleneck_blocks': len(u_trans.bottleneck.blocks),
        'full_params': sum(p.numel() for p in model_full.parameters()),
        'tiny_params': sum(p.numel() for p in model_tiny.parameters()),
    }


def draw_your_actual_architecture():
    """Draw YOUR actual implementation"""
    
    info = analyze_actual_implementation()
    
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(10, 13.5, 'YOUR FuXi Implementation', 
            fontsize=24, weight='bold', ha='center', color='red')
    ax.text(10, 13, '(As Actually Coded)', fontsize=18, ha='center')
    
    # Color scheme
    color_input = '#E3F2FD'
    color_embed = '#C5E1A5'
    color_transformer = '#FFE082'
    color_output = '#FFCCBC'
    
    def draw_box(x, y, w, h, text, color, fontsize=11):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h/2 + (len(lines)/2 - i - 0.5)*0.2, 
                   line, fontsize=fontsize, ha='center', va='center', weight='bold')
    
    def draw_arrow(x1, y1, x2, y2, label=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 1. INPUT
    y_pos = 11
    draw_box(0.5, y_pos, 3, 1, 'Input\n(1, 70, 2, 721, 1440)', color_input, 10)
    
    # 2. CUBE EMBEDDING (YOUR CODE)
    draw_arrow(2, y_pos, 2, y_pos - 1)
    y_pos = 9
    draw_box(0.5, y_pos, 3, 1.5, 'CubeEmbedding3D\nConv3D(2,4,4)', color_embed, 11)
    ax.text(2, y_pos + 0.75, 'patch_size=(2,4,4)\nGroupNorm + SiLU', fontsize=9, ha='center')
    ax.text(2, y_pos - 0.3, f'→ {info["cube_output"]}', fontsize=8, ha='center', style='italic')
    
    # 3. U-TRANSFORMER (YOUR ACTUAL STRUCTURE)
    draw_arrow(3.5, y_pos + 0.75, 5, y_pos + 0.75)
    
    # Encoder
    y_pos = 9
    draw_box(5, y_pos + 1.5, 3.5, 1.2, 
            f'Encoder\n{info["encoder_blocks"]} Swin Blocks', color_transformer, 11)
    ax.text(6.75, y_pos + 1.1, 'SwinStage\nResolution: 180×360\nWindow: 10×10', 
           fontsize=8, ha='center')
    
    # Down Block (YOUR CODE)
    draw_arrow(6.75, y_pos + 1.5, 6.75, y_pos + 1)
    draw_box(5, y_pos - 0.2, 3.5, 1, 'DownBlock\n3×3 Conv stride=2', color_transformer, 10)
    ax.text(6.75, y_pos - 0.5, 'ResidualBlock\n→ 90×180', fontsize=8, ha='center', style='italic')
    
    # Bottleneck (YOUR CODE)
    draw_arrow(6.75, y_pos - 0.2, 6.75, y_pos - 1)
    draw_box(5, y_pos - 2.2, 3.5, 1.2, 
            f'Bottleneck\n{info["bottleneck_blocks"]} Swin Blocks', color_transformer, 11)
    ax.text(6.75, y_pos - 2.6, 'SwinStage\nResolution: 90×180\nWindow: 10×10', 
           fontsize=8, ha='center')
    
    # Up Block (YOUR CODE)
    draw_arrow(6.75, y_pos - 2.2, 6.75, y_pos - 3)
    draw_box(5, y_pos - 4, 3.5, 1, 'UpBlock\nConvTranspose2d', color_transformer, 10)
    ax.text(6.75, y_pos - 4.3, 'Residual + Upsample\n→ 180×360', fontsize=8, ha='center', style='italic')
    
    # Skip Connections (YOUR CODE)
    draw_arrow(8.5, y_pos + 2.1, 10, y_pos + 2.1, 'skip_high')
    draw_arrow(10, y_pos + 2.1, 10, y_pos - 3)
    draw_arrow(10, y_pos - 3, 8.5, y_pos - 3)
    
    draw_arrow(8.5, y_pos - 0.7, 11, y_pos - 0.7, 'skip_low')
    draw_arrow(11, y_pos - 0.7, 11, y_pos - 2.4)
    draw_arrow(11, y_pos - 2.4, 8.5, y_pos - 2.4)
    
    ax.text(10.3, y_pos - 0.5, 'Skip\nConnections', fontsize=9, ha='left', 
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Fusion (YOUR CODE)
    draw_arrow(6.75, y_pos - 4, 6.75, y_pos - 4.8)
    draw_box(5, y_pos - 6, 3.5, 1, 'high_fuse\nConcat + Conv + Residual', color_transformer, 10)
    
    # 4. OUTPUT HEAD (YOUR CODE)
    draw_arrow(8.5, y_pos - 5.5, 11.5, y_pos - 5.5)
    draw_box(11.5, y_pos - 6, 3.5, 1.5, 'Output Head', color_output, 11)
    ax.text(13.25, y_pos - 5.25, 'Conv(1536→768)\nGELU\nConv(768→70)', fontsize=9, ha='center')
    
    # 5. UPSAMPLE (YOUR CODE)
    draw_arrow(15, y_pos - 5.25, 16.5, y_pos - 5.25)
    draw_box(16.5, y_pos - 6, 3, 1.5, 'Bilinear\nInterpolate', color_output, 11)
    ax.text(18, y_pos - 5.25, '180×360\n↓\n721×1440', fontsize=10, ha='center')
    
    # OUTPUT
    draw_arrow(19.5, y_pos - 5.25, 19.5, 11)
    draw_arrow(19.5, 11, 3.5, 11)
    draw_box(16.5, 10.5, 3, 1, 'Output\n(1, 70, 721, 1440)', color_output, 10)
    
    # YOUR IMPLEMENTATION DETAILS
    details_y = 2
    ax.text(1, details_y + 1, 'YOUR IMPLEMENTATION SPECS:', fontsize=13, weight='bold', color='red')
    
    specs = [
        f'Total Swin Blocks: {info["encoder_blocks"] + info["bottleneck_blocks"]}',
        f'Full Model Params: {info["full_params"]:,} ({info["full_params"]/1e9:.2f}B)',
        f'Tiny Model Params: {info["tiny_params"]:,} ({info["tiny_params"]/1e6:.1f}M)',
        '',
        'Components:',
        '• CubeEmbedding3D with Conv3D',
        '• UTransformer (encoder + bottleneck)',
        '• ResidualBlock (Conv+GN+SiLU)',
        '• DownBlock (stride-2 conv)',
        '• UpBlock (transpose conv)',
        '• Skip connections (high + low)',
    ]
    
    for i, spec in enumerate(specs):
        weight = 'bold' if spec.startswith('•') or 'Params' in spec else 'normal'
        ax.text(1, details_y + 0.6 - i * 0.25, spec, fontsize=9, weight=weight)
    
    # Model Variants YOU CREATED
    variants_x = 11
    ax.text(variants_x, details_y + 1, 'YOUR MODEL VARIANTS:', fontsize=13, weight='bold', color='red')
    
    variants = [
        ('create_fuxi_full()', 'embed_dim=1536', 'depths=(24,24)', f'{info["full_params"]/1e9:.1f}B params'),
        ('create_fuxi_medium()', 'embed_dim=1024', 'depths=(18,18)', '~400M params'),
        ('create_fuxi_small()', 'embed_dim=768', 'depths=(12,12)', '~200M params'),
        ('create_fuxi_tiny()', 'embed_dim=384', 'depths=(6,6)', f'{info["tiny_params"]/1e6:.0f}M params'),
    ]
    
    for i, (func, dim, depth, params) in enumerate(variants):
        y = details_y + 0.6 - i * 0.25
        ax.text(variants_x, y, f'{func}: {dim}, {depth} → {params}', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('YOUR_fuxi_architecture.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: YOUR_fuxi_architecture.png (shows YOUR actual code)")
    plt.close()


if __name__ == "__main__":
    draw_your_actual_architecture()
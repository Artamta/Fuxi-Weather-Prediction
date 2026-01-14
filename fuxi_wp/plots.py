import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

def draw_block(ax, xy, width, height, label, color, fontsize=14, lw=2, zorder=2):
    box = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.08", ec="black", fc=color, lw=lw, zorder=zorder)
    ax.add_patch(box)
    ax.text(xy[0]+width/2, xy[1]+height/2, label, ha="center", va="center", fontsize=fontsize, color="black", weight="bold", zorder=zorder+1)

def draw_arrow(ax, start, end, color="gray", lw=2, style="->", zorder=1):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle=style, lw=lw, color=color), zorder=zorder)

def draw_skip(ax, start, end, color="#ff7f0e", lw=2, curve=0.2):
    # Draws a curved skip connection
    con = ConnectionPatch(start, end, "data", "data", arrowstyle="->", color=color, lw=lw, zorder=1, shrinkA=8, shrinkB=8, connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(con)

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Colors
input_c = "#f7f7f7"
enc_c = "#a6cee3"
swin_c = "#fb9a99"
bottleneck_c = "#cab2d6"
dec_c = "#b2df8a"
output_c = "#f7f7f7"
skip_c = "#ff7f0e"

# Block positions and labels
blocks = [
    (1, 7, 2, 1.2, "Input\n(Weather Maps)", input_c),
    (3.5, 7, 2, 1.2, "Encoder 1\nConv+Pool", enc_c),
    (6, 7, 2, 1.2, "Encoder 2\nConv+Pool", enc_c),
    (8.5, 7, 2.2, 1.4, "Swin Transformer\nBlocks", swin_c),
    (11.2, 7, 2, 1.2, "Bottleneck\nConv", bottleneck_c),
    (13.5, 7, 2, 1.2, "Decoder 1\nUpConv", dec_c),
    (13.5, 4.5, 2, 1.2, "Decoder 2\nUpConv", dec_c),
    (15.5, 7, 1.2, 1.2, "Output\n(Prediction)", output_c),
]

for x, y, w, h, label, color in blocks:
    draw_block(ax, (x, y), w, h, label, color, fontsize=15)

# Main horizontal arrows
draw_arrow(ax, (3, 7.6), (3.5, 7.6))
draw_arrow(ax, (5.5, 7.6), (6, 7.6))
draw_arrow(ax, (8.2, 7.7), (8.5, 7.7))
draw_arrow(ax, (10.7, 7.6), (11.2, 7.6))
draw_arrow(ax, (13.2, 7.6), (13.5, 7.6))
draw_arrow(ax, (15.5, 7.6), (16.1, 7.6))

# Down arrow to lower decoder
draw_arrow(ax, (14.5, 7), (14.5, 5.7), color="#555", lw=2.5)
draw_arrow(ax, (15.5, 5.7), (16.1, 5.7), color="#555", lw=2.5)

# Skip connections (curved, from encoder to decoder)
draw_skip(ax, (5, 8.2), (14.5, 5.9), color=skip_c, lw=2, curve=-0.3)
draw_skip(ax, (3, 8.2), (13.5, 8.2), color=skip_c, lw=2, curve=0.2)

# Stage labels
ax.text(2.5, 9.2, "Encoder", fontsize=16, color=enc_c, weight="bold", ha="center")
ax.text(9.6, 9.2, "Swin Transformer", fontsize=16, color=swin_c, weight="bold", ha="center")
ax.text(12.2, 9.2, "Bottleneck", fontsize=16, color=bottleneck_c, weight="bold", ha="center")
ax.text(14.5, 9.2, "Decoder", fontsize=16, color=dec_c, weight="bold", ha="center")

# Legend for skip connections
ax.plot([], [], color=skip_c, lw=3, label="Skip Connection")
ax.legend(loc="lower left", fontsize=13, frameon=False)

# Title
ax.text(8, 10, "Swin Transformer + U-Net Architecture for Weather Forecasting", fontsize=20, ha="center", va="center", weight="bold")

plt.tight_layout()
plt.savefig("swin_unet_architecture_pub_v2.png", dpi=500)
plt.show()
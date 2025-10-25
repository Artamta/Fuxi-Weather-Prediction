# Save as swin_unet_architecture.py and run with: python swin_unet_architecture.py
from graphviz import Digraph

dot = Digraph(comment='Swin-UNet Architecture', format='pdf')  # Use PDF for high quality
dot.graph_attr['dpi'] = '300'
dot.graph_attr['size'] = '16,10'
dot.attr(rankdir='LR')


# Encoder
dot.node('Input', 'Input\n(B, C, H, W)')
dot.node('Patch', 'Patch Embedding\nConv2d/CubeEmbedding')
dot.node('Enc1', 'Encoder Stage 1\nSwinBlockStack')
dot.node('Down1', 'Downsample\nConv2d/Pool')
dot.node('Enc2', 'Encoder Stage 2\nSwinBlockStack')
dot.node('Down2', 'Downsample\nConv2d/Pool')
dot.node('Enc3', 'Encoder Stage 3\nSwinBlockStack')
dot.node('Down3', 'Downsample\nConv2d/Pool')

# Bottleneck
dot.node('Bottleneck', 'Bottleneck\nSwinBlockStack')

# Decoder
dot.node('Up3', 'Upsample\nConvTranspose2d')
dot.node('Dec3', 'Decoder Stage 3\nConv2d + Skip')
dot.node('Up2', 'Upsample\nConvTranspose2d')
dot.node('Dec2', 'Decoder Stage 2\nConv2d + Skip')
dot.node('Up1', 'Upsample\nConvTranspose2d')
dot.node('Dec1', 'Decoder Stage 1\nConv2d + Skip')
dot.node('Final', 'Final Conv\nConv2d\nOutput (B, C, H, W)')

# Connections
dot.edge('Input', 'Patch')
dot.edge('Patch', 'Enc1')
dot.edge('Enc1', 'Down1')
dot.edge('Down1', 'Enc2')
dot.edge('Enc2', 'Down2')
dot.edge('Down2', 'Enc3')
dot.edge('Enc3', 'Down3')
dot.edge('Down3', 'Bottleneck')

dot.edge('Bottleneck', 'Up3')
dot.edge('Up3', 'Dec3')
dot.edge('Dec3', 'Up2')
dot.edge('Up2', 'Dec2')
dot.edge('Dec2', 'Up1')
dot.edge('Up1', 'Dec1')
dot.edge('Dec1', 'Final')

# Skip connections
dot.edge('Enc3', 'Dec3', label='Skip 3', style='dashed')
dot.edge('Enc2', 'Dec2', label='Skip 2', style='dashed')
dot.edge('Enc1', 'Dec1', label='Skip 1', style='dashed')

dot.render('swin_unet_architecture', view=True)
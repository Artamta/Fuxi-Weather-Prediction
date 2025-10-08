import graphviz

dot = graphviz.Digraph('FuXiTiny', format='png')

dot.attr(rankdir='LR', fontsize='10', splines='spline', nodesep='1', ranksep='0.75')

dot.node('Input', 'Input\n(1, 70, 2, 721, 1440)', shape='box', style='filled', fillcolor='#E3F2FD')
dot.node('Cube', 'CubeEmbedding3D\nConv3D(2×4×4)\n→ (1, 384, 180, 360)', shape='box', style='filled', fillcolor='#C5E1A5')
dot.node('Enc', 'Encoder Stage\n6 × Swin Block\nWindow 10×10\n(180×360)', shape='box', style='filled', fillcolor='#FFE082')
dot.node('Down', 'DownBlock\n3×3 Conv stride 2\n→ (90×180)', shape='box', style='filled', fillcolor='#FFE082')
dot.node('Bot', 'Bottleneck Stage\n6 × Swin Block\nWindow 10×10\n(90×180)', shape='box', style='filled', fillcolor='#FFE082')
dot.node('Up', 'UpBlock\nTranspose Conv 2×\n+ Residual', shape='box', style='filled', fillcolor='#FFE082')
dot.node('Fuse', 'High-level Fusion\nConcat skip_high\nConv1×1 + Residual', shape='box', style='filled', fillcolor='#FFE082')
dot.node('Head', 'Output Head\nConv 384→192→70\n(180×360)', shape='box', style='filled', fillcolor='#FFCCBC')
dot.node('Upsample', 'Bilinear Upsample\n→ (721×1440)', shape='box', style='filled', fillcolor='#FFCCBC')
dot.node('Output', 'Forecast\n(1, 70, 721, 1440)', shape='box', style='filled', fillcolor='#FFCCBC')

dot.edges([
    ('Input', 'Cube'),
    ('Cube', 'Enc'),
    ('Enc', 'Down'),
    ('Down', 'Bot'),
    ('Bot', 'Up'),
    ('Up', 'Fuse'),
    ('Fuse', 'Head'),
    ('Head', 'Upsample'),
    ('Upsample', 'Output')
])

dot.edge('Enc', 'Fuse', label='skip_high', style='dashed', color='#616161')
dot.edge('Down', 'Up', label='skip_low', style='dashed', color='#616161')

dot.attr(label='FuXi-Tiny Architecture\n(embed_dim=384, depths=(6, 6), ~96M parameters)', labelloc='t', fontsize='12')

dot.render('fuxi_tiny_architecture', cleanup=True)
print('Created fuxi_tiny_architecture.png')
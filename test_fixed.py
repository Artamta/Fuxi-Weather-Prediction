import torch
import sys
sys.path.append('./models')

from fuxi_model import FuXiModel

print("=== Testing Fixed Implementation ===")

# Test with small dimensions
x = torch.randn(1, 70, 2, 90, 180)

# Test with same embed_dim in all components
model = FuXiModel(
    in_channels=70,
    embed_dim=384,
    depth=4,
    dropout=0.1
)

try:
    output = model(x, target_shape=(90, 180))
    print(f"✅ Fixed! {x.shape} -> {output.shape}")
    
    # Test autoregressive
    multi_output = model.predict_autoregressive(x, steps=2, target_shape=(90, 180))
    print(f"✅ Autoregressive works: {multi_output.shape}")
    
    print("\n🎉 All fixed! Ready for your prof meeting!")
    
except Exception as e:
    print(f"❌ Still an error: {e}")
    
    # Let's debug the specific issue
    print("\nDebugging...")
    from cube_embedding import CubeEmbedding3D
    from u_transformer import UTransformer
    
    cube = CubeEmbedding3D(in_channels=70, embed_dim=384)
    embedded = cube(x)
    print(f"Cube output: {embedded.shape}")
    
    try:
        transformer = UTransformer(embed_dim=384, depth=2)
        transformed = transformer(embedded)
        print(f"Transformer output: {transformed.shape}")
    except Exception as e2:
        print(f"Transformer error: {e2}")

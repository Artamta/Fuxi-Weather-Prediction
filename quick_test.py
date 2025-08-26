import torch
import sys
import os

# Add models to path
sys.path.append('./models')

print("=== Quick FuXi Test ===")

try:
    from cube_embedding import CubeEmbedding3D
    print("✅ CubeEmbedding3D imported successfully")
except Exception as e:
    print(f"❌ CubeEmbedding3D import failed: {e}")

try:
    from u_transformer import UTransformer
    print("✅ UTransformer imported successfully")
except Exception as e:
    print(f"❌ UTransformer import failed: {e}")

try:
    from fuxi_model import FuXiModel
    print("✅ FuXiModel imported successfully")
except Exception as e:
    print(f"❌ FuXiModel import failed: {e}")

# Quick functionality test
try:
    # Small test data
    x = torch.randn(1, 70, 2, 90, 180)
    
    # Test cube embedding
    cube = CubeEmbedding3D(in_channels=70, embed_dim=384)
    embedded = cube(x)
    print(f"✅ Cube embedding works: {x.shape} -> {embedded.shape}")
    
    # Test full model (small version)
    model = FuXiModel(
        in_channels=70,
        embed_dim=384,
        depth=4,
        dropout=0.1
    )
    
    output = model(x, target_shape=(90, 180))
    print(f"✅ Full model works: {x.shape} -> {output.shape}")
    
    # Test autoregressive
    multi_output = model.predict_autoregressive(x, steps=2, target_shape=(90, 180))
    print(f"✅ Autoregressive works: {multi_output.shape}")
    
    print("\n🎉 All tests passed! Your FuXi implementation is working!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

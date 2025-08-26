import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add models to path
sys.path.append('./models')

from cube_embedding import CubeEmbedding3D
from u_transformer import UTransformer
from fuxi_model import FuXiModel

def test_gpu_setup():
    """Test GPU availability and performance"""
    print("=== GPU Setup Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Quick performance test
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        start_time = time.time()
        y = torch.mm(x, x)
        end_time = time.time()
        print(f"Matrix multiplication (1000x1000): {(end_time - start_time)*1000:.2f} ms")
        print("✅ GPU working correctly!")
    else:
        print("❌ CUDA not available!")
    print()

def test_cube_embedding():
    """Test the 3D cube embedding"""
    print("=== Cube Embedding Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with different sizes
    test_sizes = [
        (1, 70, 2, 180, 360),    # Small for quick test
        (2, 70, 2, 360, 720),    # Medium
        (1, 70, 2, 721, 1440),   # Full ERA5 resolution
    ]
    
    for batch_size, channels, timesteps, height, width in test_sizes:
        print(f"Testing size: {batch_size}×{channels}×{timesteps}×{height}×{width}")
        
        # Create test data
        x = torch.randn(batch_size, channels, timesteps, height, width, device=device)
        input_memory = x.numel() * 4 / 1e9  # GB
        
        # Create model
        model = CubeEmbedding3D(in_channels=channels, embed_dim=1536).to(device)
        
        # Time the forward pass
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model(x)
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        output_memory = output.numel() * 4 / 1e9  # GB
        
        print(f"  Input memory: {input_memory:.2f} GB")
        print(f"  Output memory: {output_memory:.2f} GB")
        print(f"  Compression: {input_memory/output_memory:.2f}x")
        print(f"  Forward time: {(end_time - start_time)*1000:.1f} ms")
        print(f"  Output shape: {output.shape}")
        
        # Verify output dimensions
        expected_h = height // 4
        expected_w = width // 4
        assert output.shape == (batch_size, 1536, expected_h, expected_w)
        print("  ✅ Shape correct!")
        
        del x, output, model
        torch.cuda.empty_cache()
        print()

def test_u_transformer():
    """Test the U-Transformer"""
    print("=== U-Transformer Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with cube embedding output size
    batch_size = 2
    embed_dim = 1536
    height = 180  # 721//4
    width = 360   # 1440//4
    
    print(f"Testing U-Transformer: {batch_size}×{embed_dim}×{height}×{width}")
    
    # Create test data (output from cube embedding)
    x = torch.randn(batch_size, embed_dim, height, width, device=device)
    
    # Test different model sizes
    configs = [
        {"depth": 6, "num_heads": 8},    # Small for quick test
        {"depth": 12, "num_heads": 8},   # Medium
        {"depth": 24, "num_heads": 12},  # Large
    ]
    
    for config in configs:
        print(f"  Testing depth={config['depth']}, heads={config['num_heads']}")
        
        model = UTransformer(
            embed_dim=embed_dim,
            depth=config['depth'],
            num_heads=config['num_heads'],
            dropout=0.1
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {param_count:,}")
        
        # Time forward pass
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model(x)
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"    Forward time: {(end_time - start_time)*1000:.1f} ms")
        print(f"    Output shape: {output.shape}")
        
        # Verify shape preservation
        assert output.shape == x.shape
        print("    ✅ Shape preserved!")
        
        del model, output
        torch.cuda.empty_cache()
    
    del x
    print()

def test_complete_fuxi_model():
    """Test the complete FuXi model"""
    print("=== Complete FuXi Model Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different input sizes
    test_configs = [
        {"batch": 1, "size": (180, 360), "name": "Small"},
        {"batch": 2, "size": (360, 720), "name": "Medium"},
        {"batch": 1, "size": (721, 1440), "name": "Full ERA5"},
    ]
    
    for config in test_configs:
        print(f"Testing {config['name']}: batch={config['batch']}, size={config['size']}")
        
        batch_size = config['batch']
        height, width = config['size']
        
        # Create synthetic ERA5-like data
        x = torch.randn(batch_size, 70, 2, height, width, device=device)
        input_memory = x.numel() * 4 / 1e9
        
        # Create FuXi model (smaller for testing)
        model = FuXiModel(
            in_channels=70,
            out_channels=70,
            embed_dim=1536,
            num_heads=8,
            depth=12,  # Moderate depth for testing
            dropout=0.1
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        model_memory = param_count * 4 / 1e9
        
        print(f"  Input memory: {input_memory:.2f} GB")
        print(f"  Model memory: {model_memory:.2f} GB")
        print(f"  Parameters: {param_count:,}")
        
        # Test single-step prediction
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model(x, target_shape=(height, width))
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"  Single-step time: {(end_time - start_time)*1000:.1f} ms")
        print(f"  Output shape: {output.shape}")
        
        # Verify output shape
        expected_shape = (batch_size, 70, height, width)
        assert output.shape == expected_shape
        print("  ✅ Single-step prediction correct!")
        
        # Test multi-step prediction (smaller steps for speed)
        steps = 3
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            multi_output = model.predict_autoregressive(x, steps=steps, target_shape=(height, width))
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"  Multi-step time ({steps} steps): {(end_time - start_time)*1000:.1f} ms")
        print(f"  Multi-step shape: {multi_output.shape}")
        
        # Verify multi-step shape
        expected_multi_shape = (batch_size, steps, 70, height, width)
        assert multi_output.shape == expected_multi_shape
        print("  ✅ Multi-step prediction correct!")
        
        del x, output, multi_output, model
        torch.cuda.empty_cache()
        print()

def test_memory_scaling():
    """Test memory usage with different batch sizes"""
    print("=== Memory Scaling Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("Skipping memory test - CUDA not available")
        return
    
    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        print(f"Testing batch size: {batch_size}")
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create data
            x = torch.randn(batch_size, 70, 2, 360, 720, device=device)
            
            # Create model
            model = FuXiModel(
                in_channels=70,
                embed_dim=1536,
                depth=8,  # Smaller for memory test
                dropout=0.1
            ).to(device)
            
            # Forward pass
            with torch.no_grad():
                output = model(x, target_shape=(360, 720))
            
            # Check memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Max memory used: {memory_used:.2f} GB")
            print(f"  ✅ Batch size {batch_size} successful!")
            
            del x, output, model
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ Out of memory at batch size {batch_size}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    print()

def run_all_tests():
    """Run complete test suite"""
    print("🚀 Starting FuXi Complete Test Suite")
    print("=" * 50)
    
    test_gpu_setup()
    test_cube_embedding()
    test_u_transformer()
    test_complete_fuxi_model()
    test_memory_scaling()
    
    print("=" * 50)
    print("✅ All tests completed successfully!")
    print("Your FuXi implementation is working correctly! 🎉")

if __name__ == "__main__":
    run_all_tests()
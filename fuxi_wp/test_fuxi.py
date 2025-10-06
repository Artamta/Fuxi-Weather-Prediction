import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from fuxi import FuXiModel

def get_device():
    """Automatically select best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def test_fuxi_model():
    print("=== Testing FuXi Model ===\n")
    
    # Get device
    device = get_device()
    print()
    
    # Model setup
    model = FuXiModel(
        in_channels=70,
        out_channels=70,
        embed_dim=1536,
        depths=(24, 24),
        num_heads=(12, 12),
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Test single-step prediction
    batch_size = 1
    x = torch.randn(batch_size, 70, 2, 721, 1440).to(device)
    
    print("Single-step test:")
    with torch.no_grad():
        output = model(x)
    print(f"Input: {x.shape} on {x.device}")
    print(f"Output: {output.shape} on {output.device}")
    assert output.shape == (batch_size, 70, 721, 1440)
    print("✓ Single-step prediction works!\n")
    
    # Test multi-step prediction
    print("Multi-step test (5 days = 20 steps):")
    with torch.no_grad():
        multi_output = model.predict_autoregressive(x, steps=20)
    print(f"Output: {multi_output.shape} on {multi_output.device}")
    assert multi_output.shape == (20, batch_size, 70, 721, 1440)
    print("✓ Multi-step prediction works!\n")
    
    # Memory usage
    if device.type == 'cuda':
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    elif device.type == 'mps':
        print("MPS backend doesn't provide memory stats")

if __name__ == "__main__":
    test_fuxi_model()
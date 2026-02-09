#!/usr/bin/env python3
"""
Model Interface Compatibility Checker

Verifies that your FuXi model implementation is compatible with the training script.
Provides specific guidance on how to adapt your model if needed.

Usage: python check_model_interface.py
"""

import inspect
import sys
from typing import get_type_hints

def check_model_import():
    """Try to import the model and check its interface."""
    print("=" * 70)
    print("Checking Model Import and Interface")
    print("=" * 70 + "\n")
    
    # Try to import
    try:
        from model import FuXi
        print("✓ Successfully imported FuXi from model.py\n")
    except ImportError as e:
        print(f"✗ Failed to import FuXi from model.py")
        print(f"  Error: {e}\n")
        print("Troubleshooting:")
        print("  1. Check if model.py exists in the current directory")
        print("  2. Check if it contains a class named 'FuXi'")
        print("  3. If your model has a different name, update fuxi_train.py line 48:")
        print("     from model import YourModelName as FuXi")
        return None
    
    return FuXi


def check_init_signature(FuXi):
    """Check __init__ method signature."""
    print("\n" + "=" * 70)
    print("Checking __init__ Signature")
    print("=" * 70 + "\n")
    
    expected_params = {
        'num_variables': int,
        'embed_dim': int,
        'num_heads': int,
        'window_size': int,
        'depth_pre': int,
        'depth_mid': int,
        'depth_post': int,
        'mlp_ratio': float,
        'drop_path_rate': float,
        'input_height': int,
        'input_width': int,
        'use_checkpoint': bool,
    }
    
    # Get actual signature
    try:
        sig = inspect.signature(FuXi.__init__)
        actual_params = {
            name: param for name, param in sig.parameters.items()
            if name != 'self'
        }
        
        print("Expected parameters:")
        for name, typ in expected_params.items():
            print(f"  {name}: {typ.__name__}")
        
        print("\nActual parameters:")
        for name, param in actual_params.items():
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            print(f"  {name}{default}")
        
        # Check for missing parameters
        missing = set(expected_params.keys()) - set(actual_params.keys())
        extra = set(actual_params.keys()) - set(expected_params.keys())
        
        if not missing and not extra:
            print("\n✓ Signature matches perfectly!")
            return True
        
        if missing:
            print(f"\n⚠ Missing parameters: {', '.join(missing)}")
            print("\nYou need to add these parameters to your __init__ method.")
            print("If your model doesn't use some of these, you can accept them")
            print("as arguments but ignore them internally.")
        
        if extra:
            print(f"\n⚠ Extra parameters: {', '.join(extra)}")
            print("\nThese parameters are not expected by the training script.")
            print("Make sure they have default values or update the training script.")
        
        return False
        
    except Exception as e:
        print(f"✗ Failed to inspect __init__ signature: {e}")
        return False


def check_forward_signature(FuXi):
    """Check forward method signature."""
    print("\n" + "=" * 70)
    print("Checking forward() Signature")
    print("=" * 70 + "\n")
    
    try:
        sig = inspect.signature(FuXi.forward)
        params = list(sig.parameters.keys())
        
        print("Expected signature:")
        print("  def forward(self, x: torch.Tensor) -> torch.Tensor")
        print("    Args: x shape (B, C, T, H, W) - history frames")
        print("    Returns: shape (B, C, H, W) - predicted next frame")
        
        print("\nActual signature:")
        print(f"  def forward({', '.join(params)})")
        
        # Check parameter count (should be 2: self, x)
        if len(params) != 2:
            print(f"\n⚠ WARNING: Expected 2 parameters (self, x), got {len(params)}")
            print("  Additional parameters should have default values")
            return False
        
        if params[1] != 'x':
            print(f"\n⚠ WARNING: First argument should be 'x', got '{params[1]}'")
            print("  The training script expects forward(self, x)")
            return False
        
        print("\n✓ Signature looks good!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to inspect forward signature: {e}")
        return False


def test_forward_pass(FuXi):
    """Test actual forward pass."""
    print("\n" + "=" * 70)
    print("Testing Forward Pass")
    print("=" * 70 + "\n")
    
    try:
        import torch
        
        # Create model instance
        print("Creating model instance...")
        model = FuXi(
            num_variables=20,
            embed_dim=64,
            num_heads=4,
            window_size=5,
            depth_pre=2,
            depth_mid=6,
            depth_post=2,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            input_height=121,
            input_width=240,
            use_checkpoint=False,
        )
        print("✓ Model created successfully")
        
        # Test input
        B, C, T, H, W = 2, 20, 2, 121, 240
        x = torch.randn(B, C, T, H, W)
        
        print(f"\nInput shape: {tuple(x.shape)} (B, C, T, H, W)")
        
        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            y = model(x)
        
        print(f"Output shape: {tuple(y.shape)}")
        
        # Check output shape
        expected_shape = (B, C, H, W)
        if y.shape == expected_shape:
            print(f"\n✓ Output shape correct: {expected_shape}")
            return True
        else:
            print(f"\n✗ Output shape incorrect!")
            print(f"  Expected: {expected_shape}")
            print(f"  Got: {y.shape}")
            print("\nYour model should:")
            print("  - Take input: (B, C, T, H, W)")
            print("  - Return output: (B, C, H, W)")
            print("  where T is the number of history steps (typically 2)")
            return False
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nCommon issues:")
        print("  - Missing required parameters in __init__")
        print("  - Incompatible input/output shapes")
        print("  - Missing dependencies (torch, etc.)")
        return False


def provide_wrapper_template(FuXi):
    """Provide a wrapper template if model interface differs."""
    print("\n" + "=" * 70)
    print("Model Wrapper Template (if needed)")
    print("=" * 70 + "\n")
    
    print("If your model interface differs, use this wrapper:")
    print("\n" + "-" * 70)
    print("""
# Add this to fuxi_train.py after the imports (around line 50)

class FuXiWrapper(nn.Module):
    \"\"\"Wrapper to adapt your model to the expected interface.\"\"\"
    
    def __init__(self, num_variables, embed_dim, num_heads, window_size,
                 depth_pre, depth_mid, depth_post, mlp_ratio, drop_path_rate,
                 input_height, input_width, use_checkpoint):
        super().__init__()
        
        # Import your actual model
        from model import YourActualModelClass
        
        # Map parameters to your model's expected names
        self.model = YourActualModelClass(
            channels=num_variables,        # or in_channels, num_vars, etc.
            dim=embed_dim,                 # or hidden_dim, d_model, etc.
            heads=num_heads,               # or n_heads, num_attention_heads, etc.
            # ... map other parameters as needed
        )
    
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        
        # If your model expects different input format, reshape here
        # For example, if it expects (B, T, C, H, W):
        # x = x.permute(0, 2, 1, 3, 4)
        
        # Call your model
        out = self.model(x)
        
        # If your model returns different output format, reshape here
        # Expected output: (B, C, H, W)
        
        return out

# Then replace the FuXi import with:
# FuXi = FuXiWrapper
""")
    print("-" * 70)


def main():
    print("\n" + "=" * 70)
    print("FuXi Model Interface Compatibility Checker")
    print("=" * 70 + "\n")
    
    # Step 1: Import model
    FuXi = check_model_import()
    if FuXi is None:
        print("\n" + "=" * 70)
        print("FAILED: Cannot import model")
        print("=" * 70)
        return 1
    
    # Step 2: Check signatures
    init_ok = check_init_signature(FuXi)
    forward_ok = check_forward_signature(FuXi)
    
    # Step 3: Test forward pass
    forward_test_ok = test_forward_pass(FuXi)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70 + "\n")
    
    results = {
        'Import': FuXi is not None,
        '__init__ signature': init_ok,
        'forward() signature': forward_ok,
        'Forward pass test': forward_test_ok,
    }
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 70)
        print("✓ Your model is fully compatible!")
        print("=" * 70)
        print("\nYou can use it directly in the training script.")
        print("No wrapper needed.")
        return 0
    else:
        print("\n" + "=" * 70)
        print("⚠ Your model needs adaptation")
        print("=" * 70)
        provide_wrapper_template(FuXi)
        print("\nOptions:")
        print("  1. Modify your model.py to match the expected interface")
        print("  2. Use the wrapper template above")
        print("  3. Update fuxi_train.py to match your model's interface")
        return 1


if __name__ == "__main__":
    sys.exit(main())

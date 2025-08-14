#!/usr/bin/env python3
"""
Environment Test Script for Weather Forecasting Project
Run this script to verify all required packages are properly installed.
"""

import sys
from importlib import import_module

# Required packages for weather forecasting
REQUIRED_PACKAGES = {
    'numpy': 'Scientific computing',
    'pandas': 'Data manipulation',
    'matplotlib': 'Plotting',
    'seaborn': 'Statistical visualization',
    'plotly': 'Interactive plots',
    'scipy': 'Scientific computing',
    'sklearn': 'Machine learning (scikit-learn)',
    'torch': 'Deep learning (PyTorch)',
    'torchvision': 'Computer vision (PyTorch)',
    'xarray': 'N-dimensional arrays',
    'netCDF4': 'NetCDF file format',
    'h5py': 'HDF5 file format',
    'tqdm': 'Progress bars',
    'yaml': 'YAML configuration files (PyYAML)',
    'jupyter': 'Jupyter notebooks',
    'IPython': 'Interactive Python (IPython)',
}

def test_package_import(package_name, description):
    """Test if a package can be imported."""
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:12} {version:10} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:12} {'MISSING':10} - {description}")
        print(f"   Error: {e}")
        return False

def test_torch_functionality():
    """Test basic PyTorch functionality."""
    try:
        import torch
        # Test tensor creation
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)
        
        # Test if MPS (Metal Performance Shaders) is available on macOS
        if torch.backends.mps.is_available():
            print("✅ PyTorch MPS (GPU) support available")
        else:
            print("ℹ️  PyTorch MPS (GPU) support not available (CPU only)")
            
        return True
    except Exception as e:
        print(f"❌ PyTorch functionality test failed: {e}")
        return False

def test_data_formats():
    """Test weather data format support."""
    try:
        import xarray as xr
        import numpy as np
        
        # Create a sample weather dataset
        data = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], np.random.randn(10, 5, 5)),
            'precipitation': (['time', 'lat', 'lon'], np.random.rand(10, 5, 5))
        })
        
        print("✅ Weather data formats (xarray) working correctly")
        return True
    except Exception as e:
        print(f"❌ Weather data format test failed: {e}")
        return False

def main():
    """Run all environment tests."""
    print("🌦️  Weather Forecasting Environment Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print("=" * 50)
    
    all_passed = True
    
    # Test package imports
    print("\n📦 Testing Package Imports:")
    for package, description in REQUIRED_PACKAGES.items():
        if not test_package_import(package, description):
            all_passed = False
    
    print("\n🧪 Testing Functionality:")
    # Test PyTorch
    if not test_torch_functionality():
        all_passed = False
    
    # Test weather data formats
    if not test_data_formats():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Environment is ready for weather forecasting.")
        print("\n💡 Next steps:")
        print("   1. Start Jupyter: jupyter notebook")
        print("   2. Create your first notebook in the notebooks/ folder")
        print("   3. Begin weather data exploration!")
    else:
        print("⚠️  Some tests failed. Please install missing packages.")
        print("\n🔧 To fix issues:")
        print("   1. Activate environment: conda activate weather_forecast")
        print("   2. Install missing packages: pip install <package_name>")
        print("   3. Run this test again: python test_environment.py")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

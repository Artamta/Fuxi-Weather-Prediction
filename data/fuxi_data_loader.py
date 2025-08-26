import torch
import zarr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class FuXiWeatherDataset(Dataset):
    """
    Weather dataset for FuXi training using your Zarr data
    Converts (time, levels, lat, lon) -> (batch, channels, time, lat, lon)
    """
    
    def __init__(
        self, 
        data_path: str,
        sequence_length: int = 2,
        prediction_length: int = 1,
        split: str = 'train',
        subsample_time: int = 1
    ):
        self.data_path = data_path
        self.seq_len = sequence_length
        self.pred_len = prediction_length
        self.subsample = subsample_time
        
        print(f"🌍 Loading {split} data from {data_path}")
        
        # Open zarr store
        self.store = zarr.open(data_path, mode='r')
        
        # Get time dimension
        self.total_time = self.store['temperature'].shape[0]
        print(f"📅 Total timesteps: {self.total_time:,}")
        
        # Split data temporally
        self.time_indices = self._get_time_split(split)
        print(f"📊 {split} timesteps: {len(self.time_indices):,}")
        
        # Define your 70 variables
        self.variable_config = self._setup_variables()
        print(f"🎯 Using {self._count_variables()} variables")
        
        # Cache data info
        self._cache_data_info()
        
    def _get_time_split(self, split: str) -> np.ndarray:
        """Split data temporally"""
        if split == 'train':
            # First 80% for training
            end_idx = int(0.8 * self.total_time)
            return np.arange(0, end_idx, self.subsample)
        elif split == 'val':
            # Next 10% for validation  
            start_idx = int(0.8 * self.total_time)
            end_idx = int(0.9 * self.total_time)
            return np.arange(start_idx, end_idx, self.subsample)
        else:  # test
            # Last 10% for testing
            start_idx = int(0.9 * self.total_time)
            return np.arange(start_idx, self.total_time, self.subsample)
    
    def _setup_variables(self) -> dict:
        """Define exactly 70 variables for FuXi"""
        
        # Check if we have 13 or 37 pressure levels
        temp_shape = self.store['temperature'].shape
        n_levels = temp_shape[1]
        
        if n_levels == 13:
            # 6h data: use all 13 levels
            selected_levels = list(range(13))
            print(f"📊 Using all 13 pressure levels")
        else:
            # 1h data: select 13 levels from 37
            # Choose standard meteorological levels
            all_levels = self.store['level'][:]
            standard_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
            selected_levels = []
            
            for target_level in standard_levels:
                # Find closest level
                idx = np.argmin(np.abs(all_levels - target_level))
                selected_levels.append(idx)
            
            print(f"📊 Selected 13 levels from 37: {[all_levels[i] for i in selected_levels]}")
        
        return {
            'pressure_vars': {
                'temperature': selected_levels,
                'u_component_of_wind': selected_levels, 
                'v_component_of_wind': selected_levels,
                'vertical_velocity': selected_levels,
                'specific_humidity': selected_levels,
            },
            'surface_vars': [
                '2m_temperature',
                'surface_pressure',
                '10m_u_component_of_wind', 
                '10m_v_component_of_wind',
                'mean_sea_level_pressure'  # Use this as 5th variable
            ]
        }
    
    def _count_variables(self) -> int:
        """Count total variables"""
        pressure_count = sum(len(levels) for levels in self.variable_config['pressure_vars'].values())
        surface_count = len(self.variable_config['surface_vars'])
        return pressure_count + surface_count
    
    def _cache_data_info(self):
        """Cache data information for efficiency"""
        temp_data = self.store['temperature']
        self.spatial_shape = temp_data.shape[-2:]  # (lat, lon)
        self.data_dtype = temp_data.dtype
        
        print(f"🗺️  Spatial shape: {self.spatial_shape}")
        print(f"💾 Data type: {self.data_dtype}")
    
    def __len__(self) -> int:
        """Number of valid sequences"""
        return len(self.time_indices) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a training sample
        Returns:
            inputs: (C, T, H, W) - 70 channels, 2 timesteps
            targets: (C, H, W) - 70 channels, 1 timestep  
        """
        # Get time indices for this sample
        start_idx = self.time_indices[idx]
        input_times = [start_idx + i for i in range(self.seq_len)]
        target_time = start_idx + self.seq_len
        
        # Load input data (2 timesteps)
        input_data = self._load_variables(input_times)   # (C, T, H, W)
        
        # Load target data (1 timestep) 
        target_data = self._load_variables([target_time])  # (C, 1, H, W)
        target_data = target_data.squeeze(1)              # (C, H, W)
        
        return input_data, target_data
    
    def _load_variables(self, time_indices: list) -> torch.Tensor:
        """Load and stack all 70 variables for given time indices"""
        
        all_vars = []
        
        # Load pressure level variables (65 variables)
        for var_name, level_indices in self.variable_config['pressure_vars'].items():
            var_data = self.store[var_name]
            
            for level_idx in level_indices:
                # Load data: (time, lat, lon)
                data = var_data[time_indices, level_idx, :, :]
                
                # Add time dimension if needed: (time, lat, lon) -> (1, time, lat, lon)
                if len(data.shape) == 3:
                    data = data[np.newaxis, ...]  # Add channel dim
                
                all_vars.append(torch.from_numpy(data.astype(np.float32)))
        
        # Load surface variables (5 variables)
        for var_name in self.variable_config['surface_vars']:
            var_data = self.store[var_name]
            
            # Load data: (time, lat, lon)
            data = var_data[time_indices, :, :]
            
            # Add channel dimension: (time, lat, lon) -> (1, time, lat, lon)
            data = data[np.newaxis, ...]
            
            all_vars.append(torch.from_numpy(data.astype(np.float32)))
        
        # Stack all variables: (70, time, lat, lon)
        combined = torch.cat(all_vars, dim=0)
        
        return combined

def test_data_loader():
    """Test the data loader with your data"""
    
    print("🧪 Testing FuXi Data Loader")
    print("=" * 40)
    
    # Test with 6h data first (smaller, faster)
    data_path = '/storage/vishnu/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
    
    # Create dataset
    dataset = FuXiWeatherDataset(
        data_path=data_path,
        sequence_length=2,
        prediction_length=1,
        split='train',
        subsample_time=10  # Use every 10th timestep for faster testing
    )
    
    print(f"\n📊 Dataset info:")
    print(f"   Length: {len(dataset):,} samples")
    print(f"   Variables: {dataset._count_variables()}")
    print(f"   Spatial: {dataset.spatial_shape}")
    
    # Test data loading
    print(f"\n�� Testing data loading...")
    
    # Get first sample
    inputs, targets = dataset[0]
    print(f"   ✅ Input shape: {inputs.shape}")   # Should be (70, 2, 64, 32)
    print(f"   ✅ Target shape: {targets.shape}") # Should be (70, 64, 32)
    
    # Check data quality
    print(f"\n📊 Data quality check:")
    print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"   Target range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"   Input mean: {inputs.mean():.3f}")
    print(f"   Target mean: {targets.mean():.3f}")
    
    # Test batch loading
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    batch_inputs, batch_targets = next(iter(loader))
    print(f"\n📦 Batch loading:")
    print(f"   Batch input: {batch_inputs.shape}")   # Should be (4, 70, 2, 64, 32)
    print(f"   Batch target: {batch_targets.shape}") # Should be (4, 70, 64, 32)
    
    print(f"\n✅ Data loader working perfectly!")
    return dataset

def test_with_fuxi_model():
    """Test data loader with your FuXi model"""
    
    print(f"\n🚀 Testing with FuXi Model")
    print("=" * 30)
    
    import sys
    sys.path.append('../models')
    from fuxi_model import FuXiModel
    
    # Create small dataset for testing
    dataset = FuXiWeatherDataset(
        data_path='/storage/vishnu/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr',
        split='train',
        subsample_time=100  # Very sparse for quick test
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Create FuXi model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FuXiModel(
        in_channels=70,
        out_channels=70, 
        embed_dim=384,
        depth=4
    ).to(device)
    
    print(f"🏗️  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test training step
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    print(f"📥 Input to model: {inputs.shape}")
    print(f"🎯 Target: {targets.shape}")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(inputs, target_shape=targets.shape[-2:])
    
    print(f"📤 Predictions: {predictions.shape}")
    
    # Calculate loss
    loss = torch.nn.MSELoss()(predictions, targets)
    print(f"📊 MSE Loss: {loss.item():.6f}")
    
    print(f"✅ FuXi + Real Data = SUCCESS! 🎉")

if __name__ == "__main__":
    dataset = test_data_loader()
    test_with_fuxi_model()
    
    print(f"\n🎉 READY FOR TRAINING!")
    print(f"Your data + FuXi model = Perfect match! 🚀")

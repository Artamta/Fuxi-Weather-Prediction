import zarr
import numpy as np

def analyze_weather_variables():
    """Analyze your specific weather dataset"""
    
    print("🌍 Analyzing Your Weather Data")
    print("=" * 50)
    
    # Check both datasets
    datasets = {
        '6h_data': '/storage/vishnu/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr',
        '1h_data': '/storage/vishnu/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr'
    }
    
    for name, path in datasets.items():
        print(f"\n📊 {name.upper()}")
        print("-" * 30)
        
        store = zarr.open(path, mode='r')
        
        # Key variables for FuXi
        key_vars = [
            'temperature', 'u_component_of_wind', 'v_component_of_wind',
            'vertical_velocity', 'specific_humidity', 'geopotential',
            '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
            'surface_pressure', 'mean_sea_level_pressure'
        ]
        
        var_info = {}
        for var in key_vars:
            if var in store:
                arr = store[var]
                var_info[var] = {
                    'shape': arr.shape,
                    'dims': len(arr.shape),
                    'size_gb': arr.nbytes / 1e9
                }
                print(f"✅ {var:25s}: {arr.shape}")
            else:
                print(f"❌ {var:25s}: Not found")
        
        # Check pressure levels
        if 'level' in store:
            levels = store['level'][:]
            print(f"\n🏔️  Pressure levels: {len(levels)} levels")
            print(f"   Levels: {levels}")
        
        # Check time dimension
        if 'time' in store:
            time_arr = store['time']
            print(f"\n⏰ Time dimension: {time_arr.shape[0]:,} timesteps")
            
        # Estimate how to get 70 variables
        print(f"\n🎯 Getting 70 Variables:")
        
        if 'level' in store:
            levels = store['level']
            n_levels = len(levels)
            
            # 3D variables (with pressure levels)
            vars_3d = ['temperature', 'u_component_of_wind', 'v_component_of_wind', 
                      'vertical_velocity', 'specific_humidity']
            vars_3d_available = [v for v in vars_3d if v in store]
            total_3d = len(vars_3d_available) * n_levels
            
            # Surface variables
            vars_surface = ['2m_temperature', 'surface_pressure', '10m_u_component_of_wind',
                           '10m_v_component_of_wind', 'total_precipitation_6hr']
            vars_surface_available = [v for v in vars_surface if v in store]
            total_surface = len(vars_surface_available)
            
            print(f"   📊 3D variables: {len(vars_3d_available)} × {n_levels} levels = {total_3d}")
            print(f"   📊 Surface variables: {total_surface}")
            print(f"   📊 Total: {total_3d + total_surface}")
            
            if total_3d + total_surface >= 70:
                print(f"   ✅ Perfect! You have enough for 70 variables!")
            else:
                needed = 70 - (total_3d + total_surface)
                print(f"   ⚠️  Need {needed} more variables")
                
                # Suggest additional variables
                extra_vars = ['total_cloud_cover', 'total_column_water_vapour', 
                            'sea_surface_temperature', 'geopotential_at_surface']
                print(f"   💡 Add these: {extra_vars[:needed]}")

def create_fuxi_variable_config():
    """Create the exact 70-variable configuration for FuXi"""
    
    print(f"\n🔧 FuXi Variable Configuration")
    print("=" * 40)
    
    # This is your exact recipe for 70 variables
    config = {
        'pressure_level_vars': {
            'temperature': 13,           # 13 pressure levels
            'u_component_of_wind': 13,   
            'v_component_of_wind': 13,   
            'vertical_velocity': 13,     
            'specific_humidity': 13,     
        },
        'surface_vars': [
            '2m_temperature',
            'surface_pressure', 
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation_6hr'
        ]
    }
    
    total_3d = sum(config['pressure_level_vars'].values())
    total_surface = len(config['surface_vars'])
    total = total_3d + total_surface
    
    print(f"📊 Variable breakdown:")
    print(f"   🏔️  Pressure level variables: {total_3d}")
    for var, count in config['pressure_level_vars'].items():
        print(f"      {var}: {count} levels")
    
    print(f"   🌍 Surface variables: {total_surface}")
    for var in config['surface_vars']:
        print(f"      {var}")
    
    print(f"\n🎯 Total: {total} variables")
    
    if total == 70:
        print(f"✅ Perfect! Exactly 70 variables for FuXi!")
    else:
        print(f"⚠️  Need adjustment: {70 - total} variables")
    
    return config

if __name__ == "__main__":
    analyze_weather_variables()
    config = create_fuxi_variable_config()
    
    print(f"\n🚀 Next step: Create data loader with this configuration!")

import torch
import xarray as xr
import numpy as np

from cube_embedding import CubeEmbedding3D

PRESSURE_VARS = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"]
SURFACE_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]

ds = xr.open_dataset("mini_era5_sample.nc")
print(ds.data_vars.keys())

pressure = ds[PRESSURE_VARS].to_array().isel(time=slice(0, 2))     # (4, 2, 13, lat, lon)
surface = ds[SURFACE_VARS].to_array().isel(time=slice(0, 2))       # (5, 2, lat, lon)

pressure_np = pressure.values.reshape(
    pressure.shape[0] * pressure.shape[2],  # 4 variables * 13 levels
    pressure.shape[1],                      # time
    pressure.shape[3],                      # lat
    pressure.shape[4],                      # lon
)

surface_np = surface.values.reshape(
    surface.shape[0],
    surface.shape[1],
    surface.shape[2],
    surface.shape[3],
)

combined = np.concatenate([pressure_np, surface_np], axis=0)       # (channels, time, lat, lon)

x = torch.from_numpy(combined).float().unsqueeze(0)                # (1, C, T, H, W)

embed = CubeEmbedding3D(in_channels=x.shape[1], embed_dim=384)
out = embed(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)
import os
import numpy as np
import torch
from train import MiniFuXiDataset

# Paths
train_path = "/home/raj.ayush/New/Fuxi-Weather-Prediction/Fuxi/Fuxi_3P/Data/train_data_1959_2017.nc"
forecast_dir = "/home/raj.ayush/New/Fuxi-Weather-Prediction/Fuxi/Fuxi_3P/Forecast"

# Variable order as in your dataset/model
var_names = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
]

# 1. Load training set to get mean and std
train_set = MiniFuXiDataset(train_path, history_steps=2)
mean = train_set.mean.squeeze().numpy()  # shape: (channels,)
std = train_set.std.squeeze().numpy()    # shape: (channels,)
all_var_names = train_set.var_names      # full list of variable names in order

# 2. For each surface variable, find its index, denormalize, and save
for var in var_names:
    # Find the index of this variable in the full variable list
    try:
        idx = all_var_names.index(var)
    except ValueError:
        print(f"Variable {var} not found in training set variable list!")
        continue

    npy_file = os.path.join(forecast_dir, f"{var}_model_pred.npy")
    if not os.path.exists(npy_file):
        print(f"File not found: {npy_file}")
        continue

    arr = np.load(npy_file)  # shape: (n_cases, n_leads, lat, lon)
    # Denormalize using the correct mean and std for this variable
    denorm_arr = arr * std[idx] + mean[idx]
    out_file = os.path.join(forecast_dir, f"{var}_model_pred_denorm.npy")
    np.save(out_file, denorm_arr)
    print(f"Denormalized and saved: {out_file}")

print("All done.")
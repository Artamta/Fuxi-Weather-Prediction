import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from fuxi_small import FuXiModel
from traning_fuxi_small import MiniFuXiDataset, get_variable_names

def get_variable_names(dataset):
    pressure_vars = ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind", "geopotential"]
    surface_vars = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "surface_pressure"]
    levels = dataset.data.shape[1] // len(pressure_vars) if hasattr(dataset, 'data') else 13  # adjust if needed
    var_names = []
    for p in pressure_vars:
        for l in range(levels):
            var_names.append(f"{p}_lev{l}")
    var_names.extend(surface_vars)
    return var_names

# ---- USER SETTINGS ----
test_data_path = "test_data.nc"
checkpoint_path = "checkpoints/fuxi_epoch91.pt"  # <-- set your best epoch
batch_size = 512  # adjust as needed

# ---- MODEL HYPERPARAMETERS (set as used in training) ----
embed_dim = 96
depths = [2, 2, 4]
num_heads = [2, 2, 4]
num_down_blocks = 2
num_up_blocks = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- LOAD DATA ----
test_set = MiniFuXiDataset(test_data_path, history_steps=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
variable_names = get_variable_names(test_set)

# ---- LOAD MODEL ----
model = FuXiModel(
    in_channels=test_set.data.shape[1],
    out_channels=test_set.data.shape[1],
    swin_window_size=8,
    embed_dim=embed_dim,
    input_height=test_set.data.shape[-2],
    input_width=test_set.data.shape[-1],
    depths=depths,
    num_heads=num_heads,
    num_down_blocks=num_down_blocks,
    num_up_blocks=num_up_blocks,
).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---- EVALUATE PER-VARIABLE MAE ----
all_mae = []
with torch.no_grad():
    for history, target in test_loader:
        history = history.to(device)
        target = target.to(device)
        pred = model(history, target_shape=target.shape[-2:])
        abs_err = torch.abs(pred - target)  # (B, C, H, W)
        var_mae = abs_err.mean(dim=(0, 2, 3)).cpu().numpy()  # (C,)
        all_mae.append(var_mae)
all_mae = np.stack(all_mae).mean(axis=0)  # (C,)

# ---- SAVE & PRINT RESULTS ----
df = pd.DataFrame({
    "Variable": variable_names,
    "MAE": all_mae
})
df.to_csv("per_variable_mae.csv", index=False)
print(df)

print("\nTop 5 easiest variables (lowest MAE):")
print(df.nsmallest(5, "MAE"))
print("\nTop 5 hardest variables (highest MAE):")
print(df.nlargest(5, "MAE"))
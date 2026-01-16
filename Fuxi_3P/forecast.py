import numpy as np
import xarray as xr
import torch
import sys
import json
import os

# --- Paths ---
BASE_DIR = "/home/raj.ayush/New/Fuxi-Weather-Prediction/Fuxi/Fuxi_3P"
FORECAST_DIR = os.path.join(BASE_DIR, "15_days_forecast")
os.makedirs(FORECAST_DIR, exist_ok=True)
TEST_DATA = os.path.join(BASE_DIR, "Data/test_data_2021_2023.nc")
MODEL_PATH = os.path.join(BASE_DIR, "Models/exp_FuXi3Stage_20251118_005002/checkpoints/fuxi_epoch023.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "Models/exp_FuXi3Stage_20251118_005002/config.json")

# --- Variable definitions (must match training) ---
PRESSURE_LEVELS = [250, 500, 850]
PRESSURE_VARS = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]
SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
]

# --- Import FuXiModel ---
sys.path.append(BASE_DIR)
from fuxi import FuXiModel

# --- Load model config and weights ---
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
model = FuXiModel(
    in_channels=int(cfg["in_channels"]),
    out_channels=int(cfg["out_channels"]),
    embed_dim=int(cfg["embed_dim"]),
    encoder_dims=tuple(int(x) for x in cfg["encoder_dims"].split(",")),
    swin_depths=tuple(int(x) for x in cfg["swin_depths"].split(",")),
    swin_heads=tuple(int(x) for x in cfg["swin_heads"].split(",")),
    swin_window_size=int(cfg["swin_window_size"]),
    drop_path_rate=float(cfg["drop_path_rate"]),
    input_height=int(cfg["input_height"]),
    input_width=int(cfg["input_width"]),
)
state = torch.load(MODEL_PATH, map_location="cpu")
if "model_state" in state:
    state_dict = state["model_state"]
else:
    state_dict = state
model.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()})
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Load test data ---
ds = xr.open_dataset(TEST_DATA)
history_steps = 2
n_leads = 15

# --- Prepare input data (20 channels, order must match training) ---
input_vars = []
for var in PRESSURE_VARS:
    for lvl in PRESSURE_LEVELS:
        input_vars.append(ds[var].sel(level=lvl).values)
for var in SURFACE_VARS:
    input_vars.append(ds[var].values)
input_data = np.stack(input_vars, axis=1)  # (time, 20, H, W)
n_times, n_channels, H, W = input_data.shape

# --- Run autoregressive forecast and save ---
n_cases = n_times - history_steps - n_leads + 1
all_preds = np.zeros((n_cases, n_leads, n_channels, H, W), dtype=np.float32)

print(f"Running {n_cases} forecast cases, each for {n_leads} leads...")

for case_idx in range(n_cases):
    # Prepare initial history (real data)
    history = input_data[case_idx : case_idx + history_steps]  # (history, 20, H, W)
    history = torch.from_numpy(history).float().unsqueeze(0).to(device)  # (1, history, 20, H, W)
    history = history.permute(0, 2, 1, 3, 4)  # (1, 20, history, H, W)
    preds = []
    h = history.clone()
    with torch.no_grad():
        for lead in range(n_leads):
            pred = model(h)[0].detach().cpu().numpy()  # (20, H, W)
            preds.append(pred)
            pred_torch = torch.from_numpy(pred).unsqueeze(0).unsqueeze(2).to(device)  # (1, 20, 1, H, W)
            h = torch.cat([h[:, :, 1:], pred_torch], dim=2)
    preds = np.stack(preds, axis=0)  # (n_leads, 20, H, W)
    all_preds[case_idx] = preds
    if (case_idx + 1) % 100 == 0 or (case_idx + 1) == n_cases:
        print(f"Completed {case_idx + 1}/{n_cases} cases")

# --- Save the forecast ---
save_path = os.path.join(FORECAST_DIR, "fuxi_15day_forecast.npy")
np.save(save_path, all_preds)
print(f"Saved forecast to {save_path}")
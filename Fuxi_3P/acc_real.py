import os
import torch
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List
import zarr
from fuxi import FuXiModel
from train import MiniFuXiDataset

# ---- CONFIG ----
CLIM_ROOT = "/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5-hourly-climatology/1990-2017_6h_64x32_equiangular_conservative.zarr"
TEST_DATA = "New/Fuxi-Weather-Prediction/Fuxi/Fuxi_3P/Data/test_data_2021_2023.nc"
CKPT = "New/Fuxi-Weather-Prediction/Fuxi/Fuxi_3P/Models/exp_FuXi3Stage_20251118_005002/checkpoints/fuxi_epoch023.pt"
CONFIG = "New/Fuxi-Weather-Prediction/Fuxi/Fuxi_3P/Models/exp_FuXi3Stage_20251118_005002/config.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEAD_STEPS = 15
HISTORY_STEPS = 2
NUM_SAMPLES = 12

# ---- CLIMATOLOGY LOADER ----


# Load reference lat/lon arrays from your test data
lat_ref = xr.open_dataset(TEST_DATA)["lat"].values
lon_ref = xr.open_dataset(TEST_DATA)["lon"].values

def load_climatology(var_names: List[str], hours: List[int], lat, lon):
    clim_list = []
    for var in var_names:
        clim_path = f"{CLIM_ROOT}/{var.split('_plev')[0] if '_plev' in var else var}"
        zarr_arr = zarr.open_array(clim_path, mode="r")
        # Try to get lat/lon from Zarr attributes (WeatherBench2 stores them as attrs)
        lat_arr = zarr_arr.attrs.get("lat")
        lon_arr = zarr_arr.attrs.get("lon")
        # If not present, fallback to reference arrays and slice to match shape
        if lat_arr is None or lon_arr is None:
            lat_ref = xr.open_dataset(TEST_DATA)["lat"].values
            lon_ref = xr.open_dataset(TEST_DATA)["lon"].values
            lat_arr = lat_ref[-zarr_arr.shape[-2]:]
            lon_arr = lon_ref[-zarr_arr.shape[-1]:]
        # Build DataArray with correct dims
        if zarr_arr.ndim == 5:
            dims = ("hour", "day_of_year", "level", "lat", "lon")
            clim_ds = xr.DataArray(zarr_arr, dims=dims)
            clim_ds = clim_ds.isel(day_of_year=0)
            if "_plev" in var:
                level = int(var.split("_plev")[1])
                available_levels = clim_ds.coords["level"].values
                nearest_level = available_levels[np.argmin(np.abs(available_levels - level))]
                clim_ds = clim_ds.sel(level=nearest_level)
        elif zarr_arr.ndim == 4:
            dims = ("hour", "level", "lat", "lon")
            clim_ds = xr.DataArray(zarr_arr, dims=dims)
            if "_plev" in var:
                level = int(var.split("_plev")[1])
                available_levels = clim_ds.coords["level"].values
                nearest_level = available_levels[np.argmin(np.abs(available_levels - level))]
                clim_ds = clim_ds.sel(level=nearest_level)
        elif zarr_arr.ndim == 3:
            dims = ("hour", "lat", "lon")
            clim_ds = xr.DataArray(zarr_arr, dims=dims)
        else:
            raise ValueError(f"Unexpected shape for {var}: {zarr_arr.shape}")
        # Assign correct lat/lon coordinates
        clim_ds = clim_ds.assign_coords(lat=("lat", lat_arr), lon=("lon", lon_arr))
        arr = []
        for h in hours:
            arr.append(clim_ds.isel(hour=h).sel(lat=lat, lon=lon, method="nearest").values)
        clim_list.append(np.stack(arr, axis=0))  # (lead_steps, H, W)
    clim = np.stack(clim_list, axis=1)  # (lead_steps, C, H, W)
    return torch.from_numpy(clim).float()
# ---- METRICS ----
def latitude_weights(latitudes, device):
    w = torch.cos(torch.deg2rad(torch.tensor(latitudes, dtype=torch.float32, device=device)))
    return w / (w.mean() + 1e-8)

def latitude_weighted_acc(pred, target, climatology, latitudes):
    weights = latitude_weights(latitudes, pred.device).view(1, 1, -1, 1)
    pred_anom = pred - climatology
    targ_anom = target - climatology
    num = (weights * pred_anom * targ_anom).sum(dim=(-2, -1))
    den = torch.sqrt(
        (weights * pred_anom.pow(2)).sum(dim=(-2, -1))
        * (weights * targ_anom.pow(2)).sum(dim=(-2, -1))
        + 1e-8
    )
    return (num / den).mean().item()

# ---- STATE DICT UNWRAP ----
def unwrap_state_dict(state_dict):
    # Remove 'module.' prefix if present
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

# ---- EVALUATION ----
def evaluate(model, dataset, device, lead_steps=15, num_samples=12):
    model.eval()
    latitudes = dataset.latitudes
    longitudes = dataset.longitudes
    accs = []
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset) - lead_steps)
        history, _ = dataset[idx]
        fut = [dataset[idx + l][1] for l in range(lead_steps)]
        fut = torch.stack(fut, dim=0)  # (lead_steps, C, H, W)
        # Extract hour for each lead (assuming dataset has .time as np.datetime64)
        hours = []
        if hasattr(dataset, "time"):
            for l in range(lead_steps):
                t = dataset.time[idx + dataset.history + l]
                hour = int(str(t).split("T")[1][:2])
                hours.append(hour)
        else:
            hours = [0] * lead_steps  # fallback if no time info
        clim = load_climatology(dataset.var_names, hours, latitudes, longitudes).to(device)  # (lead_steps, C, H, W)
        history = history.unsqueeze(0).to(device)
        preds = []
        with torch.no_grad():
            h = history.clone()
            for l in range(lead_steps):
                pred = model(h, target_shape=fut[l].shape[-2:])
                preds.append(pred[0].detach())
                h = torch.cat([h[:, :, 1:], pred.unsqueeze(2)], dim=2)
        preds = torch.stack(preds, dim=0)  # (lead_steps, C, H, W)
        acc = []
        for l in range(lead_steps):
            acc.append(latitude_weighted_acc(
                preds[l:l+1], fut[l:l+1].to(device), clim[l:l+1], latitudes
            ))
        accs.append(acc)
        print(f"Sample {i+1}/{num_samples} done.")
    accs = np.array(accs)
    acc_mean = np.mean(accs, axis=0)
    print("Mean ACC vs lead:", acc_mean)
    return acc_mean

# ---- MAIN ----
if __name__ == "__main__":
    import json
    with open(CONFIG) as f:
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
    ).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    # --- FIX for DataParallel/DDP checkpoints ---
    if "model_state" in state:
        state_dict = state["model_state"]
    else:
        state_dict = state
    state_dict = unwrap_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = MiniFuXiDataset(TEST_DATA, history_steps=HISTORY_STEPS)
    acc_mean = evaluate(model, dataset, DEVICE, lead_steps=LEAD_STEPS, num_samples=NUM_SAMPLES)
    print("Final ACC vs lead:", acc_mean)
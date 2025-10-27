import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import xarray as xr

from fuxi import FuXiModel

def latitude_weighted_l1_loss(pred, target, latitudes):
    """
    pred, target: (batch, channels, lat, lon)
    latitudes: (lat,) in degrees, e.g. [-90, ..., 90]
    """
    weights = torch.cos(torch.deg2rad(latitudes)).to(pred.device)  # (lat,)
    weights = weights / weights.mean()  # Normalize so mean=1
    weights = weights.view(1, 1, -1, 1)
    loss = torch.abs(pred - target) * weights
    return loss.mean()

class MiniFuXiDataset(Dataset):
    def __init__(self, path: str, history_steps: int = 2, target_step: int = 1):
        ds = xr.open_dataset(path)
        print("Loaded variables:", list(ds.data_vars.keys()))
        rename_map = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in ds.dims}
        ds = ds.rename(rename_map)

        pressure_vars = ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind", "geopotential"]
        surface_vars = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "surface_pressure"]

        pressure = ds[pressure_vars].to_array().transpose("time", "variable", "level", "lat", "lon")
        surface = ds[surface_vars].to_array().transpose("time", "variable", "lat", "lon")

        p_np = pressure.values.reshape(pressure.shape[0], pressure.shape[1] * pressure.shape[2], pressure.shape[3], pressure.shape[4])
        s_np = surface.values  # already (time, 5, lat, lon)

        data = torch.from_numpy(np.concatenate([p_np, s_np], axis=1)).float()

        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

        self.data = (data - mean) / std
        self.mean = mean.squeeze(0)
        self.std = std.squeeze(0)
        self.history = history_steps
        self.target_offset = target_step
        self.latitudes = ds['lat'].values if 'lat' in ds else np.linspace(-90, 90, data.shape[-2])
        ds.close()

    def __len__(self):
        return len(self.data) - self.history - self.target_offset + 1

    def __getitem__(self, idx):
        past = self.data[idx : idx + self.history]
        target = self.data[idx + self.history + self.target_offset - 1]
        past = past.permute(1, 0, 2, 3)
        return past, target

def train_one_epoch(model, loader, optimizer, device, latitudes):
    model.train()
    total = 0.0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        pred = model(history, target_shape=target.shape[-2:])
        loss = latitude_weighted_l1_loss(pred, target, latitudes)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def eval_one_epoch(model, loader, device, latitudes):
    model.eval()
    total_l1, total_mae, batches = 0.0, 0.0, 0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        pred = model(history, target_shape=target.shape[-2:])
        l1 = latitude_weighted_l1_loss(pred, target, latitudes)
        mae = torch.mean(torch.abs(pred - target))
        total_l1 += l1.item()
        total_mae += mae.item()
        batches += 1
    return total_l1 / batches, total_mae / batches

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MiniFuXiDataset("big.nc")

    val_len = max(1, int(0.2 * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    spatial_shape = tuple(dataset.data.shape[-2:])
    channels = dataset.data.shape[1]
    latitudes = torch.tensor(dataset.latitudes, dtype=torch.float32)

    model = FuXiModel(
        in_channels=channels,
        out_channels=channels,
        swin_window_size=8,
        embed_dim=96,
        input_height=spatial_shape[0],
        input_width=spatial_shape[1],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2.5e-4, betas=(0.9, 0.95), weight_decay=0.1)

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    for epoch in range(20):
        print(f"\n=== Epoch {epoch + 1} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, latitudes)
        val_l1, val_mae = eval_one_epoch(model, val_loader, device, latitudes)
        print(f"Epoch {epoch + 1}: train {train_loss:.4f} | val_l1 {val_l1:.4f} | val_mae {val_mae:.4f}")

        if val_l1 < best_val:
            best_val = val_l1
            ckpt_path = f"checkpoints/fuxi_epoch{epoch + 1:02d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_l1": val_l1,
                    "val_mae": val_mae,
                },
                ckpt_path,
            )

if __name__ == "__main__":
    main()
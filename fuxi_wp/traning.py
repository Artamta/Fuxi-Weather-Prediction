import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import xarray as xr

from fuxi import FuXiTinyBackbone

class MiniFuXiDataset(Dataset):
    def __init__(self, path: str, history_steps: int = 2, target_step: int = 1):
        ds = xr.open_dataset(path)
        print(ds.data_vars.keys())
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
        ds.close()

    def __len__(self):
        return len(self.data) - self.history - self.target_offset + 1

    def __getitem__(self, idx):
        past = self.data[idx : idx + self.history]
        target = self.data[idx + self.history + self.target_offset - 1]
        past = past.permute(1, 0, 2, 3)
        return past, target


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = model(history)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_mse, total_mae, batches = 0.0, 0.0, 0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        pred = model(history)
        mse = criterion(pred, target)
        mae = torch.mean(torch.abs(pred - target))
        total_mse += mse.item()
        total_mae += mae.item()
        batches += 1
    return total_mse / batches, total_mae / batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MiniFuXiDataset("mini_era5_sample.nc")

    val_len = max(1, int(0.2 * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    spatial_shape = tuple(dataset.data.shape[-2:])
    channels = dataset.data.shape[1]

    model = FuXiTinyBackbone(
        in_channels=channels,
        output_channels=channels,
        input_shape=spatial_shape,
        window_size=8,
        depths=(6, 6),
        num_heads=(6, 6),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, val_mae = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: train {train_loss:.4f} | val_mse {val_mse:.4f} | val_mae {val_mae:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            ckpt_path = f"checkpoints/fuxi_tiny_epoch{epoch + 1:02d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
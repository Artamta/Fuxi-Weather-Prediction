import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim

from fuxi_closed import FuXiModel

# Try to import xarray, but don't fail if not available
try:
    import xarray as xr
    XR_AVAILABLE = True
except ImportError:
    XR_AVAILABLE = False

class MiniFuXiDataset(Dataset):
    def __init__(self, path: str, history_steps: int = 2, target_step: int = 1):
        if not XR_AVAILABLE:
            raise RuntimeError("xarray is not installed!")
        ds = xr.open_dataset(path)
        print("[DEBUG] Dataset variables:", list(ds.variables))
        # Assume all variables are stacked as (time, variable, lat, lon)
        data = ds.to_array().transpose("time", "variable", "lat", "lon").values
        print("[DEBUG] Raw data shape from NetCDF:", data.shape)
        data = torch.from_numpy(data).float()
        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
        self.data = (data - mean) / std
        self.mean = mean.squeeze(0)
        self.std = std.squeeze(0)
        self.history = history_steps
        self.target_offset = target_step
        ds.close()
        print("[DEBUG] Normalized data shape:", self.data.shape)

    def __len__(self):
        return len(self.data) - self.history - self.target_offset + 1

    def __getitem__(self, idx):
        past = self.data[idx : idx + self.history]  # (history, channels, H, W)
        target = self.data[idx + self.history + self.target_offset - 1]  # (channels, H, W)
        past = past.permute(1, 0, 2, 3)  # (channels, history, H, W)
        return past, target

class SyntheticFuXiDataset(Dataset):
    def __init__(self, num_samples=100, in_channels=70, history_steps=2, height=180, width=360):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.history_steps = history_steps
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        past = torch.randn(self.in_channels, self.history_steps, self.height, self.width)
        target = torch.randn(self.in_channels, self.height, self.width)
        return past, target

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for batch_idx, (history, target) in enumerate(loader):
        print(f"[DEBUG] Batch {batch_idx}: history {history.shape}, target {target.shape}")
        history = history.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = model(history)
        print(f"[DEBUG] Model output shape: {pred.shape}")
        loss = criterion(pred, target)
        print(f"[DEBUG] Loss: {loss.item():.6f}")
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_mse, total_mae, batches = 0.0, 0.0, 0
    for batch_idx, (history, target) in enumerate(loader):
        history = history.to(device)
        target = target.to(device)
        pred = model(history)
        mse = criterion(pred, target)
        mae = torch.mean(torch.abs(pred - target))
        print(f"[DEBUG][VAL] Batch {batch_idx}: mse {mse.item():.6f}, mae {mae.item():.6f}")
        total_mse += mse.item()
        total_mae += mae.item()
        batches += 1
    return total_mse / batches, total_mae / batches

def main():
    # Device selection: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Try to load real data, else use synthetic
    try:
        dataset = MiniFuXiDataset("/Users/ayush/Desktop/weather_forcast/fuxi_wp/mini_era5_sample.nc", history_steps=2, target_step=1)
        print("[INFO] Loaded real NetCDF dataset.")
    except Exception as e:
        print(f"[WARN] Failed to load NetCDF: {e}")
        print("[INFO] Using synthetic data for debugging.")
        dataset = SyntheticFuXiDataset(num_samples=20, in_channels=70, history_steps=2, height=180, width=360)

    val_len = max(1, int(0.2 * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # Model: low-res for Mac
    in_channels = 70
    model = FuXiModel(
        in_channels=in_channels,
        out_channels=in_channels,
        embed_dim=64,           # Small for Mac!
        swin_window_size=4,
        input_height=180,   # <-- add this
        input_width=360
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    for epoch in range(3):  # Fewer epochs for quick test
        print(f"\n[INFO] Starting epoch {epoch + 1}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, val_mae = eval_one_epoch(model, val_loader, criterion, device)
        print(f"[RESULT] Epoch {epoch + 1}: train {train_loss:.4f} | val_mse {val_mse:.4f} | val_mae {val_mae:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            ckpt_path = f"checkpoints/fuxi_mac_epoch{epoch + 1:02d}.pt"
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
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
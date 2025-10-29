import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import xarray as xr
import matplotlib.pyplot as plt

from fuxi import FuXiModel

def latitude_weighted_l1_loss(pred, target, latitudes):
    weights = torch.cos(torch.deg2rad(latitudes)).to(pred.device)
    weights = weights / weights.mean()
    weights = weights.view(1, 1, -1, 1)
    loss = torch.abs(pred - target) * weights
    return loss.mean()

class MiniFuXiDataset(Dataset):
    def __init__(self, path: str, history_steps: int = 2):
        ds = xr.open_dataset(path)
        print(f"Loaded {path} variables:", list(ds.data_vars.keys()))
        rename_map = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in ds.dims}
        ds = ds.rename(rename_map)

        pressure_vars = ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind", "geopotential"]
        surface_vars = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "surface_pressure"]

        pressure = ds[pressure_vars].to_array().transpose("time", "variable", "level", "lat", "lon")
        surface = ds[surface_vars].to_array().transpose("time", "variable", "lat", "lon")

        p_np = pressure.values.reshape(pressure.shape[0], pressure.shape[1] * pressure.shape[2], pressure.shape[3], pressure.shape[4])
        s_np = surface.values

        data = torch.from_numpy(np.concatenate([p_np, s_np], axis=1)).float()

        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

        self.data = (data - mean) / std
        self.mean = mean.squeeze(0)
        self.std = std.squeeze(0)
        self.history = history_steps

        # Always match latitudes to the data shape
        latitudes = ds['lat'].values
        if latitudes.shape[0] != data.shape[-2]:
            # Subsample or slice to match data shape
            latitudes = latitudes[:data.shape[-2]]
        self.latitudes = latitudes
        ds.close()

    def __len__(self):
        return len(self.data) - self.history

    def __getitem__(self, idx):
        past = self.data[idx : idx + self.history]
        target = self.data[idx + self.history]
        past = past.permute(1, 0, 2, 3)  # (C, history, H, W)
        return past, target

def train_one_epoch(model, loader, optimizer, device, latitudes):
    model.train()
    total = 0.0
    for batch_idx, (history, target) in enumerate(loader):
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

def plot_losses(train_losses, val_losses, outdir):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Latitude-weighted L1 Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

def main():
    # Slurm: get output dir from env or use default
    outdir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
    os.makedirs(os.path.join(outdir, "checkpoints"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_steps = 2

    train_set = MiniFuXiDataset("train_data.nc", history_steps=history_steps)
    val_set = MiniFuXiDataset("val_data.nc", history_steps=history_steps)
    test_set = MiniFuXiDataset("test_data.nc", history_steps=history_steps)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    spatial_shape = tuple(train_set.data.shape[-2:])
    channels = train_set.data.shape[1]
    latitudes = torch.tensor(train_set.latitudes, dtype=torch.float32)

    model = FuXiModel(
        in_channels=channels,
        out_channels=channels,
        swin_window_size=8,
        embed_dim=96,
        input_height=spatial_shape[0],
        input_width=spatial_shape[1],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2.5e-4, betas=(0.9, 0.95), weight_decay=0.1)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, 21):
        print(f"\n=== Epoch {epoch} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, latitudes)
        val_l1, val_mae = eval_one_epoch(model, val_loader, device, latitudes)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_l1={val_l1:.4f} | val_mae={val_mae:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_l1)

        if val_l1 < best_val:
            best_val = val_l1
            ckpt_path = os.path.join(outdir, "checkpoints", f"fuxi_epoch{epoch:02d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_l1": val_l1,
                    "val_mae": val_mae,
                },
                ckpt_path,
            )
            print(f"  [Checkpoint] Saved best model at epoch {epoch}")

    plot_losses(train_losses, val_losses, outdir)

    # Final test evaluation
    test_l1, test_mae = eval_one_epoch(model, test_loader, device, latitudes)
    print(f"\nTest set: l1={test_l1:.4f} | mae={test_mae:.4f}")

    # Plot a few predictions vs targets for visual check
    with torch.no_grad():
        history, target = next(iter(test_loader))
        history = history.to(device)
        target = target.to(device)
        pred = model(history, target_shape=target.shape[-2:])
        # Plot first variable, first sample
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(target[0, 0].cpu(), cmap="coolwarm")
        plt.title("Target (sample 0, var 0)")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(pred[0, 0].cpu(), cmap="coolwarm")
        plt.title("Prediction (sample 0, var 0)")
        plt.colorbar()
        plt.suptitle("Test Sample: Target vs Prediction")
        plt.savefig(os.path.join(outdir, "test_sample_pred_vs_target.png"))
        plt.close()

if __name__ == "__main__":
    main()

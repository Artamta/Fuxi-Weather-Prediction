import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import xarray as xr
import matplotlib.pyplot as plt

from fuxi_small import FuXiModel  # Make sure this matches your model file

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

        latitudes = ds['lat'].values
        if latitudes.shape[0] != data.shape[-2]:
            latitudes = latitudes[:data.shape[-2]]
        self.latitudes = latitudes
        ds.close()

        # Save variable names for plotting
        self.pressure_vars = pressure_vars
        self.surface_vars = surface_vars
        self.levels = pressure.shape[2] if len(pressure.shape) == 5 else 1

    def __len__(self):
        return len(self.data) - self.history

    def __getitem__(self, idx):
        past = self.data[idx : idx + self.history]
        target = self.data[idx + self.history]
        past = past.permute(1, 0, 2, 3)  # (C, history, H, W)
        return past, target

def train_one_epoch(model, loader, optimizer, device, latitudes, scaler):
    model.train()
    total = 0.0
    for batch_idx, (history, target) in enumerate(loader):
        history = history.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(history, target_shape=target.shape[-2:])
            loss = latitude_weighted_l1_loss(pred, target, latitudes)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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

def get_variable_names(dataset):
    pressure_vars = dataset.pressure_vars
    surface_vars = dataset.surface_vars
    levels = dataset.levels
    var_names = []
    for p in pressure_vars:
        for l in range(levels):
            var_names.append(f"{p}_lev{l}")
    var_names.extend(surface_vars)
    return var_names

# --- Auto batch size finder ---
def find_max_batch_size(model, dataset, device, start=16, step=64, max_batch=2048):
    batch_size = start
    last_good = start
    while batch_size <= max_batch:
        try:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            history, target = next(iter(loader))
            history = history.to(device)
            target = target.to(device)
            with torch.cuda.amp.autocast():
                pred = model(history, target_shape=target.shape[-2:])
            last_good = batch_size
            batch_size += step
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch size {batch_size}, last good: {last_good}")
                break
            else:
                raise e
    return last_good

# --- Early stopping utility ---
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def main():
    outdir = os.getcwd() 
    os.makedirs(os.path.join(outdir, "checkpoints"), exist_ok=True)
    plots_dir = os.path.join(outdir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_steps = 2

    train_set = MiniFuXiDataset("train_data.nc", history_steps=history_steps)
    val_set = MiniFuXiDataset("val_data.nc", history_steps=history_steps)
    test_set = MiniFuXiDataset("test_data.nc", history_steps=history_steps)

    # --- Find max batch size ---
    model = FuXiModel(
        in_channels=train_set.data.shape[1],
        out_channels=train_set.data.shape[1],
        swin_window_size=8,
        embed_dim=192,
        input_height=train_set.data.shape[-2],
        input_width=train_set.data.shape[-1],
    ).to(device)
    max_batch = find_max_batch_size(model, train_set, device)
    print(f"Max batch size that fits: {max_batch}")

    # --- DataLoaders with optimal batch size ---
    train_loader = DataLoader(train_set, batch_size=max_batch, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=max_batch, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=max_batch, shuffle=False, num_workers=8, pin_memory=True)

    latitudes = torch.tensor(train_set.latitudes, dtype=torch.float32)

    optimizer = optim.AdamW(model.parameters(), lr=2.5e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    train_losses, val_losses = [], []
    early_stopper = EarlyStopping(patience=10, min_delta=1e-4)

    for epoch in range(1, 500):
        print(f"\n=== Epoch {epoch} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, latitudes, scaler)
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

        # --- Early stopping check ---
        early_stopper(val_l1)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    plot_losses(train_losses, val_losses, plots_dir)

    # Final test evaluation
    test_l1, test_mae = eval_one_epoch(model, test_loader, device, latitudes)
    print(f"\nTest set: l1={test_l1:.4f} | mae={test_mae:.4f}")

    # Plot a few predictions vs targets for visual check
    with torch.no_grad():
        history_batch, target_batch = next(iter(test_loader))
        history_batch = history_batch.to(device)
        target_batch = target_batch.to(device)
        pred_batch = model(history_batch, target_shape=target_batch.shape[-2:])

        variable_names = get_variable_names(train_set)
        num_samples = min(5, history_batch.shape[0])  # Plot up to 5 samples
        num_vars = min(5, target_batch.shape[1])      # Plot up to 5 variables

        for i in range(num_samples):
            for v in range(num_vars):
                var_name = variable_names[v] if v < len(variable_names) else f"var{v}"
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(target_batch[i, v].cpu(), cmap="coolwarm")
                plt.title(f"Target (sample {i}, {var_name})")
                plt.colorbar()
                plt.subplot(1, 2, 2)
                plt.imshow(pred_batch[i, v].cpu(), cmap="coolwarm")
                plt.title(f"Prediction (sample {i}, {var_name})")
                plt.colorbar()
                plt.suptitle(f"Test Sample {i}, {var_name}: Target vs Prediction")
                plt.savefig(os.path.join(plots_dir, f"test_sample_{i}_{var_name}_pred_vs_target.png"))
                plt.close()

        # Scatter plot for one variable (first variable, all pixels in first sample)
        plt.figure(figsize=(6, 6))
        t = target_batch[0, 0].cpu().flatten().numpy()
        p = pred_batch[0, 0].cpu().flatten().numpy()
        plt.scatter(t, p, alpha=0.3, s=5)
        plt.xlabel("Target")
        plt.ylabel("Prediction")
        plt.title(f"Scatter: Prediction vs Target (sample 0, {variable_names[0]})")
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f"scatter_sample0_{variable_names[0]}.png"))
        plt.close()

if __name__ == "__main__":
    main()
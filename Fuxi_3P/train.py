import argparse
import json
import os
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xarray as xr

from fuxi import FuXiModel  # assumes FuXiModel accepts the parsed hyper-parameters

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


def parse_int_tuple(raw: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def ensure_unique_dir(root: str, name: str) -> str:
    base = os.path.join(root, name)
    if not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
        return base
    idx = 1
    while True:
        candidate = f"{base}_run{idx:02d}"
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        idx += 1


def latitude_weighted_l1_loss(pred, target, latitudes):
    weights = torch.cos(torch.deg2rad(latitudes)).to(pred.device)
    weights = weights / weights.mean()
    weights = weights.view(1, 1, -1, 1)
    return (torch.abs(pred - target) * weights).mean()


class MiniFuXiDataset(Dataset):
    def __init__(self, path: str, history_steps: int = 2, mean=None, std=None):
        ds = xr.open_dataset(path)
        print(f"Loaded {path} variables:", list(ds.data_vars.keys()))
        ds = ds.rename({k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in ds.dims})

        pressure = (
            ds[PRESSURE_VARS]
            .sel(level=PRESSURE_LEVELS)
            .to_array()
            .transpose("time", "variable", "level", "lat", "lon")
        )
        surface = ds[SURFACE_VARS].to_array().transpose("time", "variable", "lat", "lon")

        p_np = pressure.values.reshape(pressure.shape[0], -1, pressure.shape[3], pressure.shape[4])
        s_np = surface.values
        data = torch.from_numpy(np.concatenate([p_np, s_np], axis=1)).float()

        if mean is None or std is None:
            mean = data.mean(dim=(0, 2, 3), keepdim=True)
            std = data.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

        self.data = (data - mean) / std
        self.mean = mean
        self.std = std
        self.history = history_steps

        latitudes = ds["lat"].values
        if latitudes.shape[0] != data.shape[-2]:
            latitudes = latitudes[: data.shape[-2]]
        longitudes = ds["lon"].values
        if longitudes.shape[0] != data.shape[-1]:
            longitudes = longitudes[: data.shape[-1]]

        var_names = [f"{var}_plev{lvl}" for var in PRESSURE_VARS for lvl in PRESSURE_LEVELS]
        var_names.extend(SURFACE_VARS)

        self.latitudes = latitudes
        self.longitudes = longitudes
        self.var_names = var_names
        ds.close()

    def __len__(self):
        return len(self.data) - self.history

    def __getitem__(self, idx):
        past = self.data[idx : idx + self.history]
        target = self.data[idx + self.history]
        past = past.permute(1, 0, 2, 3)
        return past, target


def train_one_epoch(model, loader, optimizer, device, latitudes):
    model.train()
    total = 0.0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = model(history, target_shape=target.shape[-2:])
        loss = latitude_weighted_l1_loss(pred, target, latitudes)
        loss.backward()
        optimizer.step()
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
    os.makedirs(os.path.join(outdir, "Plots"), exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Latitude-weighted L1 Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(outdir, "Plots/loss_curve.png"))
    plt.close()


@torch.no_grad()
def plot_prediction_maps(model, dataset, device, mean, std, outdir, sample_idx=0, var_indices=None):
    model.eval()
    history, target = dataset[sample_idx]
    history = history.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    pred = model(history, target_shape=target.shape[-2:])
    mean = mean.squeeze(0).cpu()
    std = std.squeeze(0).cpu()

    pred_denorm = pred.cpu()[0] * std + mean
    target_denorm = target.cpu()[0] * std + mean

    if var_indices is None:
        var_indices = [0, len(dataset.var_names) // 2, len(dataset.var_names) - 1]

    lon = dataset.longitudes
    lat = dataset.latitudes

    fig, axes = plt.subplots(len(var_indices), 2, figsize=(10, 4 * len(var_indices)), constrained_layout=True)
    if len(var_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, vid in enumerate(var_indices):
        vname = dataset.var_names[vid]
        tgt_map = target_denorm[vid].numpy()
        pred_map = pred_denorm[vid].numpy()

        for ax, data, title in zip(axes[row], [tgt_map, pred_map], ["Target", "Prediction"]):
            im = ax.imshow(
                data,
                origin="lower",
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                cmap="coolwarm",
            )
            ax.set_title(f"{title}: {vname}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(os.path.join(outdir, "Plots", f"prediction_maps_sample{sample_idx}.png"))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="FuXi advanced training")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (defaults to SLURM_ARRAY_TASK_ID or timestamp).")
    parser.add_argument("--models-root", type=str, default="Models", help="Root directory for experiment outputs.")
    parser.add_argument("--data-root", type=str, default="Data", help="Directory containing data files.")
    parser.add_argument("--train-file", type=str, default="train_data_1959_2017.nc")
    parser.add_argument("--val-file", type=str, default="val_data_2018_2020.nc")
    parser.add_argument("--test-file", type=str, default="test_data_2021_2023.nc")
    parser.add_argument("--history-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--betas", type=str, default="0.9,0.95")
    parser.add_argument("--seed", type=int, default=42)

    # Model hyper-parameters
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--encoder-dims", type=str, default="256,320,384")
    parser.add_argument("--swin-depths", type=str, default="4,4,8")
    parser.add_argument("--swin-heads", type=str, default="4,8,16")
    parser.add_argument("--swin-window-size", type=int, default=8)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--out-channels", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = args.exp_name or os.environ.get("SLURM_ARRAY_TASK_ID")
    if exp_name is None:
        exp_name = datetime.now().strftime("manual_%Y%m%d_%H%M%S")

    run_dir = ensure_unique_dir(args.models_root, f"exp_{exp_name}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plots_dir = os.path.join(run_dir, "Plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(args.data_root, args.train_file)
    val_path = os.path.join(args.data_root, args.val_file)
    test_path = os.path.join(args.data_root, args.test_file)

    train_set = MiniFuXiDataset(train_path, history_steps=args.history_steps)
    val_set = MiniFuXiDataset(val_path, history_steps=args.history_steps, mean=train_set.mean, std=train_set.std)
    test_set = MiniFuXiDataset(test_path, history_steps=args.history_steps, mean=train_set.mean, std=train_set.std)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    spatial_shape = tuple(train_set.data.shape[-2:])
    channels = train_set.data.shape[1]
    latitudes = torch.tensor(train_set.latitudes, dtype=torch.float32)

    in_channels = args.in_channels or channels
    out_channels = args.out_channels or channels
    input_height = args.input_height or spatial_shape[0]
    input_width = args.input_width or spatial_shape[1]

    model = FuXiModel(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=args.embed_dim,
        encoder_dims=parse_int_tuple(args.encoder_dims),
        swin_depths=parse_int_tuple(args.swin_depths),
        swin_heads=parse_int_tuple(args.swin_heads),
        swin_window_size=args.swin_window_size,
        drop_path_rate=args.drop_path_rate,
        input_height=input_height,
        input_width=input_width,
    ).to(device)

    beta1, beta2 = (float(x) for x in args.betas.split(","))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(beta1, beta2), weight_decay=args.weight_decay)

    best_val = float("inf")
    train_losses, val_losses = [], []
    epochs_no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, latitudes)
        val_l1, val_mae = eval_one_epoch(model, val_loader, device, latitudes)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_l1={val_l1:.4f} | val_mae={val_mae:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_l1)

        ckpt_path = os.path.join(ckpt_dir, f"fuxi_epoch{epoch:03d}.pt")
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

        if val_l1 < best_val:
            best_val = val_l1
            epochs_no_improve = 0
            print(f"  [Checkpoint] New best at epoch {epoch}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    plot_losses(train_losses, val_losses, run_dir)

    test_l1, test_mae = eval_one_epoch(model, test_loader, device, latitudes)
    print(f"\nTest set: l1={test_l1:.4f} | mae={test_mae:.4f}")

    plot_prediction_maps(model, test_set, device, train_set.mean, train_set.std, run_dir, sample_idx=0)

    with open(os.path.join(run_dir, "metrics.txt"), "w") as fp:
        fp.write(f"Test L1: {test_l1:.6f}\n")
        fp.write(f"Test MAE: {test_mae:.6f}\n")


if __name__ == "__main__":
    main()
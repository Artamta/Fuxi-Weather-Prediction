import argparse
import os
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import xarray as xr
import torch.distributed as dist

from fuxi import FuXiModel

PRESSURE_LEVELS = [250, 500, 850]
PRESSURE_VARS = ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind", "geopotential"]
SURFACE_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "surface_pressure"]


def latitude_weighted_l1_loss(pred, target, latitudes):
    weights = torch.cos(torch.deg2rad(latitudes)).to(pred.device)
    weights = weights / weights.mean()
    weights = weights.view(1, 1, -1, 1)
    return (torch.abs(pred - target) * weights).mean()


class MiniFuXiDataset(Dataset):
    def __init__(self, path: str, history_steps: int = 2, mean=None, std=None):
        ds = xr.open_dataset(path)
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
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(history, target_shape=target.shape[-2:])
        loss = latitude_weighted_l1_loss(pred, target, latitudes)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device, latitudes):
    model.eval()
    total_l1, total_mae, batches = 0.0, 0.0, 0
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(history, target_shape=target.shape[-2:])
        l1 = latitude_weighted_l1_loss(pred, target, latitudes)
        mae = torch.mean(torch.abs(pred - target))
        total_l1 += l1.item()
        total_mae += mae.item()
        batches += 1
    batches = max(1, batches)
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
    plt.savefig(os.path.join(outdir, "Plots", "loss_curve.png"))
    plt.close()


@torch.no_grad()
def plot_prediction_maps(model, dataset, device, mean, std, outdir, sample_idx=0, var_indices=None):
    net = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    net.eval()

    history, target = dataset[sample_idx]
    history = history.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    pred = net(history, target_shape=target.shape[-2:])
    mean = mean.squeeze(0).cpu()
    std = std.squeeze(0).cpu()

    pred_denorm = pred.cpu()[0] * std + mean
    target_denorm = target.cpu()[0] * std + mean

    if var_indices is None:
        var_indices = [0, len(dataset.var_names) // 2, len(dataset.var_names) - 1]

    lon = dataset.longitudes
    lat = dataset.latitudes

    os.makedirs(os.path.join(outdir, "Plots"), exist_ok=True)
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
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/home/raj.ayush/fuxi_advanced/data")
    p.add_argument("--train-file", type=str, default="train_data_tiny.nc")
    p.add_argument("--val-file", type=str, default="val_data_tiny.nc")
    p.add_argument("--test-file", type=str, default="test_data_tiny.nc")
    p.add_argument("--history-steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--betas", type=str, default="0.9,0.95")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--encoder-dims", type=str, default="256,320,384")
    p.add_argument("--swin-depths", type=str, default="4,4,8")
    p.add_argument("--swin-heads", type=str, default="4,8,16")
    p.add_argument("--swin-window-size", type=int, default=8)
    p.add_argument("--drop-path-rate", type=float, default=0.1)
    p.add_argument("--runs-dir", type=str, default="Models_simple")
    p.add_argument("--exp-name", type=str, default=None)
    return p.parse_args()


def setup_dist():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size()
    return None, 1


# ...existing imports...
def main():
    args = parse_args()

    local_rank, world_size = setup_dist()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"torch sees {torch.cuda.device_count()} CUDA device(s)")
    device = torch.device(f"cuda:{local_rank}" if use_cuda and local_rank is not None else ("cuda" if use_cuda else "cpu"))
    if local_rank in (0, None):
        print(f"Using device: {device} (GPUs: {torch.cuda.device_count()})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = args.exp_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, exp_name)
    if local_rank in (0, None):
        os.makedirs(run_dir, exist_ok=True)
        print(f"[run_dir] {run_dir}")
    ckpt_path = os.path.join(run_dir, "best.pt")

    train_set = MiniFuXiDataset(os.path.join(args.data_root, args.train_file), history_steps=args.history_steps)
    val_set = MiniFuXiDataset(os.path.join(args.data_root, args.val_file), history_steps=args.history_steps, mean=train_set.mean, std=train_set.std)
    test_set = MiniFuXiDataset(os.path.join(args.data_root, args.test_file), history_steps=args.history_steps, mean=train_set.mean, std=train_set.std)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=local_rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=local_rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, num_workers=args.num_workers, pin_memory=use_cuda)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=test_sampler, num_workers=args.num_workers, pin_memory=use_cuda)

    spatial_shape = tuple(train_set.data.shape[-2:])
    channels = train_set.data.shape[1]
    latitudes = torch.tensor(train_set.latitudes, dtype=torch.float32, device=device)

    encoder_dims = tuple(int(x) for x in args.encoder_dims.split(",") if x.strip())
    swin_depths = tuple(int(x) for x in args.swin_depths.split(",") if x.strip())
    swin_heads = tuple(int(x) for x in args.swin_heads.split(",") if x.strip())
    beta1, beta2 = (float(x) for x in args.betas.split(","))

    model = FuXiModel(
        in_channels=channels,
        out_channels=channels,
        embed_dim=args.embed_dim,
        encoder_dims=encoder_dims,
        swin_depths=swin_depths,
        swin_heads=swin_heads,
        swin_window_size=args.swin_window_size,
        drop_path_rate=args.drop_path_rate,
        input_height=spatial_shape[0],
        input_width=spatial_shape[1],
    ).to(device)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(beta1, beta2), weight_decay=args.weight_decay)

    best_val = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, args.max_epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        if local_rank in (0, None):
            print(f"\n=== Epoch {epoch} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, latitudes)
        val_l1, val_mae = eval_one_epoch(model, val_loader, device, latitudes)
        if local_rank in (0, None):
            print(f"train_loss={train_loss:.4f} | val_l1={val_l1:.4f} | val_mae={val_mae:.4f}")

            train_losses.append(train_loss)
            val_losses.append(val_l1)

            if val_l1 < best_val:
                best_val = val_l1
                epochs_no_improve = 0
                state = {
                    "epoch": epoch,
                    "model_state": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_l1": val_l1,
                    "val_mae": val_mae,
                }
                torch.save(state, ckpt_path)
                print(f"  [Saved best] {ckpt_path}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epochs.")
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    if local_rank in (0, None):
        plot_losses(train_losses, val_losses, run_dir)

        if not os.path.exists(ckpt_path):
            state = {
                "epoch": epoch,
                "model_state": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_l1": val_l1 if "val_l1" in locals() else None,
                "val_mae": val_mae if "val_mae" in locals() else None,
            }
            torch.save(state, ckpt_path)
            print(f"  [Saved fallback] {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        model_state = checkpoint["model_state"]
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        test_l1, test_mae = eval_one_epoch(model, test_loader, device, latitudes)
        print(f"\nTest set: l1={test_l1:.4f} | mae={test_mae:.4f}")

        os.makedirs(os.path.join(run_dir, "Plots"), exist_ok=True)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(
                {
                    "best_val_l1": best_val,
                    "test_l1": test_l1,
                    "test_mae": test_mae,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
                f,
                indent=2,
            )

        plot_prediction_maps(model, test_set, device, train_set.mean, train_set.std, run_dir, sample_idx=0)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
import argparse
import os
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import xarray as xr
import zarr
import torch.distributed as dist

from fuxi import FuXiModel

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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def latitude_weighted_l1_loss(pred, target, latitudes):
    weights = torch.cos(torch.deg2rad(latitudes)).to(pred.device)
    weights = weights / weights.mean()
    weights = weights.view(1, 1, -1, 1)
    return (torch.abs(pred - target) * weights).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MiniFuXiDataset(Dataset):
    """Lazy-loading dataset that reads from .nc files or .zarr stores.

    For Zarr stores, individual samples are loaded via the zarr library
    directly (bypassing dask), so init is nearly instant and RAM usage
    is minimal.  For small .nc files, data is preloaded for speed.
    """

    def __init__(self, path: str, history_steps: int = 2, mean=None, std=None,
                 time_start=None, time_end=None, stats_subsample: int = 200):
        self.history = history_steps
        self._lazy = path.endswith(".zarr")

        if self._lazy:
            self._init_zarr(path, time_start, time_end, mean, std, stats_subsample)
        else:
            self._init_nc(path, time_start, time_end, mean, std)

    # ---------------------------------------------------------------
    # Zarr lazy init  (no dask, nearly instant)
    # ---------------------------------------------------------------
    def _init_zarr(self, path, time_start, time_end, mean, std, stats_subsample):
        # Open raw zarr — no xarray, no dask
        store = zarr.open_group(path, mode="r")

        # Decode time coordinate from CF conventions (e.g. "hours since 1959-01-01")
        raw_time = store["time"][:]  # integer array
        time_attrs = dict(store["time"].attrs)
        units = time_attrs.get("units", "hours since 1959-01-01")
        # Parse "hours since 1959-01-01"
        parts = units.split(" since ")
        delta_unit = parts[0].strip().rstrip("s")  # "hour"
        base_date = np.datetime64(parts[1].strip())
        all_times = base_date + raw_time.astype("timedelta64[{}]".format(
            {"hour": "h", "minute": "m", "second": "s", "day": "D"}[delta_unit]
        ))

        # Time slicing
        mask = np.ones(len(all_times), dtype=bool)
        if time_start:
            mask &= all_times >= np.datetime64(time_start)
        if time_end:
            mask &= all_times <= np.datetime64(time_end)
        self._time_indices = np.where(mask)[0]
        self.n_times = len(self._time_indices)

        # Spatial coordinates
        self.latitudes = store["latitude"][:] if "latitude" in store else store["lat"][:]
        self.longitudes = store["longitude"][:] if "longitude" in store else store["lon"][:]
        # NOTE: zarr stores lat as (lat,) and lon as (lon,)

        # Channel info
        n_plev = len(PRESSURE_VARS) * len(PRESSURE_LEVELS)
        n_surf = len(SURFACE_VARS)
        self.channels = n_plev + n_surf
        self.spatial_shape = (len(self.latitudes), len(self.longitudes))
        self.var_names = [
            f"{var}_plev{lvl}" for var in PRESSURE_VARS for lvl in PRESSURE_LEVELS
        ] + list(SURFACE_VARS)

        # Level dimension — find indices for our pressure levels
        level_arr = store["level"][:]
        self._level_idxs = [int(np.argwhere(level_arr == lv).item()) for lv in PRESSURE_LEVELS]

        # Keep zarr array references for each variable
        self._p_arrays = [store[v] for v in PRESSURE_VARS]   # shape: (T, level, lat, lon)
        self._s_arrays = [store[v] for v in SURFACE_VARS]    # shape: (T, lat, lon)
        self._store = store  # keep reference open

        # Detect axis order: (T, level, lat, lon) or (T, level, lon, lat)?
        # WeatherBench2 uses (T, level, lon, lat) i.e. shape[2]=lon, shape[3]=lat
        pshape = self._p_arrays[0].shape
        if pshape[2] == len(self.longitudes) and pshape[3] == len(self.latitudes):
            self._transpose_spatial = True  # need to swap last two axes
        else:
            self._transpose_spatial = False

        # Compute or reuse normalization stats
        if mean is None or std is None:
            mean, std = self._compute_stats_zarr(stats_subsample)
        self.mean = mean
        self.std = std
        self.data = None

    def _compute_stats_zarr(self, n_samples: int = 200):
        """Channel-wise mean/std from a contiguous block, via raw zarr reads."""
        mid = self.n_times // 2
        half = min(n_samples, self.n_times) // 2
        t_slice = self._time_indices[mid - half : mid + half]  # global time indices
        t_start, t_end = int(t_slice[0]), int(t_slice[-1]) + 1

        chunks = []
        for arr in self._p_arrays:
            sub = arr[t_start:t_end]  # (n, all_levels, ?, ?)
            sub = sub[:, self._level_idxs, :, :]  # (n, 3, ?, ?)
            if self._transpose_spatial:
                sub = np.swapaxes(sub, -2, -1)  # → (n, 3, lat, lon)
            chunks.append(sub)
        p_all = np.concatenate(chunks, axis=1)  # (n, 15, lat, lon)

        s_chunks = []
        for arr in self._s_arrays:
            s = arr[t_start:t_end]  # (n, ?, ?)
            if self._transpose_spatial:
                s = np.swapaxes(s, -2, -1)
            s_chunks.append(s)
        s_all = np.stack(s_chunks, axis=1)  # (n, 5, lat, lon)

        data = torch.from_numpy(
            np.concatenate([p_all, s_all], axis=1).astype(np.float32)
        )  # (n, 20, lat, lon)
        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
        return mean, std

    def _load_sample_zarr(self, idx):
        """Load ONE timestep directly from zarr arrays (fast, no dask)."""
        t = int(self._time_indices[idx])

        parts = []
        for arr in self._p_arrays:
            sub = arr[t]  # (all_levels, lon, lat) or (all_levels, lat, lon)
            sub = sub[self._level_idxs]  # (3, ?, ?)
            if self._transpose_spatial:
                sub = np.swapaxes(sub, -2, -1)  # → (3, lat, lon)
            parts.append(sub)
        for arr in self._s_arrays:
            s = arr[t]  # (lon, lat) or (lat, lon)
            if self._transpose_spatial:
                s = np.swapaxes(s, -2, -1)
            parts.append(s[np.newaxis])  # (1, lat, lon)

        combined = np.concatenate(parts, axis=0).astype(np.float32)  # (C, H, W)
        sample = torch.from_numpy(combined)
        return (sample - self.mean.squeeze(0)) / self.std.squeeze(0)

    # ---------------------------------------------------------------
    # NetCDF eager init  (small files, preloads into RAM)
    # ---------------------------------------------------------------
    def _init_nc(self, path, time_start, time_end, mean, std):
        ds = xr.open_dataset(path)
        ds = ds.rename(
            {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in ds.dims}
        )
        if time_start or time_end:
            ds = ds.sel(time=slice(time_start, time_end))

        pressure = (
            ds[PRESSURE_VARS]
            .sel(level=PRESSURE_LEVELS)
            .to_array()
            .transpose("time", "variable", "level", "lat", "lon")
        )
        surface = ds[SURFACE_VARS].to_array().transpose("time", "variable", "lat", "lon")

        p_np = pressure.values.reshape(
            pressure.shape[0], -1, pressure.shape[3], pressure.shape[4]
        )
        s_np = surface.values
        data = torch.from_numpy(np.concatenate([p_np, s_np], axis=1)).float()

        if mean is None or std is None:
            mean = data.mean(dim=(0, 2, 3), keepdim=True)
            std = data.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

        self.data = (data - mean) / std
        self.mean = mean
        self.std = std

        self.latitudes = ds["lat"].values
        self.longitudes = ds["lon"].values
        self.n_times = data.shape[0]
        self.channels = data.shape[1]
        self.spatial_shape = tuple(data.shape[-2:])
        self.var_names = [
            f"{var}_plev{lvl}" for var in PRESSURE_VARS for lvl in PRESSURE_LEVELS
        ] + list(SURFACE_VARS)
        ds.close()

    # ---------------------------------------------------------------
    # Common interface
    # ---------------------------------------------------------------
    def __len__(self):
        return self.n_times - self.history

    def __getitem__(self, idx):
        if self._lazy:
            frames = [self._load_sample_zarr(idx + t) for t in range(self.history)]
            past = torch.stack(frames, dim=1)  # (C, T, H, W)
            target = self._load_sample_zarr(idx + self.history)  # (C, H, W)
        else:
            past = self.data[idx : idx + self.history]
            target = self.data[idx + self.history]
            past = past.permute(1, 0, 2, 3)  # (C, T, H, W)
        return past, target


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, latitudes, use_amp,
                    grad_clip, accum_steps=1):
    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, (history, target) in enumerate(loader):
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pred = model(history, target_shape=target.shape[-2:])
            loss = latitude_weighted_l1_loss(pred, target, latitudes)
            loss = loss / accum_steps  # scale for accumulation

        scaler.scale(loss).backward()
        total += loss.item() * accum_steps  # un-scale for logging

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return total / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device, latitudes, use_amp):
    model.eval()
    total_l1, total_mae, batches = 0.0, 0.0, 0
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            pred = model(history, target_shape=target.shape[-2:])
            l1 = latitude_weighted_l1_loss(pred, target, latitudes)
            mae = torch.mean(torch.abs(pred - target))
        total_l1 += l1.item()
        total_mae += mae.item()
        batches += 1
    batches = max(1, batches)
    return total_l1 / batches, total_mae / batches


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_losses(train_losses, val_losses, outdir):
    os.makedirs(os.path.join(outdir, "Plots"), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Latitude-weighted L1 Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(outdir, "Plots", "loss_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


@torch.no_grad()
def plot_prediction_maps(model, dataset, device, mean, std, outdir, use_amp,
                         sample_idx=0, var_indices=None):
    net = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model
    net.eval()

    history, target = dataset[sample_idx]
    history = history.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    with autocast(enabled=use_amp):
        pred = net(history, target_shape=target.shape[-2:])

    mean_cpu = mean.squeeze(0).cpu()
    std_cpu = std.squeeze(0).cpu()
    pred_denorm = pred.float().cpu()[0] * std_cpu + mean_cpu
    target_denorm = target.float().cpu()[0] * std_cpu + mean_cpu

    if var_indices is None:
        nv = len(dataset.var_names)
        var_indices = [0, nv // 2, nv - 1]

    lon = dataset.longitudes
    lat = dataset.latitudes

    os.makedirs(os.path.join(outdir, "Plots"), exist_ok=True)
    fig, axes = plt.subplots(
        len(var_indices), 2,
        figsize=(10, 4 * len(var_indices)),
        constrained_layout=True,
    )
    if len(var_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, vid in enumerate(var_indices):
        vname = dataset.var_names[vid]
        tgt_map = target_denorm[vid].numpy()
        pred_map = pred_denorm[vid].numpy()
        for ax, data, title in zip(
            axes[row], [tgt_map, pred_map], ["Target", "Prediction"]
        ):
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

    save_path = os.path.join(outdir, "Plots", f"prediction_maps_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="FuXi DDP Training")
    p.add_argument("--data-root", type=str, default="/home/raj.ayush/fuxi_advanced/data")
    p.add_argument("--train-file", type=str, default="train_data.nc")
    p.add_argument("--val-file", type=str, default="val_data.nc")
    p.add_argument("--test-file", type=str, default="test_data.nc")
    # Zarr direct-read mode (overrides --data-root / --*-file)
    p.add_argument("--zarr-store", type=str, default=None,
                   help="Path to Zarr store; reads directly, no copy needed")
    p.add_argument("--train-start", type=str, default="1979-01-01")
    p.add_argument("--train-end",   type=str, default="2015-12-31")
    p.add_argument("--val-start",   type=str, default="2016-01-01")
    p.add_argument("--val-end",     type=str, default="2017-12-31")
    p.add_argument("--test-start",  type=str, default="2018-01-01")
    p.add_argument("--test-end",    type=str, default="2018-12-31")
    p.add_argument("--history-steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--betas", type=str, default="0.9,0.95")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch_size * accum * GPUs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--encoder-dims", type=str, default="256,320,384")
    p.add_argument("--swin-depths", type=str, default="4,4,8")
    p.add_argument("--swin-heads", type=str, default="4,8,16")
    p.add_argument("--swin-window-size", type=int, default=8)
    p.add_argument("--drop-path-rate", type=float, default=0.2)
    p.add_argument("--runs-dir", type=str, default="Models")
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DDP setup / teardown
# ---------------------------------------------------------------------------
def setup_dist():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), dist.get_rank()
    return None, 1, 0


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank in (0, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    local_rank, world_size, global_rank = setup_dist()
    use_cuda = torch.cuda.is_available()
    use_amp = use_cuda
    device = torch.device(
        f"cuda:{local_rank}" if use_cuda and local_rank is not None
        else ("cuda" if use_cuda else "cpu")
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # ---- Run directory ---------------------------------------------------
    exp_name = args.exp_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, exp_name)
    if is_main(global_rank):
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "Plots"), exist_ok=True)
        print(f"{'=' * 60}")
        print(f"Experiment : {exp_name}")
        print(f"Run dir    : {run_dir}")
        print(f"Device     : {device} | World size: {world_size}")
        if use_cuda:
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        print(f"{'=' * 60}")

    # Wait for rank 0 to create dirs
    if world_size > 1:
        dist.barrier()

    ckpt_path = os.path.join(run_dir, "best.pt")
    last_ckpt_path = os.path.join(run_dir, "last.pt")

    # ---- Data ------------------------------------------------------------
    if is_main(global_rank):
        print("Loading datasets...")
    t0 = time.time()

    if args.zarr_store:
        # Read directly from Zarr — no storage copy needed
        if is_main(global_rank):
            print(f"  Zarr mode: {args.zarr_store}")
            print(f"  Train: {args.train_start} → {args.train_end}")
            print(f"  Val  : {args.val_start} → {args.val_end}")
            print(f"  Test : {args.test_start} → {args.test_end}")
        train_set = MiniFuXiDataset(
            args.zarr_store, history_steps=args.history_steps,
            time_start=args.train_start, time_end=args.train_end,
        )
        val_set = MiniFuXiDataset(
            args.zarr_store, history_steps=args.history_steps,
            mean=train_set.mean, std=train_set.std,
            time_start=args.val_start, time_end=args.val_end,
        )
        test_set = MiniFuXiDataset(
            args.zarr_store, history_steps=args.history_steps,
            mean=train_set.mean, std=train_set.std,
            time_start=args.test_start, time_end=args.test_end,
        )
    else:
        # Classic NetCDF file mode
        train_set = MiniFuXiDataset(
            os.path.join(args.data_root, args.train_file),
            history_steps=args.history_steps,
        )
        val_set = MiniFuXiDataset(
            os.path.join(args.data_root, args.val_file),
            history_steps=args.history_steps,
            mean=train_set.mean, std=train_set.std,
        )
        test_set = MiniFuXiDataset(
            os.path.join(args.data_root, args.test_file),
            history_steps=args.history_steps,
            mean=train_set.mean, std=train_set.std,
        )

    if is_main(global_rank):
        print(
            f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}"
            f"  (loaded in {time.time() - t0:.1f}s)"
        )
        print(f"  Channels: {train_set.channels} | Spatial: {train_set.spatial_shape}")

    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if world_size > 1 else None

    # For lazy Zarr, use fewer workers (I/O bound, not CPU bound)
    nw = min(args.num_workers, 2) if train_set._lazy else args.num_workers

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=nw, pin_memory=use_cuda, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=nw, pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        sampler=test_sampler, num_workers=nw, pin_memory=use_cuda,
    )

    # ---- Model -----------------------------------------------------------
    spatial_shape = train_set.spatial_shape
    channels = train_set.channels
    latitudes = torch.tensor(train_set.latitudes, dtype=torch.float32, device=device)

    encoder_dims = tuple(int(x) for x in args.encoder_dims.split(",") if x.strip())
    swin_depths = tuple(int(x) for x in args.swin_depths.split(",") if x.strip())
    swin_heads = tuple(int(x) for x in args.swin_heads.split(",") if x.strip())

    if not (len(encoder_dims) == len(swin_depths) == len(swin_heads)):
        raise ValueError(
            f"encoder_dims ({len(encoder_dims)}), swin_depths ({len(swin_depths)}), "
            f"swin_heads ({len(swin_heads)}) must have the same length"
        )

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

    if is_main(global_rank):
        total_p = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model params: {total_p:.2f}M")

    # ---- Resume ----------------------------------------------------------
    start_epoch = 1
    best_val = float("inf")

    if args.resume and os.path.isfile(args.resume):
        if is_main(global_rank):
            print(f"  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_l1", float("inf"))
        if is_main(global_rank):
            print(f"  Resumed epoch {ckpt.get('epoch')} (val_l1={ckpt.get('val_l1', '?'):.4f})")

    # ---- DDP wrap --------------------------------------------------------
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
        if is_main(global_rank):
            print(f"  Wrapped in DDP on {world_size} GPUs")

    # ---- Optimizer / scheduler / scaler ----------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(beta1, beta2), weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01,
    )
    scaler = GradScaler(enabled=use_amp)

    # Restore optimizer if resuming
    if args.resume and os.path.isfile(args.resume):
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    # ---- Save config -----------------------------------------------------
    if is_main(global_rank):
        config = vars(args).copy()
        config["world_size"] = world_size
        config["channels"] = channels
        config["spatial_shape"] = list(spatial_shape)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # ---- Training loop ---------------------------------------------------
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    if is_main(global_rank):
        print(f"\n{'=' * 60}")
        print("Starting training...")
        print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_start = time.time()

        if world_size > 1:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            device, latitudes, use_amp, args.grad_clip,
            accum_steps=args.accum_steps,
        )
        val_l1, val_mae = eval_one_epoch(
            model, val_loader, device, latitudes, use_amp,
        )
        scheduler.step()

        if is_main(global_rank):
            epoch_time = time.time() - epoch_start
            lr_now = optimizer.param_groups[0]["lr"]
            train_losses.append(train_loss)
            val_losses.append(val_l1)

            print(
                f"Epoch {epoch:3d}/{args.max_epochs} | "
                f"train={train_loss:.4f} | val_l1={val_l1:.4f} | val_mae={val_mae:.4f} | "
                f"lr={lr_now:.2e} | time={epoch_time:.1f}s"
            )

            # Best checkpoint
            if val_l1 < best_val:
                best_val = val_l1
                epochs_no_improve = 0
                state = {
                    "epoch": epoch,
                    "model_state": (
                        model.module.state_dict()
                        if isinstance(model, nn.parallel.DistributedDataParallel)
                        else model.state_dict()
                    ),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_l1": val_l1,
                    "val_mae": val_mae,
                }
                torch.save(state, ckpt_path)
                print(f"  ✓ Saved best → {ckpt_path}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve} epochs")
                if epochs_no_improve >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            # Save last checkpoint EVERY epoch (crash recovery for 7-day SLURM limit)
            if True:
                last_state = {
                    "epoch": epoch,
                    "model_state": (
                        model.module.state_dict()
                        if isinstance(model, nn.parallel.DistributedDataParallel)
                        else model.state_dict()
                    ),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_l1": val_l1,
                    "val_mae": val_mae,
                }
                torch.save(last_state, last_ckpt_path)
                print(f"  Saved last → {last_ckpt_path}")

    # ---- Post-training (rank 0 only) ------------------------------------
    # Unwrap the model so DDP collectives don't hang rank 1
    net = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    if is_main(global_rank):
        print(f"\n{'=' * 60}")
        print("Training complete. Evaluating on test set...")
        print(f"{'=' * 60}")

        # Load best
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(checkpoint["model_state"])

        test_l1, test_mae = eval_one_epoch(
            net, test_loader, device, latitudes, use_amp,
        )
        print(f"\nTest results: l1={test_l1:.4f} | mae={test_mae:.4f}")

        # Save metrics
        metrics = {
            "best_epoch": checkpoint.get("epoch", "?") if os.path.isfile(ckpt_path) else "?",
            "best_val_l1": best_val,
            "test_l1": test_l1,
            "test_mae": test_mae,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Plots
        plot_losses(train_losses, val_losses, run_dir)
        plot_prediction_maps(
            net, test_set, device,
            train_set.mean, train_set.std, run_dir, use_amp,
            sample_idx=0,
        )
        print(f"\nAll outputs saved to: {run_dir}")
        print("Done.")

    # All ranks sync before exit
    if world_size > 1:
        dist.barrier()

    cleanup_dist()


if __name__ == "__main__":
    main()
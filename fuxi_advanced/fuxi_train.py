#!/usr/bin/env python3
"""
FuXi Pre-training Script - Paper-Faithful Implementation

Supports single-step prediction pre-training with:
- Direct Zarr store reading (zero storage copy)
- Multi-GPU training (DDP)
- Gradient accumulation
- Mixed precision training
- Automatic checkpointing and resume
- TensorBoard logging

Usage:
------
# Single GPU:
python fuxi_train.py --zarr-store /path/to/data.zarr \
    --train-start 1979-01-01 --train-end 2015-12-31 \
    --val-start 2016-01-01 --val-end 2018-12-31

# Multi-GPU (DDP):
torchrun --nproc_per_node=4 fuxi_train.py --zarr-store /path/to/data.zarr \
    --train-start 1979-01-01 --train-end 2015-12-31 \
    --batch-size 4 --accum-steps 2

# Resume from checkpoint:
python fuxi_train.py --zarr-store /path/to/data.zarr \
    --resume Models/exp_name/last.pt
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import zarr
import torch.distributed as dist

# Import your model - adjust these imports to match your file structure
try:
    from model import FuXi  # or FuXiModel, adjust as needed
except ImportError:
    print("ERROR: Could not import model. Please ensure model.py is in the same directory")
    print("or adjust the import statement to match your model file structure.")
    raise


# =============================================================================
# Configuration
# =============================================================================

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


# =============================================================================
# Loss Function
# =============================================================================

class LatitudeWeightedL1Loss(nn.Module):
    """
    Latitude-weighted L1 loss (paper-faithful).
    
    Weights by cos(latitude) to account for grid cell area variation.
    """
    def __init__(self, num_lat: int, lat_range: Tuple[float, float] = (-90, 90)):
        super().__init__()
        lats = torch.linspace(lat_range[0], lat_range[1], num_lat)
        weights = torch.cos(torch.deg2rad(lats))
        weights = weights / weights.mean()
        self.register_buffer("weights", weights.view(1, 1, -1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
        Returns:
            Scalar loss
        """
        return (torch.abs(pred - target) * self.weights).mean()


def latitude_weighted_l1_loss(pred, target, latitudes):
    """Functional version for quick use."""
    weights = torch.cos(torch.deg2rad(latitudes)).to(pred.device)
    weights = weights / weights.mean()
    weights = weights.view(1, 1, -1, 1)
    return (torch.abs(pred - target) * weights).mean()


# =============================================================================
# Dataset - Lazy Zarr Loading
# =============================================================================

class FuXiZarrDataset(Dataset):
    """
    Lazy-loading dataset for FuXi training.
    
    Reads directly from Zarr store with zero data copying.
    Each sample is loaded on-demand via direct zarr array access.
    
    Features:
    - Instant initialization (no data loading)
    - Minimal RAM usage
    - Supports time-based slicing
    - Automatic normalization
    """
    
    def __init__(
        self,
        zarr_path: str,
        history_steps: int = 2,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        stats_subsample: int = 200,
    ):
        """
        Args:
            zarr_path: Path to Zarr store
            history_steps: Number of historical timesteps (typically 2)
            time_start: Start date (e.g., "1979-01-01")
            time_end: End date (e.g., "2015-12-31")
            mean: Pre-computed mean (if None, compute from data)
            std: Pre-computed std (if None, compute from data)
            stats_subsample: Number of samples for computing stats
        """
        self.history = history_steps
        self.zarr_path = zarr_path
        
        # Open zarr store (no xarray, no dask - instant!)
        self.store = zarr.open_group(zarr_path, mode="r")
        
        # Decode time coordinate
        self._decode_time_coordinate(time_start, time_end)
        
        # Load spatial coordinates
        self.latitudes = self._get_coord("latitude", "lat")
        self.longitudes = self._get_coord("longitude", "lon")
        
        # Setup channel info
        self.n_plev = len(PRESSURE_VARS) * len(PRESSURE_LEVELS)
        self.n_surf = len(SURFACE_VARS)
        self.channels = self.n_plev + self.n_surf
        self.spatial_shape = (len(self.latitudes), len(self.longitudes))
        
        self.var_names = [
            f"{var}_plev{lvl}" for var in PRESSURE_VARS for lvl in PRESSURE_LEVELS
        ] + list(SURFACE_VARS)
        
        # Get level indices for pressure levels
        level_arr = self.store["level"][:]
        self._level_idxs = [
            int(np.argwhere(level_arr == lv).item()) for lv in PRESSURE_LEVELS
        ]
        
        # Store zarr array references
        self._p_arrays = [self.store[v] for v in PRESSURE_VARS]
        self._s_arrays = [self.store[v] for v in SURFACE_VARS]
        
        # Detect axis order: (T, level, lat, lon) or (T, level, lon, lat)?
        pshape = self._p_arrays[0].shape
        if pshape[2] == len(self.longitudes) and pshape[3] == len(self.latitudes):
            self._transpose_spatial = True
        else:
            self._transpose_spatial = False
        
        # Compute or use provided normalization stats
        if mean is None or std is None:
            self.mean, self.std = self._compute_stats(stats_subsample)
        else:
            self.mean = mean
            self.std = std
    
    def _get_coord(self, name1: str, name2: str) -> np.ndarray:
        """Get coordinate array, trying both possible names."""
        if name1 in self.store:
            return self.store[name1][:]
        elif name2 in self.store:
            return self.store[name2][:]
        else:
            raise KeyError(f"Neither '{name1}' nor '{name2}' found in zarr store")
    
    def _decode_time_coordinate(self, time_start: Optional[str], time_end: Optional[str]):
        """Decode CF-convention time and apply slicing."""
        raw_time = self.store["time"][:]
        time_attrs = dict(self.store["time"].attrs)
        units = time_attrs.get("units", "hours since 1959-01-01")
        
        # Parse "hours since 1959-01-01"
        parts = units.split(" since ")
        delta_unit = parts[0].strip().rstrip("s")
        base_date = np.datetime64(parts[1].strip())
        
        # Convert to datetime64
        unit_map = {"hour": "h", "minute": "m", "second": "s", "day": "D"}
        all_times = base_date + raw_time.astype(f"timedelta64[{unit_map[delta_unit]}]")
        
        # Apply time slicing
        mask = np.ones(len(all_times), dtype=bool)
        if time_start:
            mask &= all_times >= np.datetime64(time_start)
        if time_end:
            mask &= all_times <= np.datetime64(time_end)
        
        self._time_indices = np.where(mask)[0]
        self.n_times = len(self._time_indices)
        
        if self.n_times == 0:
            raise ValueError(f"No data found in time range {time_start} to {time_end}")
    
    def _compute_stats(self, n_samples: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute channel-wise mean and std from a contiguous block.
        
        Uses direct zarr reads for efficiency.
        """
        mid = self.n_times // 2
        half = min(n_samples, self.n_times) // 2
        t_slice = self._time_indices[mid - half : mid + half]
        t_start, t_end = int(t_slice[0]), int(t_slice[-1]) + 1
        
        # Load pressure data
        p_chunks = []
        for arr in self._p_arrays:
            sub = arr[t_start:t_end]  # (n, all_levels, ?, ?)
            sub = sub[:, self._level_idxs, :, :]  # (n, 3, ?, ?)
            if self._transpose_spatial:
                sub = np.swapaxes(sub, -2, -1)
            p_chunks.append(sub)
        p_all = np.concatenate(p_chunks, axis=1)  # (n, 15, lat, lon)
        
        # Load surface data
        s_chunks = []
        for arr in self._s_arrays:
            s = arr[t_start:t_end]
            if self._transpose_spatial:
                s = np.swapaxes(s, -2, -1)
            s_chunks.append(s)
        s_all = np.stack(s_chunks, axis=1)  # (n, 5, lat, lon)
        
        # Combine and compute stats
        data = torch.from_numpy(
            np.concatenate([p_all, s_all], axis=1).astype(np.float32)
        )
        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
        
        return mean, std
    
    def _load_sample(self, idx: int) -> torch.Tensor:
        """Load a single timestep from zarr arrays."""
        t = int(self._time_indices[idx])
        
        parts = []
        # Pressure levels
        for arr in self._p_arrays:
            sub = arr[t][self._level_idxs]  # (3, ?, ?)
            if self._transpose_spatial:
                sub = np.swapaxes(sub, -2, -1)
            parts.append(sub)
        
        # Surface variables
        for arr in self._s_arrays:
            s = arr[t]
            if self._transpose_spatial:
                s = np.swapaxes(s, -2, -1)
            parts.append(s[np.newaxis])
        
        combined = np.concatenate(parts, axis=0).astype(np.float32)
        sample = torch.from_numpy(combined)
        
        # Normalize
        return (sample - self.mean.squeeze(0)) / self.std.squeeze(0)
    
    def __len__(self) -> int:
        return self.n_times - self.history
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            history: (C, T, H, W) - past frames
            target: (C, H, W) - next frame to predict
        """
        # Load history frames
        frames = [self._load_sample(idx + t) for t in range(self.history)]
        history = torch.stack(frames, dim=1)  # (C, T, H, W)
        
        # Load target frame
        target = self._load_sample(idx + self.history)
        
        return history, target


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float = 1.0,
    accum_steps: int = 1,
    max_iters: Optional[int] = None,
    global_step: int = 0,
) -> Tuple[float, int]:
    """
    Single epoch of pre-training (single-step prediction).
    
    Args:
        model: FuXi model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
        grad_clip: Gradient clipping value
        accum_steps: Gradient accumulation steps
        max_iters: Maximum iterations (for paper's 40k limit)
        global_step: Current global step counter
    
    Returns:
        avg_loss: Average loss for the epoch
        global_step: Updated global step counter
    """
    model.train()
    total_loss = 0.0
    count = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, (history, target) in enumerate(loader):
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            # Forward pass
            pred = model(history)
            loss = criterion(pred, target)
            loss = loss / accum_steps  # Scale for accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accum_steps
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accum_steps
        count += 1
        global_step += 1
        
        # Check iteration limit (paper: 40k for pre-training)
        if max_iters is not None and global_step >= max_iters:
            break
    
    return total_loss / max(count, 1), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """
    Evaluate on validation/test set.
    
    Returns:
        avg_loss: Average loss
        avg_mae: Average mean absolute error
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            pred = model(history)
            loss = criterion(pred, target)
        
        total_loss += loss.item()
        total_mae += torch.abs(pred.float() - target.float()).mean().item()
        count += 1
    
    return total_loss / max(count, 1), total_mae / max(count, 1)


# =============================================================================
# Visualization
# =============================================================================

def plot_losses(train_losses, val_losses, outdir):
    """Plot training curves."""
    os.makedirs(os.path.join(outdir, "Plots"), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Latitude-weighted L1 Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.title("FuXi Pre-training: Loss Curves", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(outdir, "Plots", "loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


@torch.no_grad()
def plot_prediction_maps(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    outdir: str,
    use_amp: bool,
    sample_idx: int = 0,
    var_indices: Optional[list] = None,
):
    """Plot prediction vs target for selected variables."""
    # Unwrap DDP if needed
    net = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    net.eval()
    
    history, target = dataset[sample_idx]
    history = history.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    
    with autocast(enabled=use_amp):
        pred = net(history)
    
    # Denormalize
    mean_cpu = mean.squeeze(0).cpu()
    std_cpu = std.squeeze(0).cpu()
    pred_denorm = pred.float().cpu()[0] * std_cpu + mean_cpu
    target_denorm = target.float().cpu()[0] * std_cpu + mean_cpu
    
    # Select variables to plot
    if var_indices is None:
        nv = len(dataset.var_names)
        var_indices = [0, nv // 2, nv - 1]
    
    lon = dataset.longitudes
    lat = dataset.latitudes
    
    os.makedirs(os.path.join(outdir, "Plots"), exist_ok=True)
    
    fig, axes = plt.subplots(
        len(var_indices), 2,
        figsize=(12, 4.5 * len(var_indices)),
        constrained_layout=True,
    )
    if len(var_indices) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for row, vid in enumerate(var_indices):
        vname = dataset.var_names[vid]
        tgt_map = target_denorm[vid].numpy()
        pred_map = pred_denorm[vid].numpy()
        
        vmin = min(tgt_map.min(), pred_map.min())
        vmax = max(tgt_map.max(), pred_map.max())
        
        for col, (data, title) in enumerate(zip(
            [tgt_map, pred_map],
            ["Ground Truth", "Prediction"]
        )):
            ax = axes[row, col]
            im = ax.imshow(
                data,
                origin="lower",
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{title}: {vname}", fontsize=11)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    save_path = os.path.join(outdir, "Plots", f"predictions_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# =============================================================================
# Distributed Training Setup
# =============================================================================

def setup_distributed():
    """Initialize distributed training if using torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), dist.get_rank()
    return None, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process."""
    return rank in (0, None)


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="FuXi Pre-training with Zarr Store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    data = p.add_argument_group("Data")
    data.add_argument("--zarr-store", type=str, required=True,
                      help="Path to Zarr store")
    data.add_argument("--train-start", type=str, default="1979-01-01",
                      help="Training start date (YYYY-MM-DD)")
    data.add_argument("--train-end", type=str, default="2015-12-31",
                      help="Training end date (YYYY-MM-DD)")
    data.add_argument("--val-start", type=str, default="2016-01-01",
                      help="Validation start date (YYYY-MM-DD)")
    data.add_argument("--val-end", type=str, default="2018-12-31",
                      help="Validation end date (YYYY-MM-DD)")
    data.add_argument("--test-start", type=str, default="2019-01-01",
                      help="Test start date (YYYY-MM-DD)")
    data.add_argument("--test-end", type=str, default="2020-12-31",
                      help="Test end date (YYYY-MM-DD)")
    data.add_argument("--history-steps", type=int, default=2,
                      help="Number of historical timesteps")
    
    # Training arguments
    train = p.add_argument_group("Training")
    train.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per GPU")
    train.add_argument("--accum-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    train.add_argument("--max-epochs", type=int, default=100,
                       help="Maximum training epochs")
    train.add_argument("--max-iters", type=int, default=None,
                       help="Max iterations (paper: 40000 for pre-train)")
    train.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    train.add_argument("--num-workers", type=int, default=4,
                       help="DataLoader workers")
    
    # Optimizer arguments (paper values)
    opt = p.add_argument_group("Optimizer")
    opt.add_argument("--lr", type=float, default=2.5e-4,
                     help="Learning rate (paper: 2.5e-4 for pre-train)")
    opt.add_argument("--weight-decay", type=float, default=0.1,
                     help="Weight decay")
    opt.add_argument("--beta1", type=float, default=0.9,
                     help="Adam beta1")
    opt.add_argument("--beta2", type=float, default=0.95,
                     help="Adam beta2")
    opt.add_argument("--grad-clip", type=float, default=1.0,
                     help="Gradient clipping (0 to disable)")
    
    # Model arguments
    model = p.add_argument_group("Model")
    model.add_argument("--embed-dim", type=int, default=256,
                       help="Embedding dimension")
    model.add_argument("--num-heads", type=int, default=8,
                       help="Number of attention heads")
    model.add_argument("--window-size", type=int, default=5,
                       help="Window size for attention")
    model.add_argument("--depth-pre", type=int, default=2,
                       help="Depth of pre-processing blocks")
    model.add_argument("--depth-mid", type=int, default=12,
                       help="Depth of middle (main) blocks")
    model.add_argument("--depth-post", type=int, default=2,
                       help="Depth of post-processing blocks")
    model.add_argument("--mlp-ratio", type=float, default=4.0,
                       help="MLP expansion ratio")
    model.add_argument("--drop-path-rate", type=float, default=0.2,
                       help="Stochastic depth rate")
    model.add_argument("--use-checkpoint", action="store_true",
                       help="Use gradient checkpointing (saves memory)")
    
    # I/O arguments
    io = p.add_argument_group("I/O")
    io.add_argument("--runs-dir", type=str, default="Models",
                    help="Root directory for outputs")
    io.add_argument("--exp-name", type=str, default=None,
                    help="Experiment name (auto-generated if not specified)")
    io.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    io.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    
    return p.parse_args()


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    args = parse_args()
    
    # Setup distributed training
    local_rank, world_size, global_rank = setup_distributed()
    
    # Device setup
    use_cuda = torch.cuda.is_available()
    use_amp = use_cuda
    device = torch.device(
        f"cuda:{local_rank}" if use_cuda and local_rank is not None
        else ("cuda" if use_cuda else "cpu")
    )
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # Create experiment directory
    exp_name = args.exp_name or datetime.now().strftime("pretrain_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, exp_name)
    
    if is_main_process(global_rank):
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "Plots"), exist_ok=True)
        
        print("=" * 70)
        print(f"FuXi Pre-training")
        print("=" * 70)
        print(f"Experiment : {exp_name}")
        print(f"Run dir    : {run_dir}")
        print(f"Device     : {device}")
        print(f"World size : {world_size}")
        print(f"Zarr store : {args.zarr_store}")
        if use_cuda:
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        print("=" * 70)
    
    # Synchronize all processes
    if world_size > 1:
        dist.barrier()
    
    # =============================================================================
    # Load Data
    # =============================================================================
    
    if is_main_process(global_rank):
        print("\nLoading datasets...")
        print(f"  Train: {args.train_start} → {args.train_end}")
        print(f"  Val  : {args.val_start} → {args.val_end}")
        print(f"  Test : {args.test_start} → {args.test_end}")
    
    t0 = time.time()
    
    # Training set
    train_set = FuXiZarrDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        time_start=args.train_start,
        time_end=args.train_end,
    )
    
    # Validation set (reuse train stats)
    val_set = FuXiZarrDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        mean=train_set.mean,
        std=train_set.std,
        time_start=args.val_start,
        time_end=args.val_end,
    )
    
    # Test set (reuse train stats)
    test_set = FuXiZarrDataset(
        args.zarr_store,
        history_steps=args.history_steps,
        mean=train_set.mean,
        std=train_set.std,
        time_start=args.test_start,
        time_end=args.test_end,
    )
    
    if is_main_process(global_rank):
        print(f"  Loaded in {time.time() - t0:.1f}s")
        print(f"  Train: {len(train_set):,} samples")
        print(f"  Val  : {len(val_set):,} samples")
        print(f"  Test : {len(test_set):,} samples")
        print(f"  Channels: {train_set.channels}")
        print(f"  Spatial : {train_set.spatial_shape}")
    
    # Create data loaders
    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if world_size > 1 else None
    
    # Use fewer workers for lazy loading
    num_workers = min(args.num_workers, 2)
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    
    # =============================================================================
    # Create Model
    # =============================================================================
    
    if is_main_process(global_rank):
        print("\nInitializing model...")
    
    spatial_h, spatial_w = train_set.spatial_shape
    num_vars = train_set.channels
    
    model = FuXi(
        num_variables=num_vars,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        window_size=args.window_size,
        depth_pre=args.depth_pre,
        depth_mid=args.depth_mid,
        depth_post=args.depth_post,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        input_height=spatial_h,
        input_width=spatial_w,
        use_checkpoint=args.use_checkpoint,
    ).to(device)
    
    if is_main_process(global_rank):
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # =============================================================================
    # Resume from Checkpoint (if specified)
    # =============================================================================
    
    start_epoch = 1
    best_val = float("inf")
    global_step = 0
    
    if args.resume and os.path.isfile(args.resume):
        if is_main_process(global_rank):
            print(f"\nResuming from: {args.resume}")
        
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss", float("inf"))
        global_step = ckpt.get("global_step", 0)
        
        if is_main_process(global_rank):
            print(f"  Resumed from epoch {ckpt.get('epoch', '?')}")
            print(f"  Best val loss: {best_val:.5f}")
            print(f"  Global step: {global_step}")
    
    # =============================================================================
    # Wrap in DDP (if multi-GPU)
    # =============================================================================
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if is_main_process(global_rank):
            print(f"  Wrapped in DDP on {world_size} GPUs")
    
    # Keep reference to raw model for saving
    raw_model = model.module if hasattr(model, "module") else model
    
    # =============================================================================
    # Setup Optimizer & Scheduler
    # =============================================================================
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.lr * 0.01,
    )
    
    # Resume optimizer state if available
    if args.resume and os.path.isfile(args.resume):
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
    
    if is_main_process(global_rank):
        print(f"\nOptimizer: AdamW")
        print(f"  LR: {args.lr}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Betas: ({args.beta1}, {args.beta2})")
        print(f"  Effective batch size: {args.batch_size * args.accum_steps * world_size}")
    
    # =============================================================================
    # Setup Loss Function
    # =============================================================================
    
    latitudes = torch.tensor(train_set.latitudes, dtype=torch.float32)
    criterion = LatitudeWeightedL1Loss(
        num_lat=spatial_h,
        lat_range=(latitudes.min().item(), latitudes.max().item()),
    ).to(device)
    
    # Also need latitudes tensor for evaluation
    latitudes = latitudes.to(device)
    
    # =============================================================================
    # Setup Mixed Precision
    # =============================================================================
    
    scaler = GradScaler(enabled=use_amp)
    
    # =============================================================================
    # Save Configuration
    # =============================================================================
    
    if is_main_process(global_rank):
        config = vars(args).copy()
        config["world_size"] = world_size
        config["num_parameters"] = sum(p.numel() for p in model.parameters())
        config["spatial_shape"] = list(train_set.spatial_shape)
        config["num_channels"] = train_set.channels
        
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\nConfig saved to: {os.path.join(run_dir, 'config.json')}")
    
    # =============================================================================
    # Training Loop
    # =============================================================================
    
    if is_main_process(global_rank):
        print("\n" + "=" * 70)
        print("Starting Pre-training (Single-Step Prediction)")
        print("=" * 70 + "\n")
    
    train_losses = []
    val_losses = []
    no_improve = 0
    
    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_start = time.time()
        
        # Set epoch for distributed sampler
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, use_amp, args.grad_clip, args.accum_steps,
            args.max_iters, global_step,
        )
        
        # Validate
        val_loss, val_mae = evaluate(
            model, val_loader, criterion, device, use_amp,
        )
        
        # Step scheduler
        scheduler.step()
        
        # Logging (main process only)
        if is_main_process(global_rank):
            epoch_time = time.time() - epoch_start
            lr_now = optimizer.param_groups[0]["lr"]
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(
                f"Epoch {epoch:3d}/{args.max_epochs} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | mae={val_mae:.5f} | "
                f"lr={lr_now:.2e} | step={global_step} | time={epoch_time:.1f}s"
            )
            
            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "config": config,
            }
            
            # Always save last checkpoint
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            
            # Save best checkpoint
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  ★ New best: {best_val:.5f}")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"\nEarly stopping after {args.patience} epochs without improvement")
                    break
            
            # Check iteration limit
            if args.max_iters and global_step >= args.max_iters:
                print(f"\nReached max iterations ({args.max_iters})")
                break
    
    # =============================================================================
    # Final Evaluation & Visualization
    # =============================================================================
    
    # Unwrap model for final evaluation
    net = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    
    if is_main_process(global_rank):
        print("\n" + "=" * 70)
        print("Training Complete - Final Evaluation")
        print("=" * 70)
        
        # Load best checkpoint
        best_ckpt_path = os.path.join(run_dir, "best.pt")
        if os.path.isfile(best_ckpt_path):
            print(f"\nLoading best checkpoint: {best_ckpt_path}")
            checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            net.load_state_dict(checkpoint["model_state"])
        
        # Test evaluation
        print("\nEvaluating on test set...")
        test_loss, test_mae = evaluate(net, test_loader, criterion, device, use_amp)
        print(f"Test Loss: {test_loss:.5f}")
        print(f"Test MAE : {test_mae:.5f}")
        
        # Save metrics
        metrics = {
            "best_epoch": checkpoint.get("epoch", "?") if os.path.isfile(best_ckpt_path) else "?",
            "best_val_loss": best_val,
            "test_loss": test_loss,
            "test_mae": test_mae,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {os.path.join(run_dir, 'metrics.json')}")
        
        # Generate plots
        print("\nGenerating visualizations...")
        plot_losses(train_losses, val_losses, run_dir)
        plot_prediction_maps(
            net, test_set, device,
            train_set.mean, train_set.std,
            run_dir, use_amp,
            sample_idx=0,
        )
        
        print("\n" + "=" * 70)
        print(f"All outputs saved to: {run_dir}")
        print("=" * 70)
    
    # Synchronize before cleanup
    if world_size > 1:
        dist.barrier()
    
    cleanup_distributed()
    
    if is_main_process(global_rank):
        print("\nTraining complete! 🎉")


if __name__ == "__main__":
    main()

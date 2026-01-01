
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, List

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs  # pip install xskillscore

sns.set_theme(style="whitegrid", context="talk", palette="deep")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def latitude_weights(latitudes: torch.Tensor, device: torch.device) -> torch.Tensor:
    weights = torch.cos(torch.deg2rad(latitudes)).to(device)
    return weights / weights.mean()


def latitude_weighted_rmse(pred, target, latitudes):
    weights = latitude_weights(latitudes, pred.device).view(1, 1, -1, 1)
    return torch.sqrt(((pred - target) ** 2 * weights).mean()).item()


def latitude_weighted_acc(pred, target, climatology, latitudes):
    weights = latitude_weights(latitudes, pred.device).view(1, 1, -1, 1)
    pred_anom = pred - climatology
    targ_anom = target - climatology
    num = (weights * pred_anom * targ_anom).sum(dim=(-2, -1))
    denom = torch.sqrt(
        (weights * pred_anom.pow(2)).sum(dim=(-2, -1))
        * (weights * targ_anom.pow(2)).sum(dim=(-2, -1))
        + 1e-8
    )
    return (num / denom).mean().item()


def _normalize_var_names(var_names: Sequence[str]):
    return list(var_names)


def _collapse_batch_channels(tensor: torch.Tensor, var_names):
    labels = _normalize_var_names(var_names)
    batch, channels, height, width = tensor.shape
    n_vars = len(labels)
    if channels == n_vars:
        return tensor, labels
    if channels % n_vars == 0:
        c_per_var = channels // n_vars
        collapsed = tensor.reshape(batch, n_vars, c_per_var, height, width).mean(dim=2)
        return collapsed, labels
    truncated = tensor[:, :n_vars]
    print(f"[WARN] Truncated channels from {channels} to {n_vars} for per-variable metrics")
    return truncated, labels[:truncated.shape[1]]


def _collapse_sample_channels(tensor: torch.Tensor, var_names):
    labels = _normalize_var_names(var_names)
    if tensor.dim() == 4:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        else:
            raise ValueError(f"Unexpected 4D tensor with leading dim {tensor.shape[0]}")
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C,H,W), got shape {tuple(tensor.shape)}")
    channels, height, width = tensor.shape
    n_vars = len(labels)
    if channels == n_vars:
        return tensor, labels
    if channels % n_vars == 0:
        c_per_var = channels // n_vars
        collapsed = tensor.reshape(n_vars, c_per_var, height, width).mean(dim=1)
        return collapsed, labels
    truncated = tensor[:n_vars]
    print(f"[WARN] Truncated sample channels from {channels} to {n_vars}")
    return truncated, labels[:truncated.shape[0]]


def plot_prediction_maps_all_vars(
    pred_denorm,
    target_denorm,
    lon,
    lat,
    var_names,
    outdir,
    sample_idx=0,
):
    print(f"[DEBUG] Plotting triptych for sample {sample_idx}")
    _ensure_dir(outdir)

    pred_tensor = pred_denorm.detach().cpu() if torch.is_tensor(pred_denorm) else torch.as_tensor(pred_denorm)
    target_tensor = target_denorm.detach().cpu() if torch.is_tensor(target_denorm) else torch.as_tensor(target_denorm)

    pred_tensor, labels = _collapse_sample_channels(pred_tensor, var_names)
    target_tensor, _ = _collapse_sample_channels(target_tensor, labels)

    n_vars = len(labels)
    fig, axes = plt.subplots(n_vars, 3, figsize=(14, 3.2 * n_vars), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, vname in enumerate(labels):
        tgt = target_tensor[i].numpy()
        pred = pred_tensor[i].numpy()
        err = pred - tgt
        panels = [(tgt, "Target", "coolwarm"), (pred, "Prediction", "coolwarm"), (err, "Error", "RdBu_r")]
        for j, (data, title, cmap) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(
                data,
                origin="lower",
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                cmap=cmap,
            )
            ax.set_title(f"{vname} — {title}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(outdir / f"sample_{sample_idx:03d}_triptych.png", dpi=300)
    plt.close(fig)


def plot_per_variable_metrics(per_var_df: pd.DataFrame, outdir: Path):
    print("[DEBUG] Plotting per-variable metrics")
    _ensure_dir(outdir)

    df = per_var_df.sort_values("lat_rmse", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=df,
        y="variable",
        x="lat_rmse",
        color="#1f77b4",
        ax=ax,
        edgecolor="none",
    )
    ax.set_xlabel("Latitude-weighted RMSE")
    ax.set_ylabel("")
    ax.set_title("Per-variable Error Profile", pad=14)

    ax2 = ax.twiny()
    sns.scatterplot(
        data=df,
        x="mae",
        y="variable",
        color="#ff7f0e",
        s=45,
        ax=ax2,
        legend=False,
    )
    ax2.set_xlabel("MAE")
    ax.grid(axis="x", color="0.85")
    fig.tight_layout()
    fig.savefig(outdir / "per_variable_errors.png", dpi=300)
    plt.close(fig)


def plot_latitudinal_error(preds, targets, latitudes, outdir: Path):
    print("[DEBUG] Plotting latitudinal error bands")
    _ensure_dir(outdir)
    latitudes = latitudes.cpu().numpy()
    mae_lat = (preds - targets).abs().mean(dim=1).mean(dim=-1).cpu().numpy()
    mean_err = mae_lat.mean(axis=0)
    std_err = mae_lat.std(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(latitudes, mean_err, color="#1f77b4", linewidth=2)
    ax.fill_between(latitudes, mean_err - std_err, mean_err + std_err, color="#1f77b4", alpha=0.25)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("|Error|")
    ax.set_title("Latitudinal Error Envelope")
    fig.tight_layout()
    fig.savefig(outdir / "latitudinal_error.png", dpi=300)
    plt.close(fig)


def plot_temporal_rmse(preds, targets, outdir: Path):
    print("[DEBUG] Plotting temporal RMSE curve")
    _ensure_dir(outdir)
    diff = preds - targets
    rmse = diff.view(diff.shape[0], -1).pow(2).mean(dim=1).sqrt().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(rmse)), rmse, color="#d62728", linewidth=1.6)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("RMSE")
    ax.set_title("Temporal RMSE Drift")
    fig.tight_layout()
    fig.savefig(outdir / "temporal_rmse.png", dpi=300)
    plt.close(fig)


def plot_scatter(preds_denorm, targets_denorm, var_names, outdir: Path, var_subset=None):
    print("[DEBUG] Plotting scatter diagnostics")
    _ensure_dir(outdir)
    preds_c, labels = _collapse_batch_channels(preds_denorm, var_names)
    targets_c, _ = _collapse_batch_channels(targets_denorm, labels)
    if var_subset is None:
        var_subset = labels[:4]
    for vname in var_subset:
        if vname not in labels:
            continue
        idx = labels.index(vname)
        x = targets_c[:, idx].reshape(-1).numpy()
        y = preds_c[:, idx].reshape(-1).numpy()
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        sns.scatterplot(x=x, y=y, s=6, alpha=0.25, ax=ax, edgecolor=None)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.set_title(f"Scatter: {vname}")
        fig.tight_layout()
        fig.savefig(outdir / f"scatter_{vname}.png", dpi=300)
        plt.close(fig)


def plot_residual_hist(preds_denorm, targets_denorm, var_names, outdir: Path, max_vars: int = 6):
    print("[DEBUG] Plotting residual histograms")
    _ensure_dir(outdir)
    preds_c, labels = _collapse_batch_channels(preds_denorm, var_names)
    targets_c, _ = _collapse_batch_channels(targets_denorm, labels)
    selected = labels[:max_vars]
    for vname in selected:
        idx = labels.index(vname)
        residual = (preds_c[:, idx] - targets_c[:, idx]).reshape(-1).numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residual, bins=60, kde=True, ax=ax, color="#9467bd", stat="density")
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"Residual Distribution: {vname}", fontsize=11, pad=10)
        ax.set_xlabel("Prediction − Target")
        ax.set_ylabel("Density")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(outdir / f"residual_hist_{vname}.png", dpi=300)
        plt.close(fig)


def plot_training_curves(history_csv: Optional[str], outdir: Path):
    if history_csv is None or not Path(history_csv).is_file():
        print("[DEBUG] Training history not found; skipping curve plot")
        return
    print(f"[DEBUG] Plotting training curves from {history_csv}")
    history = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, color in zip(("train_loss", "val_l1", "val_mae"), ("#1f77b4", "#ff7f0e", "#2ca02c")):
        if col in history:
            sns.lineplot(data=history, x="epoch", y=col, ax=ax, label=col, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Error")
    ax.set_title("Training vs Validation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "training_curves.png", dpi=300)
    plt.close(fig)


def per_variable_metrics(preds, targets, latitudes, var_names):
    print("[DEBUG] Computing per-variable metrics")
    preds_var, labels = _collapse_batch_channels(preds, var_names)
    targets_var, _ = _collapse_batch_channels(targets, labels)

    weights = latitude_weights(latitudes, preds.device).view(1, 1, -1, 1)
    rows = []
    for i, vname in enumerate(labels):
        diff = preds_var[:, i] - targets_var[:, i]
        rows.append(
            {
                "variable": vname,
                "lat_rmse": torch.sqrt((diff ** 2 * weights).mean()).item(),
                "mae": diff.abs().mean().item(),
            }
        )
    return pd.DataFrame(rows).sort_values("lat_rmse").reset_index(drop=True), labels


def sample_metrics(pred_sample, targ_sample, latitudes, var_names):
    print("[DEBUG] Computing sample metrics")
    pred_var, labels = _collapse_sample_channels(pred_sample, var_names)
    targ_var, _ = _collapse_sample_channels(targ_sample, labels)

    weights = latitude_weights(latitudes, pred_sample.device).view(1, -1, 1)
    rows = []
    for i, vname in enumerate(labels):
        diff = pred_var[i] - targ_var[i]
        rows.append(
            {
                "variable": vname,
                "sample_lat_rmse": torch.sqrt((diff.pow(2) * weights).mean()).item(),
                "sample_mae": diff.abs().mean().item(),
                "max_abs": diff.abs().max().item(),
            }
        )
    return pd.DataFrame(rows)


def _fetch_future_targets(dataset, start_idx: int, steps: int):
    for name in ("get_future_targets", "get_future_sequence", "get_rollout_targets"):
        fetcher = getattr(dataset, name, None)
        if callable(fetcher):
            future = fetcher(start_idx, steps)
            if isinstance(future, tuple):
                future = future[-1]
            future_tensor = torch.as_tensor(future)
            if future_tensor.ndim == 4:
                return future_tensor
            raise ValueError(f"Future helper {name} returned shape {future_tensor.shape}")
    seq = []
    for offset in range(steps):
        idx = start_idx + offset
        if idx >= len(dataset):
            break
        _, targ = dataset[idx]
        if not torch.is_tensor(targ):
            targ = torch.as_tensor(targ)
        seq.append(targ)
    if not seq:
        return None
    return torch.stack(seq)


def _prepare_history(seed_history: torch.Tensor) -> torch.Tensor:
    if seed_history.ndim == 3:
        return seed_history.unsqueeze(0)
    if seed_history.ndim == 4:
        return seed_history.unsqueeze(0)
    raise ValueError(f"Unsupported seed history shape: {seed_history.shape}")


def _roll_history(history: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if history.dim() == pred.dim() + 1:
        return torch.cat([history[:, 1:], pred.unsqueeze(1)], dim=1)
    if history.dim() == pred.dim():
        out_channels = pred.shape[1]
        if history.shape[1] < out_channels:
            raise ValueError(
                f"Cannot roll history with {history.shape[1]} channels using prediction with {out_channels} channels"
            )
        return torch.cat([history[:, out_channels:], pred], dim=1)
    raise ValueError(f"Incompatible history/pred dims: {history.dim()} vs {pred.dim()}")


def compute_acc_vs_lead(
    model,
    seed_history: torch.Tensor,
    future_targets: torch.Tensor,
    latitudes,
    climatology: Optional[torch.Tensor] = None,
):
    if future_targets.ndim != 4:
        raise ValueError(f"future_targets must be (T,C,H,W), got {future_targets.shape}")

    device = next(model.parameters()).device
    lat_tensor = torch.as_tensor(latitudes, dtype=torch.float32, device=device)
    history = _prepare_history(seed_history).to(device)
    targets = future_targets.to(device)

    clim_tensor = None
    if climatology is not None:
        clim_tensor = torch.as_tensor(climatology, dtype=torch.float32)
        if clim_tensor.ndim == 3:
            clim_tensor = clim_tensor.unsqueeze(0).to(device)
        elif clim_tensor.ndim == 4 and clim_tensor.shape[0] >= targets.shape[0]:
            clim_tensor = clim_tensor.to(device)
        else:
            raise ValueError(f"Unexpected climatology shape: {clim_tensor.shape}")

    acc_scores: List[float] = []
    with torch.inference_mode():
        for lead in range(targets.shape[0]):
            pred = model(history, target_shape=targets.shape[-2:])
            if clim_tensor is None:
                clim = torch.zeros_like(pred)
            elif clim_tensor.ndim == 4:
                clim = clim_tensor[lead : lead + 1]
            else:
                clim = clim_tensor
            acc_val = latitude_weighted_acc(pred, targets[lead : lead + 1], clim, lat_tensor)
            acc_scores.append(acc_val)
            history = _roll_history(history, pred)

    return acc_scores


def plot_acc_vs_lead(acc_scores, outdir: Path, baseline: float = 0.5):
    print("[DEBUG] Plotting ACC vs lead time")
    _ensure_dir(outdir)
    leads = np.arange(1, len(acc_scores) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=leads, y=acc_scores, marker="o", linewidth=2.3, ax=ax, label="Average ACC")
    ax.axhline(baseline, color="red", linestyle="--", linewidth=1.4, label=f"ACC = {baseline}")
    ax.set_xlabel("Lead Time (steps)")
    ax.set_ylabel("ACC")
    ax.set_ylim(0, 1)
    ax.set_title("Average ACC vs Lead Time")
    ax.grid(alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "acc_vs_lead.png", dpi=300)
    plt.close(fig)


def eval_single_step(
    model,
    dataset,
    device,
    mean,
    std,
    outdir,
    climatology=None,
    sample_idx: int = 0,
    history_csv: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    acc_lead_steps: int = 0,
    acc_num_samples: int = 4,
) -> Dict[str, float]:
    print("[DEBUG] Starting evaluation")
    outdir = _ensure_dir(Path(outdir))
    plots_dir = _ensure_dir(outdir / "Plots")

    pin_memory = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    latitudes = torch.tensor(dataset.latitudes, dtype=torch.float32, device=device)
    print(f"[DEBUG] Data loader batches: {len(loader)}, batch_size: {batch_size}, device: {device}")

    mean_cpu = torch.as_tensor(mean).cpu()
    std_cpu = torch.as_tensor(std).cpu()

    preds, targets = [], []
    with torch.inference_mode():
        for idx, (history, target) in enumerate(loader):
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(history, target_shape=target.shape[-2:])
            preds.append(pred.cpu())
            targets.append(target.cpu())
            if idx % 10 == 0:
                print(f"[DEBUG] Processed batch {idx + 1}/{len(loader)}")
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    print(f"[DEBUG] Prediction tensor shape: {preds.shape}")

    mean_dev = mean_cpu.to(preds.device)
    std_dev = std_cpu.to(preds.device)
    preds_denorm = preds * std_dev + mean_dev
    targets_denorm = targets * std_dev + mean_dev

    metrics = {
        "lat_rmse": latitude_weighted_rmse(preds, targets, latitudes),
        "mae": torch.mean(torch.abs(preds - targets)).item(),
    }

    if climatology is not None:
        print("[DEBUG] Computing ACC with provided climatology")
        clim = torch.tensor(climatology, dtype=preds.dtype, device=preds.device)
        metrics["acc"] = latitude_weighted_acc(preds, targets, clim, latitudes)
    else:
        metrics["acc"] = None

    print("[DEBUG] Computing CRPS")
    targets_da = xr.DataArray(
        targets_denorm.numpy(),
        dims=("sample", "variable", "lat", "lon"),
    )
    preds_da = xr.DataArray(
        preds_denorm.numpy(),
        dims=("sample", "variable", "lat", "lon"),
    ).expand_dims(member=[0])
    metrics["crps"] = xs.crps_ensemble(
        observations=targets_da,
        forecasts=preds_da,
        member_dim="member",
    ).mean().item()

    per_var_df, per_var_labels = per_variable_metrics(preds, targets, latitudes, dataset.var_names)
    per_var_df.to_csv(outdir / "per_variable_metrics.csv", index=False)
    plot_per_variable_metrics(per_var_df, plots_dir)
    plot_latitudinal_error(preds, targets, latitudes, plots_dir)
    plot_temporal_rmse(preds, targets, plots_dir)
    plot_scatter(preds_denorm, targets_denorm, per_var_labels, plots_dir)
    plot_residual_hist(preds_denorm, targets_denorm, per_var_labels, plots_dir)
    plot_training_curves(history_csv, plots_dir)

    print(f"[DEBUG] Running single-sample visualization for idx {sample_idx}")
    history_seed, target_sample = dataset[sample_idx]
    history_sample = history_seed.unsqueeze(0).to(device) if history_seed.ndim == 3 else history_seed.unsqueeze(0).to(device)
    target_sample = target_sample.to(device)
    with torch.inference_mode():
        pred_sample = model(history_sample, target_shape=target_sample.shape[-2:])

    pred_sample_cpu = pred_sample[0].cpu()
    target_sample_cpu = target_sample.cpu()
    pred_sample_denorm = pred_sample_cpu * std_cpu + mean_cpu
    target_sample_denorm = target_sample_cpu * std_cpu + mean_cpu

    latitudes_cpu = latitudes.cpu()
    sample_df = sample_metrics(pred_sample_cpu, target_sample_cpu, latitudes_cpu, per_var_labels)
    sample_df.to_csv(outdir / f"sample_{sample_idx:03d}_metrics.csv", index=False)
    plot_prediction_maps_all_vars(
        pred_sample_denorm,
        target_sample_denorm,
        dataset.longitudes,
        dataset.latitudes,
        per_var_labels,
        plots_dir,
        sample_idx=sample_idx,
    )

    if acc_lead_steps > 0:
        max_start = len(dataset) - acc_lead_steps
        if max_start <= 0:
            print("[WARN] Dataset too short for requested lead steps; skipping ACC vs lead plot.")
        else:
            acc_indices = list(range(min(acc_num_samples, max_start)))
            acc_matrix = []
            for idx in acc_indices:
                seed_history, _ = dataset[idx]
                future_targets = _fetch_future_targets(dataset, idx, acc_lead_steps)
                if future_targets is None or future_targets.shape[0] == 0:
                    print(f"[WARN] Could not fetch future targets for index {idx}; skipping.")
                    continue
                scores = compute_acc_vs_lead(
                    model=model,
                    seed_history=seed_history,
                    future_targets=future_targets,
                    latitudes=latitudes_cpu,
                    climatology=climatology,
                )
                if scores:
                    acc_matrix.append(scores)
            if acc_matrix:
                max_len = max(len(row) for row in acc_matrix)
                acc_array = np.full((len(acc_matrix), max_len), np.nan, dtype=np.float32)
                for ridx, row in enumerate(acc_matrix):
                    acc_array[ridx, : len(row)] = row
                avg_scores = np.nanmean(acc_array, axis=0)
                plot_acc_vs_lead(avg_scores, plots_dir)
                pd.DataFrame({"lead": np.arange(1, len(avg_scores) + 1), "acc": avg_scores}).to_csv(
                    outdir / "acc_vs_lead.csv", index=False
                )
            else:
                print("[WARN] ACC vs lead computation skipped; no valid sequences found.")

    with open(outdir / "summary_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if v is not None}, f, indent=2)

    print("=== Global Metrics ===")
    for k, v in metrics.items():
        if v is not None:
            print(f"{k}: {v:.4f}")

    print("\n=== Per-variable Metrics ===")
    print(per_var_df)

    print("\n=== Sample Metrics ===")
    print(sample_df)

    return metrics


def _parse_tuple(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(
        int(v.strip()) for v in str(value).replace("[", "").replace("]", "").split(",") if v.strip()
    )


def _unwrap_state_dict(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent
    sys.path.append(str(ROOT))

    from train import MiniFuXiDataset as FuXiDataset
    from fuxi import FuXiModel

    ckpt = "Models/exp_FuXi3Stage_20251118_005002/checkpoints/fuxi_epoch023.pt"
    cfg_path = "Models/exp_FuXi3Stage_20251118_005002/config.json"
    outdir = "Models/exp_FuXi3Stage_20251118_005002/Eval"
    history_csv = "Models/exp_FuXi3Stage_20251118_005002/train_log.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[DEBUG] Loading config from {cfg_path}")
    config = json.load(open(cfg_path))
    encoder_dims = _parse_tuple(config["encoder_dims"])
    swin_depths = _parse_tuple(config["swin_depths"])
    swin_heads = _parse_tuple(config["swin_heads"])
    in_channels = int(config["in_channels"])
    out_channels = int(config["out_channels"])
    input_height = int(config["input_height"])
    input_width = int(config["input_width"])

    print(f"[DEBUG] Loading checkpoint from {ckpt}")
    raw_state = torch.load(ckpt, map_location=device)
    model_state = _unwrap_state_dict(raw_state.get("model_state", raw_state))

    data_root = "/home/raj.ayush/New/Fuxi-Weather-Prediction/Fuxi/Fuxi_3P/Data"
    dataset_path = f"{data_root}/test_data_2021_2023.nc"
    print(f"[DEBUG] Initializing dataset from {dataset_path}")
    dataset = FuXiDataset(
        path=dataset_path,
        history_steps=config.get("history_steps", 2),
        mean=None,
        std=None,
    )
    print(f"[DEBUG] Dataset samples: {len(dataset)}, mean shape: {dataset.mean.shape}, std shape: {dataset.std.shape}")

    print("[DEBUG] Building FuXiModel")
    model = FuXiModel(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=int(config["embed_dim"]),
        encoder_dims=encoder_dims,
        swin_depths=swin_depths,
        swin_heads=swin_heads,
        swin_window_size=int(config["swin_window_size"]),
        drop_path_rate=float(config["drop_path_rate"]),
        input_height=input_height,
        input_width=input_width,
    ).to(device)
    print("[DEBUG] Loading model weights")
    model.load_state_dict(model_state)

    eval_bs = int(config.get("test_batch_size", config.get("val_batch_size", 1)))
    eval_workers = min(4, eval_bs)
    acc_leads = int(config.get("eval_acc_lead_steps", 10))
    acc_samples = int(config.get("eval_acc_num_samples", 6))

    eval_single_step(
        model=model,
        dataset=dataset,
        device=device,
        mean=dataset.mean,
        std=dataset.std,
        outdir=outdir,
        climatology=getattr(dataset, "climatology", None),
        sample_idx=0,
        history_csv=history_csv,
        batch_size=eval_bs,
        num_workers=eval_workers,
        acc_lead_steps=acc_leads,
        acc_num_samples=acc_samples,
    )
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk", palette="deep")


# ---------- IO / utils ----------
def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_tuple(value) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    parts = str(value).replace("[", "").replace("]", "").split(",")
    return tuple(int(p.strip()) for p in parts if p.strip())


def _unwrap_state_dict(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# ---------- metrics ----------
def latitude_weights(latitudes: torch.Tensor, device: torch.device) -> torch.Tensor:
    w = torch.cos(torch.deg2rad(latitudes)).to(device)
    return w / (w.mean() + 1e-8)


def latitude_weighted_acc(pred, target, climatology, latitudes):
    # pred/target: (B=1,C,H,W)
    weights = latitude_weights(latitudes, pred.device).view(1, 1, -1, 1)
    pred_anom = pred - climatology
    targ_anom = target - climatology
    num = (weights * pred_anom * targ_anom).sum(dim=(-2, -1))
    den = torch.sqrt(
        (weights * pred_anom.pow(2)).sum(dim=(-2, -1))
        * (weights * targ_anom.pow(2)).sum(dim=(-2, -1))
        + 1e-8
    )
    return (num / den).mean().item()


def latitude_weighted_rmse(pred, target, latitudes):
    weights = latitude_weights(latitudes, pred.device).view(1, 1, -1, 1)
    mse = (weights * (pred - target).pow(2)).sum(dim=(-2, -1)) / (weights.sum(dim=(-2, -1)) + 1e-8)
    return torch.sqrt(mse.mean()).item()


# ---------- sequence prep ----------
def _prepare_history(seed_history: torch.Tensor, history_steps: int) -> torch.Tensor:
    # Return shape (B=1, C, T, H, W)
    x = torch.as_tensor(seed_history, dtype=torch.float32)
    if x.ndim == 5:
        return x
    if x.ndim == 4:
        # Either (T,C,H,W) or (C,T,H,W)
        if x.shape[0] == history_steps:      # (T,C,H,W)
            x = x.permute(1, 0, 2, 3)        # -> (C,T,H,W)
        elif x.shape[1] != history_steps:    # (C,T,H,W) or bad
            raise ValueError(f"Incompatible 4D history shape {tuple(x.shape)} for history_steps={history_steps}")
        return x.unsqueeze(0)
    if x.ndim == 3:
        c_total = x.shape[0]
        if c_total % history_steps != 0:
            raise ValueError("History channels not divisible by history_steps")
        per_step = c_total // history_steps
        x = x.view(history_steps, per_step, x.shape[1], x.shape[2]).permute(1, 0, 2, 3)
        return x.unsqueeze(0)
    raise ValueError(f"Unsupported history shape {tuple(x.shape)}")


def _roll_history(history: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # history: (B,C,T,H,W), pred: (B,C,H,W) -> append as new last T
    return torch.cat([history[:, :, 1:], pred.unsqueeze(2)], dim=2)


def _fetch_future_targets(dataset, start_idx: int, steps: int):
    # Prefer dataset helper if available
    for name in ("get_future_targets", "get_future_sequence", "get_rollout_targets"):
        fn = getattr(dataset, name, None)
        if callable(fn):
            fut = fn(start_idx, steps)
            if isinstance(fut, tuple):
                fut = fut[-1]
            fut = torch.as_tensor(fut)
            if fut.ndim == 4 and fut.shape[0] >= steps:
                return fut[:steps]
    # Fallback: sequential __getitem__
    seq = []
    for o in range(steps):
        i = start_idx + o
        if i >= len(dataset):
            return None
        _, targ = dataset[i]
        seq.append(torch.as_tensor(targ))
    return torch.stack(seq) if len(seq) == steps else None


# ---------- climatology helpers ----------
def _clim_for_lead(clim: Optional[torch.Tensor], lead: int, like: torch.Tensor) -> torch.Tensor:
    # like: (B=1,C,H,W)
    if clim is None:
        return torch.zeros_like(like)
    c = torch.as_tensor(clim, dtype=torch.float32, device=like.device)
    if c.ndim == 3:           # (C,H,W) constant
        return c.unsqueeze(0)
    if c.ndim == 4:           # (T,C,H,W)
        if lead < c.shape[0]:
            return c[lead:lead + 1]
        return c[-1:].expand_as(like)  # last available
    # Unknown -> zeros
    return torch.zeros_like(like)


# ---------- core eval ----------
def _autoregressive_rollout(model, history, steps, spatial_size):
    preds = []
    with torch.inference_mode():
        for _ in range(steps):
            pred = model(history, target_shape=spatial_size)  # (B,C,H,W)
            preds.append(pred.detach())
            history = _roll_history(history, pred)
    return torch.stack(preds)  # (T,B,C,H,W)


def compute_metrics_vs_lead(model, seed_history, future_targets, latitudes, climatology, history_steps):
    device = next(model.parameters()).device
    lat = torch.as_tensor(latitudes, dtype=torch.float32, device=device)
    history = _prepare_history(seed_history, history_steps).to(device)
    targets = torch.as_tensor(future_targets, dtype=torch.float32, device=device)  # (T,C,H,W)
    steps = targets.shape[0]

    preds = _autoregressive_rollout(model, history, steps, spatial_size=targets.shape[-2:])  # (T,1,C,H,W) if B=1
    if preds.dim() == 5:
        preds = preds[:, 0]  # (T,C,H,W)

    acc_list, rmse_list = [], []
    acc_persist_list = []

    # Persistence baseline
    last_obs = history[:, :, -1].detach()  # (B=1,C,H,W)
    persist = torch.stack([last_obs[0]] * steps)  # (T,C,H,W)

    for t in range(steps):
        pred = preds[t : t + 1]                  # (1,C,H,W)
        targ = targets[t : t + 1]                # (1,C,H,W)
        clim_t = _clim_for_lead(climatology, t, like=pred)

        acc = latitude_weighted_acc(pred, targ, clim_t, lat)
        rmse = latitude_weighted_rmse(pred, targ, lat)
        acc_list.append(acc)
        rmse_list.append(rmse)

        p_step = persist[t : t + 1]
        acc_persist_list.append(latitude_weighted_acc(p_step, targ, clim_t, lat))

    return np.array(acc_list), np.array(rmse_list), np.array(acc_persist_list)


# ---------- plotting ----------
def _plot_curve(x, y, ylabel, outpath, extra=None, title=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(x=x, y=y, marker="o", linewidth=2.4, ax=ax, label="Model")
    if extra is not None:
        sns.lineplot(x=x, y=extra, marker="o", linewidth=1.8, ax=ax, label="Persistence")
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.35)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ---------- driver ----------
def evaluate_and_plot(
    model,
    dataset,
    device,
    outdir,
    lead_steps: int = 15,
    num_samples: int = 12,
    history_steps: int = 1,
    seed: int = 42,
):
    outdir = _ensure_dir(Path(outdir))
    plots_dir = _ensure_dir(outdir / "Plots")

    latitudes = torch.tensor(dataset.latitudes, dtype=torch.float32, device=device)

    max_start = len(dataset) - lead_steps
    if max_start <= 0:
        raise RuntimeError("Dataset too short for requested lead_steps.")

    random.seed(seed)
    start_indices = random.sample(range(max_start), k=min(num_samples, max_start))

    acc_rows, rmse_rows, acc_persist_rows = [], [], []
    for s, idx in enumerate(start_indices, 1):
        seed_history, _ = dataset[idx]
        fut = _fetch_future_targets(dataset, idx, lead_steps)
        if fut is None:
            print(f"[WARN] idx={idx}: missing {lead_steps} future steps; skip.")
            continue

        clim = getattr(dataset, "climatology", None)
        acc, rmse, acc_p = compute_metrics_vs_lead(
            model=model,
            seed_history=seed_history,
            future_targets=fut,
            latitudes=latitudes,
            climatology=clim,
            history_steps=history_steps,
        )
        acc_rows.append(acc)
        rmse_rows.append(rmse)
        acc_persist_rows.append(acc_p)
        print(f"[INFO] sample {s}/{len(start_indices)} (idx={idx}) done.")

    if not acc_rows:
        raise RuntimeError("No valid sequences processed.")

    acc_avg = np.nanmean(np.stack(acc_rows), axis=0)
    rmse_avg = np.nanmean(np.stack(rmse_rows), axis=0)
    acc_persist_avg = np.nanmean(np.stack(acc_persist_rows), axis=0)
    leads_days = np.arange(1, len(acc_avg) + 1)

    # Save CSV
    csv_path = outdir / "metrics_vs_lead.csv"
    np.savetxt(
        csv_path,
        np.column_stack((leads_days, acc_avg, rmse_avg, acc_persist_avg)),
        delimiter=",",
        header="lead_days,acc,rmse,acc_persistence",
        comments="",
        fmt="%.6f",
    )
    print(f"[INFO] Saved metrics to {csv_path}")

    # Plots
    _plot_curve(leads_days, acc_avg, "ACC", plots_dir / "acc_vs_lead.png", extra=acc_persist_avg, title="ACC vs Lead (Model vs Persistence)")
    _plot_curve(leads_days, rmse_avg, "RMSE", plots_dir / "rmse_vs_lead.png", title="RMSE vs Lead")
    print(f"[INFO] Saved plots to {plots_dir}")


if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent
    sys.path.append(str(ROOT))

    from train import MiniFuXiDataset as FuXiDataset
    from fuxi import FuXiModel

    ckpt = "Models/exp_FuXi3Stage_20251118_005002/checkpoints/fuxi_epoch023.pt"
    cfg_path = "Models/exp_FuXi3Stage_20251118_005002/config.json"
    outdir = "Models/exp_FuXi3Stage_20251118_005002/Eval"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[DEBUG] Loading config from {cfg_path}")
    with open(cfg_path) as f:
        config = json.load(f)

    encoder_dims = _parse_tuple(config["encoder_dims"])
    swin_depths = _parse_tuple(config["swin_depths"])
    swin_heads = _parse_tuple(config["swin_heads"])
    in_channels = int(config["in_channels"])
    out_channels = int(config["out_channels"])
    history_steps = int(config.get("history_steps", 1))
    input_height = int(config["input_height"])
    input_width = int(config["input_width"])

    print(f"[DEBUG] Loading checkpoint from {ckpt}")
    raw_state = torch.load(ckpt, map_location=device)
    model_state = _unwrap_state_dict(raw_state.get("model_state", raw_state))

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
    model.load_state_dict(model_state)
    model.eval()

    data_root = "/home/raj.ayush/New/Fuxi-Weather-Prediction/Fuxi/Fuxi_3P/Data"
    dataset_path = f"{data_root}/test_data_2021_2023.nc"
    print(f"[DEBUG] Initializing dataset from {dataset_path}")
    dataset = FuXiDataset(path=dataset_path, history_steps=history_steps, mean=None, std=None)
    dataset_history_steps = getattr(dataset, "history_steps", history_steps)
    print(f"[DEBUG] Dataset samples: {len(dataset)}, history_steps={dataset_history_steps}")

    # Configure leads/samples here (e.g., 5 days => 5)
    lead_steps = int(config.get("eval_acc_lead_steps", 15))
    num_samples = int(config.get("eval_acc_num_samples", 12))

    evaluate_and_plot(
        model=model,
        dataset=dataset,
        device=device,
        outdir=outdir,
        lead_steps=lead_steps,
        num_samples=num_samples,
        history_steps=dataset_history_steps,
        seed=int(config.get("eval_acc_seed", 42)),
    )
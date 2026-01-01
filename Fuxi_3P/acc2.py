import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk", palette="deep")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_tuple(v) -> Tuple[int, ...]:
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return tuple(int(x.strip()) for x in str(v).replace("[","").replace("]","").split(",") if x.strip())


def _unwrap_state(sd):
    if any(k.startswith("module.") for k in sd):
        return {k.replace("module.","",1): v for k,v in sd.items()}
    return sd


def lat_weights(lat: torch.Tensor) -> torch.Tensor:
    w = torch.cos(torch.deg2rad(lat))
    return w / (w.mean() + 1e-8)


def acc_weighted(pred, targ, clim, lat):
    w = lat_weights(lat).view(1,1,-1,1)
    pa = pred - clim
    ta = targ - clim
    num = (w * pa * ta).sum(dim=(-2,-1))
    den = torch.sqrt((w * pa.pow(2)).sum(dim=(-2,-1)) * (w * ta.pow(2)).sum(dim=(-2,-1)) + 1e-8)
    return (num/den).mean().item()


def rmse_weighted(pred, targ, lat):
    w = lat_weights(lat).view(1,1,-1,1)
    mse = (w * (pred - targ).pow(2)).sum(dim=(-2,-1)) / (w.sum(dim=(-2,-1)) + 1e-8)
    return torch.sqrt(mse.mean()).item()


def prep_history(seed_hist: torch.Tensor, history_steps: int) -> torch.Tensor:
    x = torch.as_tensor(seed_hist, dtype=torch.float32)
    if x.ndim == 5:
        return x
    if x.ndim == 4:
        if x.shape[0] == history_steps:
            x = x.permute(1,0,2,3)
        elif x.shape[1] != history_steps:
            raise ValueError(f"Bad 4D history shape {tuple(x.shape)}")
        return x.unsqueeze(0)
    if x.ndim == 3:
        ctot = x.shape[0]
        if ctot % history_steps != 0:
            raise ValueError("Channels not divisible by history_steps")
        per = ctot // history_steps
        x = x.view(history_steps, per, x.shape[1], x.shape[2]).permute(1,0,2,3)
        return x.unsqueeze(0)
    raise ValueError(f"Unsupported history shape {tuple(x.shape)}")


def roll_history(hist: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.cat([hist[:, :, 1:], pred.unsqueeze(2)], dim=2)


def fetch_future(dataset, start: int, steps: int):
    for name in ("get_future_targets","get_future_sequence","get_rollout_targets"):
        fn = getattr(dataset, name, None)
        if callable(fn):
            fut = fn(start, steps)
            if isinstance(fut, tuple):
                fut = fut[-1]
            fut = torch.as_tensor(fut)
            if fut.ndim == 4 and fut.shape[0] >= steps:
                return fut[:steps]
    seq = []
    for o in range(steps):
        i = start + o
        if i >= len(dataset):
            return None
        _, targ = dataset[i]
        seq.append(torch.as_tensor(targ))
    return torch.stack(seq) if len(seq) == steps else None


def clim_for_lead(clim, lead: int, like: torch.Tensor):
    if clim is None:
        return torch.zeros_like(like)
    c = torch.as_tensor(clim, dtype=torch.float32, device=like.device)
    if c.ndim == 3:
        return c.unsqueeze(0)
    if c.ndim == 4:
        if lead < c.shape[0]:
            return c[lead:lead+1]
        return c[-1:].expand_as(like)
    return torch.zeros_like(like)


def rollout(model, history, steps, spatial):
    preds = []
    with torch.inference_mode():
        for _ in range(steps):
            p = model(history, target_shape=spatial)
            preds.append(p.detach())
            history = roll_history(history, p)
    return torch.stack(preds)  # (T,B,C,H,W)


def check_climatology(clim):
    if clim is None:
        return {"has_climatology": False, "mean_abs": 0.0, "std": 0.0}
    c = torch.as_tensor(clim, dtype=torch.float32)
    return {"has_climatology": True, "mean_abs": float(c.abs().mean()), "std": float(c.std())}


def evaluate(model, dataset, device, outdir, lead_steps, num_samples, history_steps, seed):
    outdir = _ensure_dir(Path(outdir))
    plots_dir = _ensure_dir(outdir / "Plots")

    lat = torch.tensor(dataset.latitudes, dtype=torch.float32, device=device)
    max_start = len(dataset) - lead_steps
    if max_start <= 0:
        raise RuntimeError("Dataset too short")

    random.seed(seed)
    starts = random.sample(range(max_start), k=min(num_samples, max_start))

    var_names = getattr(dataset, "variable_names", None) or getattr(dataset, "variables", None)
    if var_names is not None:
        var_names = list(var_names)
    else:
        var_names = [f"ch_{i}" for i in range(getattr(dataset, "out_channels", 0) or 0)]

    clim = getattr(dataset, "climatology", None)
    clim_diag = check_climatology(clim)
    print(f"[CHECK] Climatology: {clim_diag}")

    acc_rows = []
    rmse_rows = []
    acc_persist_rows = []
    acc_random_rows = []
    ch_acc_all = []
    ch_rmse_all = []

    for s, start in enumerate(starts, 1):
        seed_hist, _ = dataset[start]
        fut = fetch_future(dataset, start, lead_steps)
        if fut is None:
            print(f"[WARN] start={start} insufficient future targets")
            continue
        hist = prep_history(seed_hist, history_steps).to(device)
        tgt = torch.as_tensor(fut, dtype=torch.float32, device=device)  # (T,C,H,W)
        preds = rollout(model, hist, lead_steps, spatial=tgt.shape[-2:])  # (T,B,C,H,W)
        preds = preds[:,0]

        last_obs = hist[:, :, -1].detach()[0]
        persistence = torch.stack([last_obs]*lead_steps).to(device)

        # Random baseline: spatial mean field + small noise
        spatial_mean = tgt.mean(dim=(-2,-1), keepdim=True)
        rand_base = spatial_mean + 0.01 * torch.randn_like(tgt)

        sample_acc = []
        sample_rmse = []
        sample_acc_persist = []
        sample_acc_random = []
        ch_acc_matrix = []
        ch_rmse_matrix = []

        for l in range(lead_steps):
            pred = preds[l:l+1]
            target = tgt[l:l+1]
            clim_l = clim_for_lead(clim, l, like=pred)

            acc_val = acc_weighted(pred, target, clim_l, lat)
            rmse_val = rmse_weighted(pred, target, lat)
            sample_acc.append(acc_val)
            sample_rmse.append(rmse_val)

            p_pred = persistence[l:l+1].unsqueeze(0)  # (1,C,H,W)
            sample_acc_persist.append(acc_weighted(p_pred, target, clim_l, lat))

            r_pred = rand_base[l:l+1]
            sample_acc_random.append(acc_weighted(r_pred, target, clim_l, lat))

            ch_acc_lead = []
            ch_rmse_lead = []
            for c in range(pred.shape[1]):
                pc = pred[:, c:c+1]
                tc = target[:, c:c+1]
                cc = clim_l[:, c:c+1]
                ch_acc_lead.append(acc_weighted(pc, tc, cc, lat))
                ch_rmse_lead.append(rmse_weighted(pc, tc, lat))
            ch_acc_matrix.append(ch_acc_lead)
            ch_rmse_matrix.append(ch_rmse_lead)

        acc_rows.append(sample_acc)
        rmse_rows.append(sample_rmse)
        acc_persist_rows.append(sample_acc_persist)
        acc_random_rows.append(sample_acc_random)
        ch_acc_all.append(ch_acc_matrix)
        ch_rmse_all.append(ch_rmse_matrix)
        print(f"[INFO] Sample {s}/{len(starts)} done (start={start})")

    if not acc_rows:
        raise RuntimeError("No sequences processed")

    acc_avg = np.nanmean(np.array(acc_rows), axis=0)
    rmse_avg = np.nanmean(np.array(rmse_rows), axis=0)
    acc_persist_avg = np.nanmean(np.array(acc_persist_rows), axis=0)
    acc_random_avg = np.nanmean(np.array(acc_random_rows), axis=0)

    ch_acc_arr = np.nanmean(np.array(ch_acc_all), axis=0)  # (lead, channel)
    ch_rmse_arr = np.nanmean(np.array(ch_rmse_all), axis=0)

    leads_days = np.arange(1, len(acc_avg)+1)

    np.savetxt(
        outdir / "metrics_vs_lead.csv",
        np.column_stack((leads_days, acc_avg, rmse_avg, acc_persist_avg, acc_random_avg)),
        delimiter=",",
        header="lead_days,acc,rmse,acc_persistence,acc_random",
        comments="",
        fmt="%.6f",
    )

    # Per-channel CSV (ACC + RMSE)
    pc_rows = []
    for l in range(len(leads_days)):
        for c, name in enumerate(var_names):
            pc_rows.append([leads_days[l], name, ch_acc_arr[l, c], ch_rmse_arr[l, c]])
    with open(outdir / "per_channel_metrics.csv", "w") as f:
        f.write("lead_days,variable,acc,rmse\n")
        for r in pc_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f}\n")

    # Diagnostics
    import json as _json
    with open(outdir / "diagnostics.json", "w") as f:
        _json.dump({"climatology": clim_diag, "channels": var_names}, f, indent=2)

    plot_curves(leads_days, acc_avg, acc_persist_avg, acc_random_avg, plots_dir / "acc_vs_lead.png")
    plot_curve(leads_days, rmse_avg, "RMSE", plots_dir / "rmse_vs_lead.png")
    plot_per_channel(leads_days, ch_acc_arr, var_names, plots_dir / "acc_per_channel.png", "ACC")
    plot_per_channel(leads_days, ch_rmse_arr, var_names, plots_dir / "rmse_per_channel.png", "RMSE")
    print("[INFO] Saved all outputs.")


def plot_curves(x, acc_model, acc_persist, acc_random, path):
    fig, ax = plt.subplots(figsize=(10,4.5))
    sns.lineplot(x=x, y=acc_model, marker="o", ax=ax, label="Model", linewidth=2.4)
    sns.lineplot(x=x, y=acc_persist, marker="o", ax=ax, label="Persistence", linewidth=1.8)
    sns.lineplot(x=x, y=acc_random, marker="o", ax=ax, label="RandomMean", linewidth=1.8)
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel("ACC")
    ax.grid(alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_curve(x, y, ylabel, path):
    fig, ax = plt.subplots(figsize=(10,4.5))
    sns.lineplot(x=x, y=y, marker="o", ax=ax, linewidth=2.4)
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_per_channel(leads, mat, names, path, ylabel):
    if mat.ndim != 2:
        print("[WARN] per-channel matrix shape invalid:", mat.shape)
        return
    n_ch = mat.shape[1]
    if len(names) != n_ch:
        names = [f"ch_{i}" for i in range(n_ch)]
    fig, ax = plt.subplots(figsize=(11,6))
    for i in range(n_ch):
        ax.plot(leads, mat[:, i], marker="o", linewidth=1.6, label=names[i])
    ax.set_xlabel("Lead Time (days)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lead_steps", type=int, default=15)
    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--history_steps", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = json.load(f)

    from train import MiniFuXiDataset as FuXiDataset
    from fuxi import FuXiModel

    history_steps = args.history_steps or int(cfg.get("history_steps", 1))
    model = FuXiModel(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        embed_dim=int(cfg["embed_dim"]),
        encoder_dims=_parse_tuple(cfg["encoder_dims"]),
        swin_depths=_parse_tuple(cfg["swin_depths"]),
        swin_heads=_parse_tuple(cfg["swin_heads"]),
        swin_window_size=int(cfg["swin_window_size"]),
        drop_path_rate=float(cfg["drop_path_rate"]),
        input_height=int(cfg["input_height"]),
        input_width=int(cfg["input_width"]),
    ).to(device)

    raw_state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(_unwrap_state(raw_state.get("model_state", raw_state)))
    model.eval()

    dataset = FuXiDataset(path=args.data_path, history_steps=history_steps, mean=None, std=None)

    evaluate(
        model=model,
        dataset=dataset,
        device=device,
        outdir=args.outdir,
        lead_steps=args.lead_steps,
        num_samples=args.num_samples,
        history_steps=history_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
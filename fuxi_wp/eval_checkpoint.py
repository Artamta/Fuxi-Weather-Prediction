import argparse
import torch
import xarray as xr

from traning import MiniFuXiDataset
from fuxi import FuXiTinyBackbone


def parse_args():
    parser = argparse.ArgumentParser("Evaluate FuXi checkpoint and export predictions.")
    parser.add_argument("--data", default="mini_era5_sample.nc")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument("--output", default="predictions.nc")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = MiniFuXiDataset(args.data, history_steps=args.history, target_step=args.target_offset)
    spatial_shape = tuple(dataset.data.shape[-2:])
    channels = dataset.data.shape[1]

    model = FuXiTinyBackbone(
        in_channels=channels,
        output_channels=channels,
        input_shape=spatial_shape,
        window_size=2,
        depths=(3, 3),
        num_heads=(4, 4),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    history, target = dataset[0]
    history = history.unsqueeze(0).to(device)
    pred = model(history).squeeze(0).cpu()

    mean = dataset.mean
    std = dataset.std

    pred_denorm = pred * std + mean
    target_denorm = target * std + mean

    ds_pred = xr.Dataset(
        data_vars={
            "prediction": (("channel", "lat", "lon"), pred_denorm.numpy()),
            "target": (("channel", "lat", "lon"), target_denorm.numpy()),
        }
    )
    ds_pred.to_netcdf(args.output)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
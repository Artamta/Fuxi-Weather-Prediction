import argparse
import numpy as np
import xarray as xr


VAR_NAMES = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
]


def parse_args():
    parser = argparse.ArgumentParser("Compute per-variable MAE/RMSE from predictions.nc")
    parser.add_argument("--file", default="predictions.nc")
    return parser.parse_args()


def grouped_metrics(diff):
    level_vars = diff[:65].reshape(5, 13, *diff.shape[-2:])
    surface_vars = diff[65:].reshape(5, 1, *diff.shape[-2:])
    blocks = np.concatenate([level_vars, surface_vars], axis=1)
    mae = blocks.mean(axis=(1, 2, 3))
    rmse = np.sqrt((blocks ** 2).mean(axis=(1, 2, 3)))
    return mae, rmse


def main():
    args = parse_args()
    ds = xr.open_dataset(args.file)

    pred = ds["prediction"].values
    target = ds["target"].values
    diff = pred - target

    channel_mae = np.mean(np.abs(diff), axis=(1, 2))
    channel_rmse = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))

    var_mae, var_rmse = grouped_metrics(diff)

    print("Channel | MAE | RMSE")
    for idx, (m, r) in enumerate(zip(channel_mae, channel_rmse)):
        print(f"{idx:7d} | {m:.4f} | {r:.4f}")

    print("\nVariable           | MAE     | RMSE")
    for name, m, r in zip(VAR_NAMES, var_mae, var_rmse):
        print(f"{name:18s} | {m:7.4f} | {r:7.4f}")

    print(f"\nOverall MAE (channel): {channel_mae.mean():.4f}")
    print(f"Overall RMSE (channel): {channel_rmse.mean():.4f}")
    print(f"Overall MAE (variable): {var_mae.mean():.4f}")
    print(f"Overall RMSE (variable): {var_rmse.mean():.4f}")

    ds.close()


if __name__ == "__main__":
    main()
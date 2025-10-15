import argparse
import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser("Compute per-channel MAE and RMSE from predictions.nc")
    parser.add_argument("--file", default="predictions.nc", help="NetCDF with 'prediction' and 'target'")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = xr.open_dataset(args.file)

    pred = ds["prediction"].values  # (channel, lat, lon)
    target = ds["target"].values

    diff = pred - target
    mae = np.mean(np.abs(diff), axis=(1, 2))
    rmse = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))

    print("Channel | MAE | RMSE")
    for idx, (m, r) in enumerate(zip(mae, rmse)):
        print(f"{idx:7d} | {m:.4f} | {r:.4f}")

    print(f"\nOverall MAE: {mae.mean():.4f}")
    print(f"Overall RMSE: {rmse.mean():.4f}")

    ds.close()


if __name__ == "__main__":
    main()
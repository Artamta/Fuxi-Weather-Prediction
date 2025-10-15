import argparse
import logging
import os
import sys
from typing import List

import xarray as xr


DEFAULT_PRESSURE_VARS: List[str] = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
DEFAULT_SURFACE_VARS: List[str] = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Extract a tiny ERA5 sample from the Fortytwo Zarr store.")
    parser.add_argument("--root", default="/storage/vishnu", help="Directory containing the Zarr store.")
    parser.add_argument(
        "--store",
        default="1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr",
        help="Zarr store name inside --root.",
    )
    parser.add_argument("--start", default="2000-01-01T00:00", help="Start datetime (inclusive).")
    parser.add_argument("--end", default="2000-01-02T00:00", help="End datetime (inclusive).")
    parser.add_argument("--lat-skip", type=int, default=4, help="Keep every Nth latitude for a tiny grid.")
    parser.add_argument("--lon-skip", type=int, default=4, help="Keep every Nth longitude for a tiny grid.")
    parser.add_argument("--output", default="mini_era5_sample.nc", help="Output NetCDF path.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    store_path = os.path.join(args.root, args.store)
    if not os.path.exists(store_path):
        logging.error("Zarr store not found: %s", store_path)
        sys.exit(1)

    logging.info("Opening Zarr store: %s", store_path)
    try:
        ds = xr.open_zarr(store_path, consolidated=True)
    except Exception as exc:
        logging.exception("Failed to open Zarr store.")
        sys.exit(1)

    requested_vars = DEFAULT_PRESSURE_VARS + DEFAULT_SURFACE_VARS
    missing = [v for v in requested_vars if v not in ds.variables]
    if missing:
        logging.warning("Missing variables: %s", ", ".join(missing))

    available_vars = [v for v in requested_vars if v in ds.variables]
    if not available_vars:
        logging.error("No requested variables available; aborting.")
        sys.exit(1)

    logging.info("Selecting variables: %s", ", ".join(available_vars))
    subset = ds[available_vars].sel(time=slice(args.start, args.end))
    subset = subset.isel(latitude=slice(None, None, args.lat_skip), longitude=slice(None, None, args.lon_skip))

    if "total_precipitation_6hr" in subset.variables:
        subset = subset.rename({"total_precipitation_6hr": "total_precipitation"})

    subset = subset.astype("float32")

    try:
        subset.to_netcdf(args.output)
        logging.info(
            "Saved sample: %s (time=%d, lat=%d, lon=%d)",
            args.output,
            subset.dims.get("time", 0),
            subset.dims.get("latitude", 0),
            subset.dims.get("longitude", 0),
        )
    except Exception:
        logging.exception("Failed to write NetCDF.")
        sys.exit(1)


if __name__ == "__main__":
    main()
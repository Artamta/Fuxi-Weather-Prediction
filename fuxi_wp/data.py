import argparse
import logging
import os
import sys
from typing import List

import xarray as xr

PRESSURE_VARS: List[str] = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]

PRESSURE_LEVELS = [
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    50,
]

SURFACE_VARS: List[str] = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Extract a FuXi-style ERA5 mini-sample from the Fortytwo Zarr store.")
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
    except Exception:
        logging.exception("Failed to open Zarr store.")
        sys.exit(1)

    rename_dims = {}
    if "latitude" in ds.dims:
        rename_dims["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename_dims["longitude"] = "lon"
    ds = ds.rename(rename_dims)

    missing_pressure = [v for v in PRESSURE_VARS if v not in ds]
    missing_surface = [v for v in SURFACE_VARS if v not in ds]
    if missing_pressure or missing_surface:
        logging.warning("Missing vars -> pressure:%s surface:%s", missing_pressure, missing_surface)

    present_vars = [v for v in PRESSURE_VARS + SURFACE_VARS if v in ds]
    if not present_vars:
        logging.error("No requested variables available; aborting.")
        sys.exit(1)

    subset = ds[present_vars].sel(time=slice(args.start, args.end))

    if "level" in subset.dims:
        available_levels = set(subset.coords["level"].values.tolist())
        desired_levels = [lvl for lvl in PRESSURE_LEVELS if lvl in available_levels]
        if len(desired_levels) != len(PRESSURE_LEVELS):
            logging.warning("Dropping unavailable levels: %s", sorted(set(PRESSURE_LEVELS) - available_levels))
        subset = subset.sel(level=desired_levels)

    subset = subset.isel(lat=slice(None, None, args.lat_skip), lon=slice(None, None, args.lon_skip))
    subset = subset.astype("float32")

    try:
        subset.to_netcdf(args.output)
        logging.info(
            "Saved sample: %s (time=%d, level=%d, lat=%d, lon=%d, vars=%d)",
            args.output,
            subset.sizes.get("time", 0),
            subset.sizes.get("level", 0),
            subset.sizes.get("lat", 0),
            subset.sizes.get("lon", 0),
            len(subset.data_vars),
        )
    except Exception:
        logging.exception("Failed to write NetCDF.")
        sys.exit(1)


if __name__ == "__main__":
    main()
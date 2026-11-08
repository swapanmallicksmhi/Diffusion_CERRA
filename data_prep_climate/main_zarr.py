import os
import xarray as xr
import numpy as np
from helper import find_matching_files, regrid_to_highres, crop_data
from helper import plot_all, save_to_zarr, calculate_summary_stats
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run climate data processing")

    parser.add_argument("--grid_size", type=int, required=True,
                        help="Size of the x grid")

    #parser.add_argument("--era5land_dir", type=str, required=True,
    #                    help="Path to NetCDF data directory")

    parser.add_argument("--lowres_dir", type=str, required=True,
                        help="Path to Low Resolution NetCDF data directory")

    parser.add_argument("--highres_dir", type=str, required=True,
                        help="Path to High Resolution NetCDF data directory")

    parser.add_argument("--output_dir", type=str, default="../output_plots",
                        help="Output directory for plots")

    parser.add_argument("--zarr_path", type=str, default="../output_data.zarr",
                        help="Output Zarr store")

    parser.add_argument("--lowres_variables", nargs="+", required=True,
                        help="Low Resolution Variables to process")

    parser.add_argument("--highres_variables", nargs="+", required=True,
                        help="High Resolution Variables to process")

    return parser.parse_args()



def preprocess_dataset(
    ds: xr.Dataset, drop_vars: list = None
) -> xr.Dataset:
    """Ensure lat/lon are set as coordinates and drop unwanted dimensions/variables."""
    if drop_vars is None:
        drop_vars = ["plev", "height"]

    coords_to_set = [c for c in ["latitude", "longitude"] if c in ds]
    if coords_to_set:
        ds = ds.set_coords(coords_to_set)

    vars_to_drop = [
        v for v in drop_vars if v in ds.coords or v in ds.data_vars
    ]
    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    return ds


def process_highres_dataset(highres_path: str) -> xr.Dataset:
    """Load and format the high-resolution (3-km) reference dataset."""
    ds = xr.open_dataset(highres_path, chunks={"time": 1})
    ds = preprocess_dataset(ds, drop_vars=["height"])

    print(f"High-resolution grid: x={ds.sizes['x']}, y={ds.sizes['y']}")
    print(f"Number of times: {ds.sizes['time']}")

    if "tas" not in ds.data_vars:
        raise ValueError("tas variable not found in 3-km reference file")

    ds = ds[["tas"]].rename({"tas": "tas_3km"})
    ds["tas_3km"].attrs.update(
        {"source_resolution": "3 km native", "original_variable": "tas"}
    )
    return ds


def process_single_lowres_file(
    lowres_path: str,
    ds_highres: xr.Dataset,
    lr_variables: list,
    regridder=None,
    source_lat=None,
    source_lon=None,
) -> tuple[xr.Dataset, object, object, object]:
    """Process, validate, and regrid a single low-resolution file onto high-res grid."""
    print(f"\nProcessing: {os.path.basename(lowres_path)}")

    ds_c = xr.open_dataset(lowres_path, chunks={"time": 1})
    ds_c = preprocess_dataset(ds_c, drop_vars=["plev", "height"])

    available_vars = [v for v in lr_variables if v in ds_c.data_vars]
    if not available_vars:
        raise ValueError("No requested LR_VARIABLES found in dataset")

    ds_c = ds_c[available_vars]

    print(f"Variables found: {available_vars}")
    print(f"Original grid size: x={ds_c.sizes['x']}, y={ds_c.sizes['y']}")
    print(f"Number of times: {ds_c.sizes['time']}")

    # Validation against high-res reference
    if ds_c.sizes["time"] != ds_highres.sizes["time"]:
        raise ValueError(
            "12-km and 3-km datasets have different time steps count"
        )
    if not np.array_equal(ds_c.time.values, ds_highres.time.values):
        raise ValueError("12-km and 3-km time coordinates do not match")

    # Regrid onto high-res grid
    ds_interp, regridder, source_lat, source_lon = regrid_to_highres(
        ds_c, ds_highres, regridder, source_lat, source_lon
    )

    # Label variable metadata
    if "tas" in ds_interp.data_vars:
        ds_interp = ds_interp.rename({"tas": "tas_12km_interp"})
        ds_interp["tas_12km_interp"].attrs.update(
            {
                "source_resolution": "12 km interpolated to 3 km",
                "original_variable": "tas",
            }
        )

    for var in ["ta500", "ta700", "ta850", "ta950"]:
        if var in ds_interp.data_vars:
            ds_interp[var].attrs[
                "source_resolution"
            ] = "12 km interpolated to 3 km"

    print(
        f"Interpolated grid size: x={ds_interp.sizes['x']}, y={ds_interp.sizes['y']}"
    )
    print(f"Interpolated LR variables: {list(ds_interp.data_vars)}")

    return ds_interp, regridder, source_lat, source_lon


def process_lowres_datasets(
    lowres_dir: str,
    lr_res: str,
    lr_freq: str,
    lr_variables: list,
    ds_highres: xr.Dataset,
) -> list[xr.Dataset]:
    """Find, validate, and regrid all matching low-resolution datasets."""
    pairs = find_matching_files(lowres_dir, lr_res, lr_freq, lr_variables)
    if not pairs:
        print("No matching 12-km files found.")
        return []

    interpolated_datasets = []
    regridder, source_lat, source_lon = None, None, None

    for lowres_path in pairs:
        try:
            ds_interp, regridder, source_lat, source_lon = (
                process_single_lowres_file(
                    lowres_path,
                    ds_highres,
                    lr_variables,
                    regridder,
                    source_lat,
                    source_lon,
                )
            )
            interpolated_datasets.append(ds_interp)
        except Exception as e:
            print(f"Error processing {os.path.basename(lowres_path)}: {e}")

    return interpolated_datasets


def generate_plots(
    ds: xr.Dataset, output_dir: Path, target_variables: list
) -> None:
    """Generate diagnostic plots for specified variables over all timesteps."""
    available_vars = [v for v in target_variables if v in ds.data_vars]

    for t in range(ds.sizes["time"]):
        time_val = ds.time.isel(time=t).values
        time_str_file = (
            str(time_val)[:16]
            .replace(":", "")
            .replace("-", "")
            .replace("T", "_")
        )
        plot_title = f"(Time: {str(time_val)[:16]})"
        idx = {"time": t}

        for var in available_vars:
            save_full_path = output_dir / f"plot_{var}_{time_str_file}.png"
            # plot_all(
            #     ds,
            #     idx,
            #     varname=var,
            #     title_suffix=plot_title,
            #     save_path=str(save_full_path)
            # )


def main():
    args = parse_args()

    GRID_SIZE = args.grid_size
    LR_DIR = args.lowres_dir
    HR_DIR = args.highres_dir
    OUTPUT_DIR = args.output_dir
    zarr_path = args.zarr_path
    LR_VARIABLES = args.lowres_variables
    HR_VARIABLES = args.highres_variables
    LR_RESOLUTION, LR_FREQUENCY = '12km', '6hr'
    HR_RESOLUTION, HR_FREQUENCY = '3km', '6hr'

    # Paths & parameters
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80 + "\nVariables requested\n" + "=" * 80)
    for var in LR_VARIABLES:
        print(var)

    # 1. Load High-Resolution Reference Dataset (3-km)
    highres_files = find_matching_files(
        HR_DIR, HR_RESOLUTION, HR_FREQUENCY, HR_VARIABLES
    )
    if not highres_files:
        raise ValueError("3-km reference file not found")

    print(f"\nHigh-resolution reference file:\n{highres_files}")
    ds_highres = process_highres_dataset(highres_files[0])

    # 2. Process & Regrid Low-Resolution Datasets (12-km)
    lr_datasets = process_lowres_datasets(
        LR_DIR, LR_RESOLUTION, LR_FREQUENCY, LR_VARIABLES, ds_highres
    )

    # 3. Merge Datasets on the High-Res Grid
    datasets_to_merge = [ds_highres] + lr_datasets
    print("\nMerging all variables on the 3-km grid...")
    ds_c = xr.merge(datasets_to_merge, compat="override", join="exact")
    ds_c_sub = crop_data(ds_c, GRID_SIZE)

    print("\nFinal dataset:\n", ds_c_sub)
    print("=" * 80 + "\nFinal variables\n" + "=" * 80)
    for var in ds_c_sub.data_vars:
        print(var)

    # 4. Save to Zarr and Plot
    final_variables = [
        "ta500",
        "ta700",
        "ta850",
        "ta950",
        "tas_3km",
        "tas_12km_interp",
    ]
    generate_plots(ds_c_sub, output_dir, final_variables)

    # Rename coordinates/dimensions before saving to Zarr
    ds_c_sub = ds_c_sub.rename({
        'time': 'valid_time',
        'lat': 'latitude',
        'lon': 'longitude'
    })

    save_to_zarr(zarr_path=zarr_path, ds_all=ds_c_sub)
    print(f"Saved Zarr dataset to: {zarr_path}")
    calculate_summary_stats(zarr_path)


if __name__ == "__main__":
    main()


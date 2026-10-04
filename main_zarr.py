import os
import xarray as xr
import numpy as np
from helper import find_matching_files, find_highres_file, regrid_to_highres, crop_era5land
from helper import plot_all, save_to_zarr, calculate_summary_stats
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run climate data processing")

    parser.add_argument("--grid_size", type=int, required=True,
                        help="Size of the x grid")

    parser.add_argument("--era5land_dir", type=str, required=True,
                        help="Path to NetCDF data directory")

    parser.add_argument("--output_dir", type=str, default="../output_plots",
                        help="Output directory for plots")

    parser.add_argument("--zarr_path", type=str, default="../output_data.zarr",
                        help="Output Zarr store")

    parser.add_argument("--variables", nargs="+", required=True,
                        help="Variables to process")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    GRID_SIZE = args.grid_size
    ERA_DIR = args.era5land_dir
    OUTPUT_DIR = args.output_dir
    zarr_path = args.zarr_path
    variables = args.variables

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("=" * 80)
    print("Variables requested")
    print("=" * 80)

    for var in variables:
        print(var)

    highres_path = find_highres_file(ERA_DIR)

    if highres_path is None:
        raise ValueError("3-km reference file not found")

    print("\nHigh-resolution reference file:")
    print(highres_path)

    ds_highres = xr.open_dataset(highres_path, chunks={'time': 1})

    if 'lat' in ds_highres:
        ds_highres = ds_highres.set_coords('lat')

    if 'lon' in ds_highres:
        ds_highres = ds_highres.set_coords('lon')

    print(f"High-resolution grid: x={ds_highres.sizes['x']}, y={ds_highres.sizes['y']}")
    print(f"Number of times: {ds_highres.sizes['time']}")

    if 'tas' not in ds_highres.data_vars:
        raise ValueError("tas variable not found in 3-km reference file")

    ds_highres = ds_highres[['tas']]
    ds_highres = ds_highres.rename({'tas': 'tas_3km'})

    if 'height' in ds_highres.coords:
        ds_highres = ds_highres.drop_vars('height')

    if 'height' in ds_highres.data_vars:
        ds_highres = ds_highres.drop_vars('height')

    ds_highres['tas_3km'].attrs['source_resolution'] = '3 km native'
    ds_highres['tas_3km'].attrs['original_variable'] = 'tas'

    pairs = find_matching_files(ERA_DIR, variables)

    if not pairs:
        print("No matching 12-km files found.")
        datasets = [ds_highres]
    else:
        datasets = [ds_highres]
        regridder = None
        source_lat = None
        source_lon = None

        for era5land_path in pairs:

            print(f"\nProcessing: {os.path.basename(era5land_path)}")

            try:
                ds_c = xr.open_dataset(era5land_path, chunks={'time': 1})

                if 'lat' in ds_c:
                    ds_c = ds_c.set_coords('lat')

                if 'lon' in ds_c:
                    ds_c = ds_c.set_coords('lon')

                available_vars = [v for v in variables if v in ds_c.data_vars]

                if not available_vars:
                    raise ValueError("No requested variables found in dataset")

                ds_c = ds_c[available_vars]

                if 'plev' in ds_c.coords:
                    ds_c = ds_c.drop_vars('plev')

                if 'plev' in ds_c.data_vars:
                    ds_c = ds_c.drop_vars('plev')

                if 'height' in ds_c.coords:
                    ds_c = ds_c.drop_vars('height')

                if 'height' in ds_c.data_vars:
                    ds_c = ds_c.drop_vars('height')

                print(f"Variables found: {available_vars}")
                print(f"Original grid size: x={ds_c.sizes['x']}, y={ds_c.sizes['y']}")
                print(f"Number of times: {ds_c.sizes['time']}")

                if ds_c.sizes['time'] != ds_highres.sizes['time']:
                    raise ValueError("12-km and 3-km datasets have different number of time steps")

                if not np.array_equal(ds_c.time.values, ds_highres.time.values):
                    raise ValueError("12-km and 3-km time coordinates do not match")

                ds_interp, regridder, source_lat, source_lon = regrid_to_highres(
                    ds_c,
                    ds_highres,
                    regridder,
                    source_lat,
                    source_lon
                )

                if 'tas' in ds_interp.data_vars:
                    ds_interp = ds_interp.rename({'tas': 'tas_12km_interp'})
                    ds_interp['tas_12km_interp'].attrs['source_resolution'] = '12 km interpolated to 3 km'
                    ds_interp['tas_12km_interp'].attrs['original_variable'] = 'tas'

                for var in ['ta500', 'ta700', 'ta850', 'ta950']:
                    if var in ds_interp.data_vars:
                        ds_interp[var].attrs['source_resolution'] = '12 km interpolated to 3 km'

                print(f"Interpolated grid size: x={ds_interp.sizes['x']}, y={ds_interp.sizes['y']}")
                print(f"Interpolated variables: {list(ds_interp.data_vars)}")

                datasets.append(ds_interp)

            except Exception as e:
                print(f"Error processing {os.path.basename(era5land_path)}: {e}")

    if datasets:

        print("\nMerging all variables on the 3-km grid...")

        ds_c = xr.merge(datasets, compat='override', join='exact')

        ds_c_sub = crop_era5land(
            ds_c,
            GRID_SIZE
        )

        print("\nFinal dataset:")
        print(ds_c_sub)

        print("=" * 80)
        print("Final variables")
        print("=" * 80)

        for var in ds_c_sub.data_vars:
            print(var)

        n_times = ds_c_sub.sizes['time']

        final_variables = [
            "ta500",
            "ta700",
            "ta850",
            "ta950",
            "tas_3km",
            "tas_12km_interp",
        ]

        available_final_vars = [v for v in final_variables if v in ds_c_sub.data_vars]

        for t in range(n_times):

            time_val = ds_c_sub.time.isel(time=t).values
            time_str_file = str(time_val)[:16].replace(':', '').replace('-', '').replace('T', '_')

            idx = {'time': t}

            plot_title = f"(Time: {str(time_val)[:16]})"

            for var in available_final_vars:

                filename = f"plot_{var}_{time_str_file}.png"
                save_full_path = os.path.join(OUTPUT_DIR, filename)

                #plot_all(
                #    ds_c_sub,
                #    idx,
                #    varname=var,
                #    title_suffix=plot_title,
                #    save_path=save_full_path)

        save_to_zarr(zarr_path=zarr_path, ds_era5land=ds_c_sub)
        print(zarr_path)
        calculate_summary_stats(zarr_path)

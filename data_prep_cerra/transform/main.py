import os
import random
import xarray as xr
from helpers import find_matching_files, crop_cerra, era5_to_cerra
from helpers import plot_all, plot_statistics, save_to_zarr, calculate_summary_stats
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run CERRA/ERA5 processing")
    parser.add_argument("--grid-size", type=int, required=True,
                        help="Size of the gris, 128 or 256")
    parser.add_argument("--cerra-dir", type=str, required=True,
                        help="Path to CERRA data directory")
    parser.add_argument("--era5-dir", type=str, required=True,
                        help="Path to ERA5-EDA data directory")
    parser.add_argument("--output-dir", type=str, default="../output_plots",
                        help="Output directory for plots")
    parser.add_argument("--zarr-path", type=str, default="../output_data.zarr",
                        help="Output Zarr store")

    parser.add_argument("--cerra-members", type=bool, default=False,
                        help="Include Cerra members")
    
    parser.add_argument("--era5-members", type=bool, default=False,
                        help="Include Era5 members")

    return parser.parse_args()


if __name__ == "__main__":
    # --- Configuration ---
    args = parse_args()
    GRID_SIZE = args.grid_size
    CERRA_DIR = args.cerra_dir
    ERA5_DIR  = args.era5_dir
    OUTPUT_DIR = args.output_dir
    ZARR_PATH  = args.zarr_path
    
    # Correction: Assigned args.cerra_members to CERRA and args.era5 to ERA5
    SAVE_CERRA_MEMBERS = args.cerra_members 
    SAVE_ERA5_MEMBERS = args.era5_members
     

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    YEARS = [ str(i) for i in range(2015,2025)]
    MONTHS = [f"{i:02d}" for i in range(1, 13)]
    CYCLES = ['0000'] 

    #YEARS = ["2024"]
    #MONTHS = ["10"]
    #CYCLES = ['0000', '0600', '1200', '1800'] 
    
    YEARS = [ str(i) for i in range(2015,2025)]
    MONTHS = [f"{i:02d}" for i in range(1, 13)]
    CYCLES = ['0000'] 
    
   


    LAT_MIN, LAT_MAX = 50, 73
    LON_MIN, LON_MAX = 1, 30
    

    pairs = find_matching_files(CERRA_DIR, ERA5_DIR, YEARS, MONTHS, CYCLES)

    if not pairs:
        print("No matching files found.")
    else:
        for cerra_path, era5_path in pairs:
            print(f"\nProcessing: {os.path.basename(cerra_path)}")
            
            try:
                # Load Data
                ds_c = xr.open_dataset(cerra_path)
                ds_c['longitude'] = ((ds_c.longitude + 180) % 360) - 180 
                
                ds_e = xr.open_dataset(era5_path)
                ds_e['longitude'] = ((ds_e.longitude + 180) % 360) - 180

                # =========================================================
                #  FIX: DROP EXPVER TO PREVENT MERGE CONFLICTS
                # =========================================================
                if 'expver' in ds_e.coords or 'expver' in ds_e.data_vars:
                    try:
                        ds_e = ds_e.drop_vars('expver')
                    except Exception:
                        pass
                
                if 'expver' in ds_c.coords or 'expver' in ds_c.data_vars:
                     ds_c = ds_c.drop_vars('expver')

                # Crop and Interpolate
                ds_c_sub = crop_cerra(ds_c, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_SIZE)
                ds_e_on_c, _ = era5_to_cerra(ds_e, ds_c_sub)
                
                # ====================================================
                # A. CALCULATE STATISTICS
                # ====================================================
                cerra_mean = ds_c_sub.mean(dim='number', keep_attrs=True)
                cerra_std  = ds_c_sub.std(dim='number', keep_attrs=True)
                
                era5_mean  = ds_e_on_c.mean(dim='number', keep_attrs=True)
                era5_std   = ds_e_on_c.std(dim='number', keep_attrs=True)

                # ====================================================
                # B. PLOT STATISTICS
                # ====================================================
                time_val = ds_c_sub.valid_time.isel(valid_time=0).values
                time_str_file = str(time_val)[:16].replace(':', '').replace('-', '').replace('T', '_')
                
                stats_filename = f"stats_{time_str_file}.png"
                stats_save_path = os.path.join(OUTPUT_DIR, stats_filename)
                
                plot_statistics(cerra_mean, cerra_std, era5_mean, era5_std, 
                                title_suffix=str(time_val)[:13], 
                                save_path=stats_save_path)

                # ====================================================
                # C. RANDOM MEMBER PLOT
                # ====================================================
                n_members = ds_c_sub.sizes['number']
                n_times = ds_c_sub.sizes['valid_time']
                rand_member = random.randint(0, n_members - 1)
                rand_time_idx = random.randint(0, n_times - 1)
                
                idx = {'number': rand_member, 'valid_time': rand_time_idx}
                filename = f"plot_{time_str_file}_mem{rand_member:02d}.png"
                save_full_path = os.path.join(OUTPUT_DIR, filename)
                plot_title = f"(Mb:{rand_member}, {str(time_val)[:16]})"
                
                plot_all(ds_e_on_c, ds_c_sub, idx, title_suffix=plot_title, save_path=save_full_path)

                # ====================================================
                # D. INCREMENTAL ZARR SAVING (Refactored)
                # ====================================================
                save_to_zarr(
                    zarr_path=ZARR_PATH,
                    cerra_mean=cerra_mean,
                    cerra_std=cerra_std,
                    era5_mean=era5_mean,
                    era5_std=era5_std,
                    ds_cerra=ds_c_sub,
                    ds_era5=ds_e_on_c,
                    save_cerra_members=SAVE_CERRA_MEMBERS,
                    save_era5_members=SAVE_ERA5_MEMBERS
                )
                
                ds_c.close()
                ds_e.close()

                
            except Exception as e:
                print(f"Error processing {os.path.basename(cerra_path)}: {e}")

        calculate_summary_stats(ZARR_PATH)

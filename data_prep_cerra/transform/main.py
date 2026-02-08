import os
import glob
import random
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from helpers import find_matching_files, open_ds, crop_cerra , era5_to_cerra
from helpers import plot_all


# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    CERRA_DIR = '/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/CERRA_DATA/'
    ERA5_DIR  = '/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/ERA5EDA_DATA/'
    OUTPUT_DIR = './output_plots' # Directory to save images

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    YEARS = ["2024"]
    MONTHS = ["10"]
    CYCLES = ['0000', '0600', '1200', '1800'] 
    
    # Area definition
    LAT_MIN, LAT_MAX = 50, 73
    LON_MIN, LON_MAX = 1, 30
    GRID_SIZE = 128

    # 1. Find Files
    pairs = find_matching_files(CERRA_DIR, ERA5_DIR, YEARS, MONTHS, CYCLES)

    if not pairs:
        print("No matching files found.")
    else:
        # 2. Loop through every found pair
        for cerra_path, era5_path in pairs:
            print(f"\nProcessing: {os.path.basename(cerra_path)}")
            
            try:
                # Load Data
                ds_c = xr.open_dataset(cerra_path)
                ds_c['longitude'] = ((ds_c.longitude + 180) % 360) - 180 # Normalize
                
                ds_e = xr.open_dataset(era5_path)
                ds_e['longitude'] = ((ds_e.longitude + 180) % 360) - 180 # Normalize

                # Crop and Interpolate
                ds_c_sub = crop_cerra(ds_c, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_SIZE)
                ds_e_on_c, _ = era5_to_cerra(ds_e, ds_c_sub)
                
                # --- RANDOM SELECTION LOGIC ---
                n_members = ds_c_sub.sizes['number']
                n_times = ds_c_sub.sizes['valid_time']
                
                # Pick random indices
                rand_member = random.randint(0, n_members - 1)
                rand_time_idx = random.randint(0, n_times - 1)
                
                idx = {'number': rand_member, 'valid_time': rand_time_idx}
                
                # Get string info for filename
                time_val = ds_c_sub.valid_time.isel(valid_time=rand_time_idx).values
                # Format: YYYYMMDD_HHMM
                time_str_file = str(time_val)[:16].replace(':', '').replace('-', '').replace('T', '_')
                cycle_str = os.path.basename(cerra_path).split('_')[-1].replace('.nc', '')
                
                # Generate Filename
                # Example: plot_20241015_0600_cycle0000_mem05.png
                filename = f"plot_{time_str_file}_cycle{cycle_str}_mem{rand_member:02d}.png"
                save_full_path = os.path.join(OUTPUT_DIR, filename)
                
                plot_title = f"(Mb:{rand_member}, {str(time_val)[:16]})"
                
                # Plot and Save
                plot_all(ds_e_on_c, ds_c_sub, idx, title_suffix=plot_title, save_path=save_full_path)
                
                ds_c.close()
                ds_e.close()
                
            except Exception as e:
                print(f"Error processing {os.path.basename(cerra_path)}: {e}")
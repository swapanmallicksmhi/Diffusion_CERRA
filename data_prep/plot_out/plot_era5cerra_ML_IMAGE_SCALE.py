# -*- coding: utf-8 -*-
"""
Author      : Swapan Mallick (SMHI)
Created on  : 2025-12-15

Description
-----------
Plot NetCDF ERA5-EDA and CERRA t2m mean and standard deviation for ML Input
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import argparse

# -----------------------------
# Input filenames
# -----------------------------
#filename_ce = "cerra_t2m_points_2020010100.nc"
#filename_er = "era5eda_t2m_points_2020010100.nc"
# -----------------------------
# Parse input arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Plot CERRA and ERA5-EDA t2m NetCDF data")
parser.add_argument('--cerra_file', type=str, required=True, help='Path to CERRA NetCDF file')
parser.add_argument('--era_file', type=str, required=True, help='Path to ERA5-EDA NetCDF file')
args = parser.parse_args()

filename_ce = args.cerra_file
filename_er = args.era_file
#
start_time = time.time()

def extract_timestamp(filename):
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    return name_without_ext[-10:]

timestamp = extract_timestamp(filename_er)
print(f"Timestamp extracted: {timestamp}")

# -----------------------------
# Domain
# -----------------------------
DOMAIN = {
    'lon_min': -11.0,
    'lon_max': 30.0,
    'lat_min': 35.0,
    'lat_max': 72.0
}

print(f"Using fixed domain: Lon {DOMAIN['lon_min']} to {DOMAIN['lon_max']}, "
      f"Lat {DOMAIN['lat_min']} to {DOMAIN['lat_max']}")

# -----------------------------
# Load data
# -----------------------------
ds_ce = xr.open_dataset(filename_ce, chunks={'points': 1000})
ds_er = xr.open_dataset(filename_er, chunks={'points': 1000})

data_mean_ce = ds_ce['t2m_mean'].load()
data_sd_ce = ds_ce['t2m_std'].load()
data_mean_er = ds_er['t2m_mean'].load()
data_sd_er = ds_er['t2m_std'].load()

# -----------------------------
# Helper: extract valid data within domain
# -----------------------------
def get_valid_data_within_domain(data_array, domain):
    values = data_array.values
    lats = data_array.latitude.values
    lons = data_array.longitude.values

    valid_mask = ~np.isnan(values)
    domain_mask = (
        (lats >= domain['lat_min']) & (lats <= domain['lat_max']) &
        (lons >= domain['lon_min']) & (lons <= domain['lon_max'])
    )
    combined_mask = valid_mask & domain_mask
    if np.any(combined_mask):
        return lons[combined_mask], lats[combined_mask], values[combined_mask]
    return None, None, None

cerra_mean = get_valid_data_within_domain(data_mean_ce, DOMAIN)
cerra_sd = get_valid_data_within_domain(data_sd_ce, DOMAIN)
era_mean = get_valid_data_within_domain(data_mean_er, DOMAIN)
era_sd = get_valid_data_within_domain(data_sd_er, DOMAIN)

# -----------------------------
# Plot parameters
# -----------------------------
PLOT_PARAMS = {
    'mean': {'vmin': 220, 'vmax': 300, 'cmap': 'jet', 'label': 'Temperature (K)'},
    'sd': {'vmin': 0, 'vmax': 3, 'cmap': 'jet', 'label': 'Standard Deviation (K)'}
}

def create_scatter_plot(lons, lats, values, plot_type, title, domain, output_filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    if lons is not None and len(lons) > 0:
        params = PLOT_PARAMS[plot_type]
        scatter = ax.scatter(lons, lats, c=values, cmap=params['cmap'], s=5, alpha=0.7,
                             vmin=params['vmin'], vmax=params['vmax'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.set_xlim(domain['lon_min'], domain['lon_max'])
        ax.set_ylim(domain['lat_min'], domain['lat_max'])
        ax.grid(True, alpha=0.3, linestyle='--')
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, label=params['label'])
        cbar.solids.set(alpha=1)
    else:
        ax.set_title(f'{title} - No Data')
        ax.set_xlim(domain['lon_min'], domain['lon_max'])
        ax.set_ylim(domain['lat_min'], domain['lat_max'])
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_filename}")

# -----------------------------
# Create separate PNGs
# -----------------------------
create_scatter_plot(*cerra_mean, 'mean', f'CERRA t2m Mean {timestamp}', DOMAIN,
                    f"cerra_MEAN_{timestamp}.png")
create_scatter_plot(*era_mean, 'mean', f'ERA5-EDA t2m Mean {timestamp}', DOMAIN,
                    f"era5eda_MEAN_ERA5_{timestamp}.png")
create_scatter_plot(*cerra_sd, 'sd', f'CERRA t2m SD {timestamp}', DOMAIN,
                    f"cerra_SD_{timestamp}.png")
create_scatter_plot(*era_sd, 'sd', f'ERA5-EDA t2m SD {timestamp}', DOMAIN,
                    f"era5eda_SD_{timestamp}.png")

# -----------------------------
# Close datasets and report runtime
# -----------------------------
ds_ce.close()
ds_er.close()
end_time = time.time()
print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

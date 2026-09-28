#!/bin/bash

#set -euo pipefail
set -x

# =================================================
# USER-DEFINED VARIABLES (DEFINE FIRST)
# =================================================

# Input / Output
#outputdir="${PWD}/PLOTS"
zarrinput="/cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/CERRA5.5km_JAN2024.zarr"
outputdir="/cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/STATION_PLOTS_JAN"
#zarrinput="/cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/CERRA5.5km_JUN2024.zarr"
#outputdir="/cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/STATION_PLOTS_JUN"
CSVIN_JAN="/cfs/klemming/home/s/swapanm/Diffusion_CERRA/diffusion_May26/VERIFICATION_MARCH2026/t2m_jan2024.csv"
CSVIN_JUN="/cfs/klemming/home/s/swapanm/Diffusion_CERRA/diffusion_May26/VERIFICATION_MARCH2026/t2m_jun2024.csv"
#zarrinput="/cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/CERRA5.5km_JAN2024.zarr"
#outputdir="/cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/STATION_PLOTS_JAN"
# =================================================
# PREPARE OUTPUT DIRECTORY
# =================================================
#rm -rf "${outputdir}"
mkdir -p "${outputdir}"

# Geographic domain
latmin=54.0
#latmin=49.8
latmax=71.0
lonmin=4.0
#lonmin=-1.1
lonmax=25.0
#lonmax=30.0

# Variables to plot (DEFINE AS LIST FIRST)
# ----Variables for CERRA 5.5 Km 
variables=(
    t2m
)
#----Variables for CERRA-MEAN and ERA5-EDA
#variables=(
#    t2m_cerra_mean
#    t2m_era5_mean
#    t2m_cerra_std
#    t2m_era5_std
#)

# =================================================
# LOAD ENVIRONMENT
# =================================================
ml PDC
ml miniconda3/25.3.1-1-cpeGNU-24.11
source activate torch_rocm

# =================================================
# RUN PYTHON SCRIPT
# =================================================
#python verification_statistics.py \
#python verification_statistics_all.py \
python station_plot.py \
  --zarr-path /cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/CERRA5.5km_JAN2024.zarr \
  --output-dir /cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/STATION_PLOTS_JAN \
  --lat-min 54.0 \
  --lat-max 71.0 \
  --lon-min 4.0 \
  --lon-max 25.0 \
  --csv-file /cfs/klemming/home/s/swapanm/Diffusion_CERRA/diffusion_May26/VERIFICATION_MARCH2026/t2m_jan2024.csv \
  --npy-pattern "/cfs/klemming/scratch/y/yarongc/SDM_seasonalStrongCond/cfg_1/January/*/best_overall/best_t2m_cerra_mean*.npy"

exit

python station_plot.py \
    --zarr-path "${zarrinput}" \
    --output-dir "${outputdir}" \
    --lat-min "${latmin}" \
    --lat-max "${latmax}" \
    --lon-min "${lonmin}" \
    --lon-max "${lonmax}"  \
    --csv-file "${CSVIN_JAN}"

echo "Job completed successfully."

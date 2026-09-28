#!/bin/bash

#set -euo pipefail
set -x

# =================================================
# USER-DEFINED VARIABLES (DEFINE FIRST)
# =================================================

# Input / Output
#outputdir="${PWD}/PLOTS"
outputdir="/cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/STATION_PLOTS_JAN_EXTRA"
# =================================================
# PREPARE OUTPUT DIRECTORY
# =================================================
#rm -rf "${outputdir}"
mkdir -p "${outputdir}"

# Geographic domain
latmin=54.0
latmax=71.0
lonmin=4.0
lonmax=25.0

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
python statistics_csv.py \
    --output-dir "${outputdir}" \
    --csv-file /cfs/klemming/home/s/swapanm/Diffusion_CERRA/diffusion_May26/VERIFICATION_MARCH2026/PLOT_NPY/station_timeseries_values_12UTC.csv

echo "Job completed successfully."

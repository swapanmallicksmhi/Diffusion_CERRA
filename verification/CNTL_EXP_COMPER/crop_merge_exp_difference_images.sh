#!/bin/bash
#-------------------------------------------------
##SBATCH --job-name=zarr_plots
##SBATCH --output=zarr_plots.out
##SBATCH --error=zarr_plots.out
##SBATCH --nodes=1
##SBATCH --mem=60G
##SBATCH --time=36:00:00
#-------------------------------------------------
set -x
zarrinput="/cfs/klemming/scratch/s/swapanm/all_zarr-out_2024.zarr"
inputdir="/cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/ALL_EXP_DIFF_JAN2024"
outputdir="${inputdir}/MARGE"
SEASON="January"
mkdir -p "${outputdir}"
latmin=54.0
latmax=71.0
lonmin=4.0
lonmax=25.0

ml PDC
ml miniconda3/25.3.1-1-cpeGNU-24.11
source activate torch_rocm

declare -A EXP_PATHS
python crop_merge_exp_difference_images_MEAN.py \
  --input-dir /cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/ALL_EXP_DIFF_JAN2024 \
  --output-file /cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/ALL_EXP_DIFF_JAN2024/MARGE/MEAN_EXP_DIFF_CROP_4DAYS_ALL.png \
  --dates 20240101,20240102,20240103,20240104 \
  --crop-left 10 \
  --crop-upper 1 \
  --crop-right 1080 \
  --crop-lower 800
#  --no-crop
#  STD---
python crop_merge_exp_difference_images_STD.py \
  --input-dir /cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/ALL_EXP_DIFF_JAN2024 \
  --output-file /cfs/klemming/scratch/s/swapanm/VERIFICATION_MARCH2026/ALL_EXP_DIFF_JAN2024/MARGE/STD_EXP_DIFF_CROP_4DAYS_ALL.png \
  --dates 20240101,20240102,20240103,20240104 \
  --crop-left 10 \
  --crop-upper 1 \
  --crop-right 1080 \
  --crop-lower 800
#  --no-crop
exit

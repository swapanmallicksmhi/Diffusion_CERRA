#!/bin/bash

set -evx

ml PDC
ml miniconda3/25.3.1-1-cpeGNU-24.11
source activate torch_rocm

BASE_DIR="/cfs/klemming/scratch/y/yarongc/SDM_seasonalStrongCond/cfg_1/June"
CHECKPOINT="checkpoint_ema_070000"

START_DATE="20240601"
END_DATE="20240605"
TYPE=mean
#TYPE=std

WORK_DIR="${PWD}"
IMAGE_DIR="${WORK_DIR}/IMAGE_FILES"
OUTPUT_DIR="${WORK_DIR}/OUTPUT"
mkdir -p "${IMAGE_DIR}"
mkdir -p "${OUTPUT_DIR}"

CURRENT_DATE="${START_DATE}"

while [[ "${CURRENT_DATE}" -le "${END_DATE}" ]]; do

    DATE_FMT=$(date -d "${CURRENT_DATE}" +"%Y%m%d")
    SRC_DIR="${BASE_DIR}/${DATE_FMT}"
    ERA_FILE="${SRC_DIR}/${CHECKPOINT}/conditioning_t2m_era5_${TYPE}_${DATE_FMT}_120000_UTC.png"
    CERRA_FILE="${SRC_DIR}/${CHECKPOINT}/reference_t2m_cerra_${TYPE}_${DATE_FMT}_120000_UTC.png"
    DIFFUSION_FILE=$(ls "${SRC_DIR}"/best_overall/best_t2m_cerra_${TYPE}_ema_*.png 2>/dev/null | head -n 1)

    if [[ -f "${ERA_FILE}" ]]; then
        cp "${ERA_FILE}" "${IMAGE_DIR}/ERA_${TYPE}_${DATE_FMT}_120000_UTC.png"
    fi

    if [[ -f "${CERRA_FILE}" ]]; then
        cp "${CERRA_FILE}" "${IMAGE_DIR}/CERRA_${TYPE}_${DATE_FMT}_120000_UTC.png"
    fi

    if [[ -f "${DIFFUSION_FILE}" ]]; then
        cp "${DIFFUSION_FILE}" "${IMAGE_DIR}/DIFFUSION_${TYPE}_${DATE_FMT}_120000_UTC.png"
    fi

    CURRENT_DATE=$(date -d "${CURRENT_DATE} +1 day" +"%Y%m%d")

done

python crop_${TYPE}.py \
    --input-dir "${IMAGE_DIR}" \
    --output-file "${OUTPUT_DIR}/combined_June_${TYPE}_${CURRENT_DATE}.png" \
    --crop-left 20 \
    --crop-right 100 \
    --crop-top 30 \
    --crop-bottom 10 \
    --date-font-size 150 \
    --date-width 120 \
    --date-scale 5 \
    --colorbar-panel CERRA \
    --colorbar-crop-left 850 \
    --colorbar-crop-right 25 \
    --colorbar-crop-top 10 \
    --colorbar-crop-bottom 10 \
    --colorbar-scale 2.5

rm -rf "${IMAGE_DIR}"
rm -rf "${OUTPUT_DIR}/DEBUG_colorbar_crop_rotated_scaled.png"
exit

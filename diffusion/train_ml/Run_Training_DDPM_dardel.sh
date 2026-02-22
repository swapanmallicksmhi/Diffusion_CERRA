#!/bin/sh -l

ml PDC
ml miniconda3/25.3.1-1-cpeGNU-24.11
source activate mamba-env


VAR="t2m"
TYPE="FC_DDPM"
INPUT="/cfs/klemming/home/y/yarongc/Diffusion_CERRA/diffusion/train_ml/input_test"
OUTPUT="./TRAIN_OUTPUT_test"

# Lowered batch + microbatch + enable fp16
MODEL_FLAGS="--image_size 256 --num_channels 64"   # reduce model size if needed
TOTALSTEP="--steps 20000"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --microbatch 4 --use_fp16 True"

mkdir -p $OUTPUT

# Launch
/cfs/klemming/home/s/swapanm/.conda/envs/mamba-env/bin/python Train_Main.py \
    --data_dir $INPUT --TRAIN_OUT $OUTPUT \
      $MODEL_FLAGS $TOTALSTEP $TRAIN_FLAGS

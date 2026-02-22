#!/usr/bin/env python3
#
import os
import sys
import logging
import traceback
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

GRID_SIZE = (1200, 784)  # Dimensions: 1200 x 784 pixels

# Global logger (set in main)
LOG = None

# -------- utils --------
def setup_rank_logging(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"rank_{rank}.log")

    logger_name = f"evaluate.rank{rank}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path)

    fmt_str = f"%(asctime)s [rank {rank}] %(levelname)s: %(message)s"
    fmt = logging.Formatter(fmt_str)

    stream_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def save_error_trace(output_dir, rank, tb):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"error_rank{rank}.log")
    with open(path, "w") as f:
        f.write(tb)
    return path

#---------------------------------------
def save_image(img_array, filename, data=None, upscale_factor=4, dpi=100):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    img_array_clipped = np.clip(img_array, 0, 255)
    img = Image.fromarray(np.uint8(img_array_clipped))

    # FORCE GRID SIZE
    img_resized = img.resize(GRID_SIZE, resample=Image.BICUBIC)

    fig, ax = plt.subplots(figsize=(GRID_SIZE[0]/100, GRID_SIZE[1]/100), dpi=100)
    ax.imshow(np.array(img_resized))
    ax.axis('off')

    plt.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0,
        facecolor='white',
        dpi=dpi,
        transparent=False
    )
    plt.close(fig)

# SWAPAN Alternative simplified version if you don't need the coordinate system:
def save_image_simple1(img_array, filename, upscale_factor=4):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    img_array_clipped = np.clip(img_array, 0, 255)
    img = Image.fromarray(np.uint8(img_array_clipped))

    # FORCE GRID SIZE
    img_resized = img.resize(GRID_SIZE, resample=Image.BICUBIC)
    img_resized.save(filename, quality=95)

def init_distributed_from_env():
    rank = None
    world_size = None
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        return None, None

    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    return rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

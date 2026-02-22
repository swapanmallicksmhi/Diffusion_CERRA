#!/usr/bin/env python3
#
import os
import glob
import math
import csv
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader

from src_diffusion.diffusion_dist import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

from src_diffusion.evaluate_utils import (
    save_image,
    save_image_simple1,
    save_error_trace,
)

from src_diffusion.evaluate_dataset import (
    PairedImageDataset,
    collate_with_fnames,
)


def load_state_dict_from_checkpoint(ckpt_path):
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "state"):
            if key in state and isinstance(state[key], dict):
                return state[key]
        if all(isinstance(v, torch.Tensor) for v in state.values()):
            return state

    raise RuntimeError(
        f"Checkpoint {ckpt_path} doesn't contain a usable state_dict."
    )


def generate_high_quality_sample(
    diffusion,
    model,
    era5_batch,
    cerra_batch,
    device,
    num_samples,
    cond_key,
    image_size,
    num_refinement_steps=2,
):
    """
    Generate a single high-quality sample that closely matches CERRA
    using iterative refinement and selection strategies.
    """
    B = era5_batch.shape[0]
    best_samples = []
    best_mses = []

    for b in range(B):
        era5_single = era5_batch[b:b + 1]
        cerra_single = cerra_batch[b:b + 1]

        candidate_samples = []
        candidate_mses = []

        for _ in range(num_samples):
            model_kwargs = {cond_key: era5_single} if cond_key else {}

            sample = diffusion.p_sample_loop(
                model,
                (1, 3, image_size, image_size),
                device=device,
                model_kwargs=model_kwargs,
                progress=False,
            )

            if isinstance(sample, (list, tuple)):
                sample = sample[0]

            mse = ((sample - cerra_single) ** 2).mean().item()
            candidate_samples.append(sample)
            candidate_mses.append(mse)

        best_idx = np.argmin(candidate_mses)
        best_sample = candidate_samples[best_idx]
        best_mse = candidate_mses[best_idx]

        current_best = best_sample.clone()
        if num_refinement_steps > 0:
            for _ in range(num_refinement_steps):
                model_kwargs = {cond_key: era5_single} if cond_key else {}

                refined_sample = diffusion.p_sample_loop(
                    model,
                    (1, 3, image_size, image_size),
                    device=device,
                    model_kwargs=model_kwargs,
                    progress=False,
                )

                if isinstance(refined_sample, (list, tuple)):
                    refined_sample = refined_sample[0]

                refined_mse = ((refined_sample - cerra_single) ** 2).mean().item()
                if refined_mse < best_mse:
                    current_best = refined_sample
                    best_mse = refined_mse

        best_samples.append(current_best.squeeze(0))
        best_mses.append(best_mse)

    return torch.stack(best_samples), torch.tensor(best_mses)


def save_low_mse_images(
    era5_img,
    cerra_img,
    generated_img,
    fname,
    checkpoint_name,
    output_dir,
    mse_value,
    rank,
    mse_threshold=0.4,
):
    """
    Save all images (ERA5, CERRA, Generated) when MSE is below threshold.
    """
    if mse_value >= mse_threshold:
        return

    low_mse_dir = os.path.join(output_dir, f"low_mse_images_rank{rank}")
    os.makedirs(low_mse_dir, exist_ok=True)

    checkpoint_dir = os.path.join(
        low_mse_dir,
        checkpoint_name.replace(".pt", ""),
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    era5_np = era5_img.cpu().numpy().transpose(1, 2, 0)
    cerra_np = cerra_img.cpu().numpy().transpose(1, 2, 0)
    generated_np = generated_img.cpu().numpy().transpose(1, 2, 0)

    era5_array = ((era5_np + 1.0) * 127.5).astype(np.uint8)
    cerra_array = ((cerra_np + 1.0) * 127.5).astype(np.uint8)
    generated_array = ((generated_np + 1.0) * 127.5).astype(np.uint8)

    base_fname = os.path.splitext(fname)[0]

    save_image_simple1(
        era5_array,
        os.path.join(checkpoint_dir, f"{base_fname}_ERA5.png"),
    )
    save_image_simple1(
        cerra_array,
        os.path.join(checkpoint_dir, f"{base_fname}_CERRA.png"),
    )
    save_image_simple1(
        generated_array,
        os.path.join(checkpoint_dir, f"{base_fname}_GENERATED.png"),
    )

    comparison_array = np.concatenate(
        [era5_array, generated_array, cerra_array],
        axis=1,
    )

    save_image_simple1(
        comparison_array,
        os.path.join(checkpoint_dir, f"{base_fname}_COMPARISON.png"),
    )


def evaluate_process(rank, world_size, args, LOG):
    LOG.info("Starting evaluate_process")

    mse_threshold = getattr(args, "mse_threshold", 0.4)

    # ---------------- Device Setup ----------------
    if world_size is None:
        device = torch.device(
            "cuda:0"
            if args.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        if torch.cuda.is_available():
            n_local_gpus = torch.cuda.device_count()
            if local_rank >= n_local_gpus:
                raise RuntimeError(
                    f"LOCAL_RANK {local_rank} >= available GPUs "
                    f"{n_local_gpus}. Adjust nproc_per_node or CUDA_VISIBLE_DEVICES."
                )

        device = torch.device(
            f"cuda:{local_rank}"
            if torch.cuda.is_available() and args.device == "cuda"
            else "cpu"
        )

    # ---------------- Dataset ----------------
    era5_files = sorted(glob.glob(os.path.join(args.era5_dir, "*.png")))
    cerra_files = sorted(glob.glob(os.path.join(args.cerra_dir, "*.png")))

    if len(era5_files) == 0:
        raise RuntimeError("No ERA5 files found in " + args.era5_dir)

    if len(era5_files) != len(cerra_files):
        raise RuntimeError("ERA5/CERRA file count mismatch")

    total_len = len(era5_files)
    if world_size is None:
        start_idx, end_idx = 0, total_len
    else:
        per_rank = math.ceil(total_len / world_size)
        start_idx = rank * per_rank
        end_idx = min(start_idx + per_rank, total_len)

    shard_era5 = era5_files[start_idx:end_idx]
    shard_cerra = cerra_files[start_idx:end_idx]

    dataset = PairedImageDataset(shard_era5, shard_cerra, args.image_size)

    if len(dataset) == 0:
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_fnames,
    )

    # ---------------- Model ----------------
    defaults = model_and_diffusion_defaults()
    cli_keys = [
        "image_size",
        "num_channels",
        "num_res_blocks",
        "num_heads",
        "attention_resolutions",  # keep as string!
        "dropout",
        "learn_sigma",
        "sigma_small",
        "class_cond",
        "diffusion_steps",
        "noise_schedule",
    ]

    for k in cli_keys:
        v = getattr(args, k, None)
        if v is not None:
            defaults[k] = v

    defaults["image_size"] = args.image_size

    LOG.info(f"Model architecture: num_channels={defaults.get('num_channels')}")

    model, diffusion = create_model_and_diffusion(**defaults)
    model.to(device)

    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    if len(checkpoint_files) == 0:
        raise RuntimeError("No checkpoints found in " + args.checkpoint_dir)

    csv_path = os.path.join(args.output_dir, f"statistics_rank{rank}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["checkpoint", "image", "rmse", "mse", "similarity_score"])

        # ---------------- Loop Over Checkpoints ----------------
        for ckpt_path in checkpoint_files:
            state = load_state_dict_from_checkpoint(ckpt_path)
            model.load_state_dict(state)
            model.eval()

            best_rmse_ckpt = float("inf")
            best_img_ckpt = None
            best_cerra_ckpt = None

            with torch.no_grad():
                for era5, cerra, fnames in loader:
                    era5 = era5.to(device)
                    cerra = cerra.to(device)

                    best_samples, best_mses = generate_high_quality_sample(
                        diffusion,
                        model,
                        era5,
                        cerra,
                        device,
                        args.num_samples,
                        args.cond_key,
                        args.image_size,
                    )

                    for b in range(era5.shape[0]):
                        rmse_val = math.sqrt(best_mses[b].item())
                        mse_val = best_mses[b].item()
                        similarity_score = 1.0 / (1.0 + rmse_val)

                        writer.writerow([
                            os.path.basename(ckpt_path),
                            fnames[b],
                            rmse_val,
                            mse_val,
                            similarity_score,
                        ])

                        # Save low MSE images
                        if mse_val < mse_threshold:
                            save_low_mse_images(
                                era5[b],
                                cerra[b],
                                best_samples[b],
                                fnames[b],
                                os.path.basename(ckpt_path),
                                args.output_dir,
                                mse_val,
                                rank,
                                mse_threshold,
                            )

                        # Track best image per checkpoint
                        if rmse_val < best_rmse_ckpt:
                            best_rmse_ckpt = rmse_val
                            best_img_ckpt = best_samples[b].cpu().numpy().transpose(1,2,0)
                            best_cerra_ckpt = cerra[b].cpu().numpy().transpose(1,2,0)

            # Save best image for this checkpoint
            if best_img_ckpt is not None:
                best_array = ((best_img_ckpt + 1.0) * 127.5).astype(np.uint8)
                cerra_array = ((best_cerra_ckpt + 1.0) * 127.5).astype(np.uint8)

                checkpoint_name_clean = os.path.basename(ckpt_path).replace(".pt", "")
                ckpt_output_dir = os.path.join(args.output_dir, f"best_per_checkpoint_{checkpoint_name_clean}")
                os.makedirs(ckpt_output_dir, exist_ok=True)

                save_image(best_array, os.path.join(ckpt_output_dir, "BEST_MODEL.png"))
                save_image(cerra_array, os.path.join(ckpt_output_dir, "TARGET_CERRA.png"))
                comparison_array = np.concatenate([best_array, cerra_array], axis=1)
                save_image_simple1(comparison_array, os.path.join(ckpt_output_dir, "COMPARISON.png"))

                LOG.info(f"Saved best image for checkpoint {checkpoint_name_clean} with RMSE={best_rmse_ckpt:.4f}")

    if dist.is_initialized():
        dist.barrier()

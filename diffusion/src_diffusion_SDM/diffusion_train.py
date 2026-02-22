"""
Author: Swapan Mallick
Date : 10 March 2025
TrainLoop that uses diffusion.training_losses(...) and passes
model_kwargs={'cond': era5_batch} so UNet receives cross-attention conditioning.
"""

import os
import torch
from torch.optim import AdamW
import numpy as np
from . import diffusion_dist, logger

class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,                 # DataLoader yielding (era5_batch, cerra_batch)
        lr=1e-4,
        steps=50000,
        device=None,
        save_interval=1000,
        outdir="outputs",
        weight_decay=0.0,
        schedule_sampler=None,
        microbatch=-1,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.lr = lr
        self.steps = int(steps)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.save_interval = int(save_interval)
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.schedule_sampler = schedule_sampler
        self.microbatch = microbatch if microbatch > 0 else None

        # get num_timesteps if available (DDPM)
        self.num_timesteps = getattr(self.diffusion, "num_timesteps", None)
        if self.num_timesteps is None and hasattr(self.diffusion, "use_timesteps"):
            try:
                self.num_timesteps = int(max(self.diffusion.use_timesteps)) + 1
            except Exception:
                self.num_timesteps = None

        # detect SDE-style diffusion by presence of sample_q & marginal_prob
        self.is_sde = callable(getattr(self.diffusion, "sample_q", None)) and callable(getattr(self.diffusion, "marginal_prob", None))
        # if SDE has 'N' attribute (number of discrete steps), use it for mapping; fallback to num_timesteps
        self.sde_N = getattr(self.diffusion, "N", None)

    def _sample_timesteps_and_weights(self, batch_size, device):
        if self.schedule_sampler is not None:
            t, weights = self.schedule_sampler.sample(batch_size, device)
            return t.long(), weights.float()
        # uniform random timesteps for DDPM (integer indices)
        if self.num_timesteps is None:
            raise ValueError("Cannot sample timesteps: diffusion.num_timesteps unknown and no schedule_sampler provided.")
        t = torch.randint(low=0, high=self.num_timesteps, size=(batch_size,), device=device, dtype=torch.long)
        weights = torch.ones(batch_size, device=device, dtype=torch.float)
        return t, weights

    def run_loop(self):
        self.model.to(self.device)
        self.model.train()

        step = 0
        loader = self.data
        infinite_loader = not hasattr(loader, "__len__")

        while step < self.steps:
            for batch in loader:
                if step >= self.steps:
                    break

                # Expect batch to be (era5_batch, cerra_batch)
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    era5_batch, cerra_batch = batch[0], batch[1]
                elif isinstance(batch, dict) and "era5" in batch and "cerra" in batch:
                    era5_batch, cerra_batch = batch["era5"], batch["cerra"]
                else:
                    raise ValueError("Data loader must yield (era5, cerra) tuples or dict with 'era5','cerra'.")

                era5_batch = era5_batch.to(self.device)
                cerra_batch = cerra_batch.to(self.device)

                B = cerra_batch.shape[0]
                micro = self.microbatch or B

                total_loss = 0.0

                for i in range(0, B, micro):
                    xb = cerra_batch[i : i + micro]
                    cond_slice = era5_batch[i : i + micro]

                    # SDE branch: use continuous t in [0,1] and sample q(x_t | x_0)
                    if self.is_sde:
                        batch_size_mb = xb.shape[0]
                        # sample continuous times t ~ Uniform(0,1), shape [B]
                        t_cont = torch.rand(batch_size_mb, device=self.device, dtype=torch.float32)

                        # sample x_t and underlying noise eps from the SDE marginal
                        xt, eps = self.diffusion.sample_q(xb, t_cont)

                        # Determine what to pass to model as timestep argument:
                        # Many UNet implementations expect integer timesteps; if so, map t_cont -> integer indices.
                        # Use sde_N (discrete N) if available, else fallback to num_timesteps-1
                        if self.sde_N is not None:
                            max_idx = max(1, int(self.sde_N) - 1)
                        elif self.num_timesteps is not None:
                            max_idx = max(1, int(self.num_timesteps) - 1)
                        else:
                            max_idx = 1

                        # integer timestep index for model embedding (safe fallback)
                        t_model_idx = (t_cont * float(max_idx))
                        # If model expects integer indices, convert to long; otherwise many models accept float and embed accordingly.
                        # We'll attempt to pass long indices if model's forward seems to expect integer timesteps.
                        # Safe approach: pass t_model_idx.long() \u2014 if model expects float it will cast; if not, this is required.
                        t_for_model = t_model_idx.long()

                        model_kwargs = {"cond": cond_slice}

                        # Forward pass: expect model signature model(x, t, **model_kwargs)
                        score_pred = self.model(xt, t_for_model, **model_kwargs)

                        # Compute target score: -eps / std where std = diffusion.marginal_prob(x0, t).std
                        _, std = self.diffusion.marginal_prob(xb, t_cont)
                        # ensure std broadcastable: [B,1,1,1]
                        target = -eps / std

                        # Compute per-sample MSE (spatial mean) and final scalar loss
                        per_sample_sq = ((score_pred - target) ** 2).view(xb.shape[0], -1).mean(dim=1)  # [B]
                        loss = per_sample_sq.mean()
                        losses = {"loss": loss, "mse": per_sample_sq}

                        # No extra weights from sampler in SDE branch (uniform)
                        weights = torch.ones_like(per_sample_sq, device=self.device, dtype=torch.float)

                        # Backprop / step
                        self.opt.zero_grad()
                        loss.backward()

                    else:
                        # DDPM branch: use existing diffusion.training_losses
                        t, weights = self._sample_timesteps_and_weights(xb.shape[0], device=self.device)
                        model_kwargs = {"cond": cond_slice}

                        losses = self.diffusion.training_losses(
                            self.model, xb, t, model_kwargs=model_kwargs
                        )

                        if "loss" not in losses:
                            raise RuntimeError("diffusion.training_losses must return dict containing 'loss' key.")

                        loss = (losses["loss"] * weights).mean()

                        self.opt.zero_grad()
                        loss.backward()

                    # ---- extra logging: gradient norm ----
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5

                    self.opt.step()
                    total_loss += loss.item()

                    # ---- per-step statistics ----
                    logger.logkv("loss", loss.item())
                    logger.logkv("grad_norm", grad_norm)
                    if "mse" in losses:
                        # losses["mse"] may be per-sample; take mean for logging
                        mse_val = losses["mse"].mean().item() if isinstance(losses["mse"], torch.Tensor) else float(losses["mse"])
                        logger.logkv("mse", mse_val)
                    if "vb" in losses:
                        logger.logkv("vb", losses["vb"].mean().item())

                    # ---- running mean statistics ----
                    logger.logkv_mean("loss_mean", loss.item())
                    logger.logkv_mean("grad_norm_mean", grad_norm)
                    if "mse" in losses:
                        logger.logkv_mean("mse_mean", mse_val)
                    if "vb" in losses:
                        logger.logkv_mean("vb_mean", losses["vb"].mean().item())

                if step % 10 == 0:
                    print(f"[step {step}] loss = {total_loss:.6f}")
                    logger.logkv("step", step)

                if step % self.save_interval == 0:
                    ckpt_path = os.path.join(self.outdir, f"model{step:06d}.pt")
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f"[step {step}] saved checkpoint: {ckpt_path}")

                step += 1

            if not infinite_loader:
                continue

        # final save
        final_ckpt = os.path.join(self.outdir, f"model{step:06d}.pt")
        torch.save(self.model.state_dict(), final_ckpt)
        print("Training complete. Final checkpoint:", final_ckpt)


#..
def parse_resume_step_from_filename(filename):
    try:
        return int(filename.split("model")[-1].split(".")[0])
    except (IndexError, ValueError):
        return 0


def get_blob_logdir():
    return os.getenv("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    logdir = get_blob_logdir()
    if not bf.exists(logdir):
        return None

    ckpts = [f for f in bf.listdir(logdir) if f.startswith("model") and f.endswith(".pt")]
    if not ckpts:
        return None

    ckpts.sort(key=lambda f: parse_resume_step_from_filename(f))
    latest = ckpts[-1]
    return bf.join(logdir, latest)


def find_ema_checkpoint(main_checkpoint, step, rate):
    if not main_checkpoint:
        return None
    path = bf.join(bf.dirname(main_checkpoint), f"ema_{rate}_{step:06d}.pt")
    return path if bf.exists(path) else None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for t_idx, loss_val in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * t_idx / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", loss_val)

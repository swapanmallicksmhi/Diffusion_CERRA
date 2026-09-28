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
from copy import deepcopy
from torch.amp import GradScaler, autocast
#from torch.cuda.amp import GradScaler, autocast
from . import diffusion_dist, logger


class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval=1,
        save_interval=2,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        steps=50000,
        device=None,
        outdir="outputs",
        weight_decay=0.0,
        schedule_sampler=None,
        lr_anneal_steps=0,
        cond_dropout_prob=0.0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.lr = lr
        self.steps = int(steps)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.save_interval = int(save_interval)
        self.log_interval = int(log_interval)
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.schedule_sampler = schedule_sampler
        self.microbatch = microbatch if microbatch > 0 else None
        self.cond_dropout_prob = float(cond_dropout_prob) #new_add

        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        #self.scaler = GradScaler(enabled=self.use_fp16)
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

        self.ema_rate = float(ema_rate)
        self.ema_model = deepcopy(self.model).to(self.device)
        self.ema_model.eval()

        self.num_timesteps = getattr(self.diffusion, "num_timesteps", None)
        if self.num_timesteps is None and hasattr(self.diffusion, "use_timesteps"):
            try:
                self.num_timesteps = int(max(self.diffusion.use_timesteps)) + 1
            except Exception:
                self.num_timesteps = None

    def _is_sde_diffusion(self):
        return hasattr(self.diffusion, "sde_type")

    def _sample_timesteps_and_weights(self, batch_size, device):
        if self._is_sde_diffusion():
            t = torch.rand(batch_size, device=device) * 0.999 + 0.001
            weights = torch.ones(batch_size, device=device)
            return t, weights

        if self.schedule_sampler is not None:
            t, weights = self.schedule_sampler.sample(batch_size, device)
            return t.long(), weights.float()

        if self.num_timesteps is None:
            raise ValueError("Cannot sample timesteps.")

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        weights = torch.ones(batch_size, device=device)
        return t, weights

    def _update_ema(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_rate).add_(p.data, alpha=1 - self.ema_rate)

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

                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    target_batch, cond_batch = batch[0], batch[1]
                elif isinstance(batch, dict) and "era5" in batch and "cerra" in batch:
                    target_batch, cond_batch = batch["cerra"], batch["era5"]
                else:
                    raise ValueError("Data loader must yield (era5, cerra) tuples or dict.")

                target_batch = target_batch.to(self.device)
                cond_batch = cond_batch.to(self.device)

                if not torch.isfinite(target_batch).all() or not torch.isfinite(cond_batch).all():
                    continue

                B = target_batch.shape[0]
                micro = self.microbatch or B

                self.opt.zero_grad(set_to_none=True) #new_add: micorbatch_logic
                total_loss = 0.0
                num_microbatches = 0

                for i in range(0, B, micro):
                    xb = target_batch[i: i + micro]
                    cond_slice = cond_batch[i: i + micro]
                    
                    if cond_slice is not None and self.cond_dropout_prob > 0: #new_add
                        drop_mask = (
                            torch.rand(cond_slice.shape[0], device=cond_slice.device)
                            < self.cond_dropout_prob
                        )
                        if drop_mask.any():
                            cond_slice = cond_slice.clone()
                            cond_slice[drop_mask] = 0.0
                    
                    t, weights = self._sample_timesteps_and_weights(
                        xb.shape[0], device=self.device
                    )

                    #with autocast(enabled=self.use_fp16):
                    with autocast("cuda", enabled=self.use_fp16):
                        losses = self.diffusion.training_losses(
                            self.model, xb, t, model_kwargs={"cond": cond_slice}
                        )
                        loss = (losses["loss"] * weights).mean()

                    if not torch.isfinite(loss):
                        continue
                    
                    self.scaler.scale(loss).backward()
                    total_loss += loss.item()
                    num_microbatches += 1
                
                #new_add
                if num_microbatches == 0:
                    print(f"Warning: no valid microbatches at step {step}, skipping optimizer step")
                    self.opt.zero_grad(set_to_none=True)
                    step += 1
                    continue
                
                avg_loss = total_loss / num_microbatches

                self.scaler.unscale_(self.opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # new_add: guard against non-finite grad norm
                if not torch.isfinite(grad_norm):
                    print(f"Warning: Non-finite grad norm at step {step}, skipping optimizer step")
                    self.opt.zero_grad(set_to_none=True)
                    step += 1
                    continue

                self.scaler.step(self.opt)
                self.scaler.update()

                self._update_ema()

                logger.logkv("loss", avg_loss)
                logger.logkv("grad_norm", grad_norm.item() if grad_norm is not None else 0.0)

                logger.logkv_mean("loss_mean", avg_loss)
                logger.logkv_mean("grad_norm_mean", grad_norm.item() if grad_norm is not None else 0.0)

                if step % 10 == 0:
                    print(f"[step {step}] loss = {avg_loss:.6f}, grad_norm = {grad_norm.item() if grad_norm is not None else 0.0:.6f}")
                    logger.logkv("step", step)

                if step % self.log_interval == 0:
                    logger.dumpkvs()

                if step % self.save_interval == 0:
                    ckpt_path = os.path.join(self.outdir, f"model{step:06d}.pt")
                    torch.save(self.model.state_dict(), ckpt_path)

                    ema_path = os.path.join(self.outdir, f"ema_{step:06d}.pt")
                    torch.save(self.ema_model.state_dict(), ema_path)

                    print(f"[step {step}] saved checkpoint")

                step += 1

            if not infinite_loader:
                continue

        final_ckpt = os.path.join(self.outdir, f"model{step:06d}.pt")
        torch.save(self.model.state_dict(), final_ckpt)

        ema_final = os.path.join(self.outdir, f"ema_{step:06d}.pt")
        torch.save(self.ema_model.state_dict(), ema_final)

        print("Training complete. Final checkpoint:", final_ckpt)


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

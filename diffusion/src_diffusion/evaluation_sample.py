#!/usr/bin/env python3
"""
Sample Generation Subroutine
"""

import torch as th
import time
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src_diffusion.diffusion_dist import create_model_and_diffusion
from src_diffusion.zarr_load import ZarrNormalizer, get_vars_indices
import zarr
import numpy as np


class SampleGenerator:
    """Handles model loading and sample generation"""
    
    def __init__(self, device=None):
        self.device = device or th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model = None
        self.diffusion = None
        self.target_normalizer = None
        self.input_normalizer = None
        print(f"SampleGenerator initialized with device: {self.device}")
    
    def load_model(self, checkpoint_path, model_config):
        """Load model from checkpoint"""
        print(f"\n{'='*60}")
        print(f"Loading model from: {checkpoint_path}")
        print(f"{'='*60}")
        
        # Create model with the same configuration as training
        self.model, self.diffusion = create_model_and_diffusion(**model_config)
        
        # Load checkpoint
        checkpoint = th.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            print("Loaded model state from checkpoint")
        elif 'ema' in checkpoint:
            self.model.load_state_dict(checkpoint['ema'])
            print("Loaded EMA model from checkpoint")
        #new_add
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model_state_dict from checkpoint")
        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded checkpoint directly")
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get step information if available
        if 'step' in checkpoint:
            print(f"Checkpoint step: {checkpoint['step']}")
        
        return self.model, self.diffusion
    
    def load_normalizers(self, zarr_path, input_vars, target_vars):
        """Load normalizers from Zarr metadata"""
        print(f"\n{'='*60}")
        print("Loading normalizers from Zarr metadata")
        print(f"{'='*60}")
        
        # Open Zarr store
        z = zarr.open(zarr_path, mode='r')
        
        # Get variable indices
        def get_vars_indices(vars_list, z):
            indices = []
            all_vars = np.array(z['variable'])
            for v in vars_list:
                for j, av in enumerate(all_vars):
                    if v == av:
                        indices.append(j)
                        break
            return indices
        
        target_indices = get_vars_indices(target_vars, z)
        input_indices = get_vars_indices(input_vars, z)
        
        # Load statistics and move to device
        min_stats = th.from_numpy(z['min'][:].astype(np.float32)).to(self.device)
        max_stats = th.from_numpy(z['max'][:].astype(np.float32)).to(self.device)
        
        # Extract statistics
        target_min = min_stats[target_indices]
        target_max = max_stats[target_indices]
        input_min = min_stats[input_indices]
        input_max = max_stats[input_indices]
        
        # Create normalizers
        self.target_normalizer = ZarrNormalizer(min_val=target_min, max_val=target_max)
        self.input_normalizer = ZarrNormalizer(min_val=input_min, max_val=input_max)
        
        print(f"Target variables: {target_vars}")
        print(f"Target min shape: {target_min.shape}")
        
        return self.target_normalizer, self.input_normalizer
    
    def find_index_by_datetime(self, zarr_path, target_datetime):
        """
        Find the index in Zarr store corresponding to a specific datetime
        
        Args:
            zarr_path: Path to Zarr store
            target_datetime: Datetime string (e.g., "2015-01-15 12:00:00")
            
        Returns:
            index: The index in the Zarr array
            actual_datetime: The actual datetime at that index
        """
        store = zarr.open(zarr_path, mode='r')
        
        if 'valid_time' not in store:
            raise ValueError("No 'valid_time' variable found in Zarr store")
        
        # Load time data
        time_data = store['valid_time'][:]
        units = store['valid_time'].attrs.get('units', 'seconds since 1970-01-01')
        
        # Convert to datetime
        if 'seconds since' in units:
            datetimes = pd.to_datetime(time_data, unit='s', origin='unix', utc=True)
        else:
            # Try other common units
            datetimes = pd.to_datetime(time_data, unit='h', origin='unix', utc=True)
        
        # Parse target datetime
        target_ts = pd.Timestamp(target_datetime, tz='UTC')
        
        # Find closest index
        time_diffs = np.abs((datetimes - target_ts).total_seconds())
        closest_idx = np.argmin(time_diffs)
        closest_datetime = datetimes[closest_idx]
        
        time_diff_seconds = time_diffs[closest_idx]
        if time_diff_seconds > 3600:  # More than 1 hour difference
            print(f" Warning: Closest datetime is {time_diff_seconds/3600:.1f} hours from target")
        
        print(f"Target datetime: {target_ts}")
        print(f"Found index {closest_idx} with datetime: {closest_datetime}")
        
        return closest_idx, closest_datetime
    
    def load_real_conditioning_by_datetime(
        self, 
        zarr_path, 
        input_vars, 
        datetime_str, 
        num_samples,
        add_seasonal_cond=False, #new_add
        seasonal_channels=0,
    ):
        """
        Load real conditioning data from Zarr by datetime
        
        Args:
            zarr_path: Path to Zarr store
            input_vars: List of input variable names
            datetime_str: Datetime string (e.g., "2015-01-15 12:00:00")
            num_samples: Number of samples to generate
            
        Returns:
            cond_norm: Normalized conditioning tensor
            cond_denorm: Denormalized conditioning tensor
            actual_datetime: The actual datetime used
        """
        # Find index for datetime
        index, actual_datetime = self.find_index_by_datetime(zarr_path, datetime_str)
        
        print(f"\nLoading real conditioning data from index {index} (datetime: {actual_datetime})")
        store = zarr.open(zarr_path, mode='r')
        
        # Load conditioning data
        cond_data = []
        for var in input_vars:
            data = store[var][index].astype(np.float32)
            cond_data.append(data)
        
        cond_denorm = th.from_numpy(np.stack(cond_data, axis=0)).unsqueeze(0).float().to(self.device)
        cond_norm = self.input_normalizer.normalize(cond_denorm.clone())
        #new_add
        if add_seasonal_cond and seasonal_channels > 0:
            day_of_year = actual_datetime.dayofyear

            seasonal_features = []
            frequencies = [1, 2, 3, 4, 6, 12]
            for freq in frequencies:
                angle = 2 * np.pi * freq * (day_of_year - 1) / 365.25
                seasonal_features.append(np.sin(angle))
                seasonal_features.append(np.cos(angle))

            seasonal_features = seasonal_features[:seasonal_channels]

            h, w = cond_norm.shape[-2], cond_norm.shape[-1]
            seasonal_tensor = th.tensor(
                seasonal_features, dtype=th.float32, device=self.device
            ).view(1, -1, 1, 1).expand(1, -1, h, w)

            cond_norm = th.cat([cond_norm, seasonal_tensor], dim=1)
        
        # Repeat to match num_samples if needed
        if cond_norm.shape[0] < num_samples:
            cond_norm = cond_norm.repeat(num_samples, 1, 1, 1)
            cond_denorm = cond_denorm.repeat(num_samples, 1, 1, 1)
        
        return cond_norm, cond_denorm, actual_datetime
    
    def create_random_conditioning(
        self, num_samples, channels, height, width,
        add_seasonal_cond=False, seasonal_channels=0, #new_add
    ):
        """Create random conditioning data"""
        # Create random data in denormalized space
        cond_denorm = th.randn(num_samples, channels, height, width).to(self.device)
        
        # Scale to realistic ranges
        if channels == 2:  # [mean, std]
            cond_denorm[:, 0, :, :] = cond_denorm[:, 0, :, :] * 10 + 285  # Temperature mean
            cond_denorm[:, 1, :, :] = th.abs(cond_denorm[:, 1, :, :]) * 5 + 2  # Temperature std
        
        # Normalize
        cond_norm = self.input_normalizer.normalize(cond_denorm.clone())
        #new_add
        if add_seasonal_cond and seasonal_channels > 0:
            seasonal_tensor = th.zeros(
                num_samples, seasonal_channels, height, width, device=self.device
            )
            cond_norm = th.cat([cond_norm, seasonal_tensor], dim=1)
        
        return cond_norm, cond_denorm
    
    @th.no_grad()
    def generate_samples(self, cond_tensor, num_samples, progress=True):
        """Generate samples using the diffusion model"""
        print(f"\n{'='*60}")
        print("Generating samples...")
        print(f"{'='*60}")
        
        self.model.eval()
        
        # Ensure cond_tensor has the right batch size
        if cond_tensor.shape[0] < num_samples:
            cond_tensor = cond_tensor.repeat(num_samples, 1, 1, 1)
        elif cond_tensor.shape[0] > num_samples:
            cond_tensor = cond_tensor[:num_samples]
        
        cond_tensor = cond_tensor.to(self.device)
        #cond_tensor = th.zeros_like(cond_tensor) #test for zero_cond
        
        print(f"Conditioning shape: {cond_tensor.shape}")
        
        # Create initial noise
        shape = (num_samples, self.model.in_channels, 
                 cond_tensor.shape[2], cond_tensor.shape[3])
        noise = th.randn(shape, device=self.device)
        
        print(f"Initial noise shape: {shape}")
        print(f"Sampling with {self.diffusion.num_timesteps} steps...")
        
        # Time the sampling
        start_time = time.time()
        
        # Run sampling
        samples = self.diffusion.p_sample_loop(
            self.model,
            shape,
            noise=noise,
            clip_denoised=True,
            model_kwargs={"cond": cond_tensor},
            device=self.device,
            progress=progress
        )
        
        sampling_time = time.time() - start_time
        
        print(f"Generated samples shape: {samples.shape}")
        print(f"Sampling time: {sampling_time:.2f} seconds")
        
        return samples, sampling_time
    
    def denormalize_samples(self, samples):
        """Convert normalized samples back to original scale"""
        if self.target_normalizer:
            # Ensure normalizer is on same device
            if self.target_normalizer.min.device != samples.device:
                self.target_normalizer.min = self.target_normalizer.min.to(samples.device)
                self.target_normalizer.max = self.target_normalizer.max.to(samples.device)
                self.target_normalizer.range = self.target_normalizer.range.to(samples.device)
            return self.target_normalizer.denormalize(samples)
        return samples
    
    def get_model_info(self):
        """Get model information dictionary"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "in_channels": self.model.in_channels,
            "out_channels": self.model.out_channels,
            "model_channels": self.model.model_channels,
            "diffusion_steps": self.diffusion.num_timesteps if self.diffusion else "N/A",
        }

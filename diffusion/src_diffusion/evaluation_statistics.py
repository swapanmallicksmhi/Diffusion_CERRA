#!/usr/bin/env python3
"""
Statistics Logger Subroutine
Handles comprehensive statistics logging to text and JSON files
"""

import os
import json
import time
import torch as th
import numpy as np
from datetime import datetime


class StatisticsLogger:
    """Comprehensive statistics logger"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create stats file with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats_file = os.path.join(output_dir, f"generation_stats_{self.timestamp}.txt")
        self.json_file = os.path.join(output_dir, f"generation_stats_{self.timestamp}.json")
        
        # Initialize statistics dictionary
        self.stats = {
            "timestamp": self.timestamp,
            "command_line_args": {},
            "system_info": {},
            "model_info": {},
            "data_info": {},
            "normalizer_info": {},
            "generation_info": {},
            "sample_statistics": {},
            "timing_info": {}
        }
        
        # Start timing
        self.start_time = time.time()
        
        # Write header
        self._write_header()
    
    def _write_header(self):
        """Write header to stats file"""
        with open(self.stats_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DIFFUSION MODEL SAMPLE GENERATION STATISTICS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def _append_to_file(self, text):
        """Append text to stats file"""
        with open(self.stats_file, 'a') as f:
            f.write(text)
    
    def _prepare_for_json(self, obj):
        """Recursively convert tensors to JSON-serializable types"""
        if isinstance(obj, th.Tensor):
            return obj.cpu().detach().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def log_command_line_args(self, args):
        """Log command line arguments"""
        self.stats["command_line_args"] = vars(args)
        self._append_to_file("\n--- COMMAND LINE ARGUMENTS ---\n")
        for key, value in vars(args).items():
            self._append_to_file(f"{key}: {value}\n")
    
    def log_system_info(self, device):
        """Log system information"""
        self.stats["system_info"] = {
            "device": str(device),
            "cuda_available": th.cuda.is_available(),
            "cuda_device_count": th.cuda.device_count() if th.cuda.is_available() else 0,
            "cuda_device_name": th.cuda.get_device_name(0) if th.cuda.is_available() else "N/A",
            "pytorch_version": th.__version__,
        }
        
        self._append_to_file("\n--- SYSTEM INFORMATION ---\n")
        for key, value in self.stats["system_info"].items():
            self._append_to_file(f"{key}: {value}\n")
    
    def log_model_info(self, model_info):
        """Log model information"""
        self.stats["model_info"] = model_info
        
        self._append_to_file("\n--- MODEL INFORMATION ---\n")
        self._append_to_file(f"Total Parameters: {model_info.get('total_parameters', 'N/A'):,}\n")
        self._append_to_file(f"Trainable Parameters: {model_info.get('trainable_parameters', 'N/A'):,}\n")
        self._append_to_file(f"In Channels: {model_info.get('in_channels', 'N/A')}\n")
        self._append_to_file(f"Out Channels: {model_info.get('out_channels', 'N/A')}\n")
        self._append_to_file(f"Diffusion Steps: {model_info.get('diffusion_steps', 'N/A')}\n")
    
    def log_data_info(self, zarr_path, input_vars, target_vars, train_indices=None):
        """Log dataset information"""
        data_info = {
            "zarr_path": zarr_path,
            "input_variables": input_vars,
            "target_variables": target_vars,
        }
        
        if train_indices is not None:
            data_info["training_samples"] = len(train_indices)
        
        self.stats["data_info"] = data_info
        
        self._append_to_file("\n--- DATA INFORMATION ---\n")
        self._append_to_file(f"Zarr Path: {zarr_path}\n")
        self._append_to_file(f"Input Variables: {input_vars}\n")
        self._append_to_file(f"Target Variables: {target_vars}\n")
        if train_indices is not None:
            self._append_to_file(f"Training Samples: {len(train_indices)}\n")
    
    def log_normalizer_info(self, target_normalizer, input_normalizer):
        """Log normalizer statistics"""
        norm_info = {
            "target_mean_shape": list(target_normalizer.mean.shape) if target_normalizer else None,
            "target_std_shape": list(target_normalizer.std.shape) if target_normalizer else None,
            "input_mean_shape": list(input_normalizer.mean.shape) if input_normalizer else None,
            "input_std_shape": list(input_normalizer.std.shape) if input_normalizer else None,
        }
        
        self.stats["normalizer_info"] = norm_info
        
        self._append_to_file("\n--- NORMALIZER INFORMATION ---\n")
        if target_normalizer:
            self._append_to_file(f"Target Mean Shape: {target_normalizer.mean.shape}\n")
    
    def log_conditioning_info(self, cond_norm, cond_denorm, is_real=False):
        """Log conditioning data statistics"""
        cond_info = {
            "is_real_data": is_real,
            "shape": list(cond_norm.shape),
            "normalized_stats": {
                "mean": cond_norm.mean().item(),
                "std": cond_norm.std().item(),
                "min": cond_norm.min().item(),
                "max": cond_norm.max().item(),
            },
            "denormalized_stats": {
                "mean": cond_denorm.mean().item(),
                "std": cond_denorm.std().item(),
                "min": cond_denorm.min().item(),
                "max": cond_denorm.max().item(),
            }
        }
        
        self.stats["conditioning_info"] = cond_info
        
        self._append_to_file("\n--- CONDITIONING DATA STATISTICS ---\n")
        self._append_to_file(f"Shape: {cond_norm.shape}\n")
        self._append_to_file(f"Type: {'Real' if is_real else 'Random'}\n")
    
    def log_generation_info(self, num_samples, sampling_steps):
        """Log generation parameters"""
        self.stats["generation_info"] = {
            "num_samples": num_samples,
            "sampling_steps": sampling_steps,
        }
        
        self._append_to_file("\n--- GENERATION PARAMETERS ---\n")
        self._append_to_file(f"Number of Samples: {num_samples}\n")
        self._append_to_file(f"Sampling Steps: {sampling_steps}\n")
    
    def log_sample_statistics(self, samples_norm, samples_denorm, target_vars):
        """Log comprehensive sample statistics"""
        sample_stats = {
            "normalized": {
                "overall_mean": samples_norm.mean().item(),
                "overall_std": samples_norm.std().item(),
                "overall_min": samples_norm.min().item(),
                "overall_max": samples_norm.max().item(),
            },
            "denormalized": {
                "overall_mean": samples_denorm.mean().item(),
                "overall_std": samples_denorm.std().item(),
                "overall_min": samples_denorm.min().item(),
                "overall_max": samples_denorm.max().item(),
            },
            "per_channel": []
        }
        
        self._append_to_file("\n--- GENERATED SAMPLE STATISTICS ---\n")
        self._append_to_file(f"Shape: {samples_norm.shape}\n\n")
        
        for c in range(samples_denorm.shape[1]):
            var_name = target_vars[c] if c < len(target_vars) else f"Channel_{c}"
            channel_stats = {
                "channel": c,
                "variable": var_name,
                "normalized": {
                    "mean": samples_norm[:, c].mean().item(),
                    "std": samples_norm[:, c].std().item(),
                    "min": samples_norm[:, c].min().item(),
                    "max": samples_norm[:, c].max().item(),
                },
                "denormalized": {
                    "mean": samples_denorm[:, c].mean().item(),
                    "std": samples_denorm[:, c].std().item(),
                    "min": samples_denorm[:, c].min().item(),
                    "max": samples_denorm[:, c].max().item(),
                }
            }
            sample_stats["per_channel"].append(channel_stats)
            
            self._append_to_file(f"\n{var_name}:\n")
            self._append_to_file(f"  Normalized   - Mean: {channel_stats['normalized']['mean']:8.4f}, "
                                f"Range: [{channel_stats['normalized']['min']:8.4f}, {channel_stats['normalized']['max']:8.4f}]\n")
            self._append_to_file(f"  Denormalized - Mean: {channel_stats['denormalized']['mean']:8.2f}, "
                                f"Range: [{channel_stats['denormalized']['min']:8.2f}, {channel_stats['denormalized']['max']:8.2f}]\n")
        
        self.stats["sample_statistics"] = sample_stats
    
    def log_timing(self, stage, duration):
        """Log timing information"""
        if "timing_info" not in self.stats:
            self.stats["timing_info"] = {}
        self.stats["timing_info"][stage] = f"{duration:.2f} seconds"
    
    def finalize(self):
        """Finalize statistics and save to files"""
        # Total time
        total_time = time.time() - self.start_time
        self.stats["timing_info"]["total_time"] = f"{total_time:.2f} seconds"
        
        # Write summary
        self._append_to_file("\n" + "=" * 80 + "\n")
        self._append_to_file("SUMMARY\n")
        self._append_to_file("=" * 80 + "\n")
        self._append_to_file(f"Total Execution Time: {total_time:.2f} seconds\n")
        
        # Save as JSON
        with open(self.json_file, 'w') as f:
            json_stats = self._prepare_for_json(self.stats)
            json.dump(json_stats, f, indent=2)
        
        print(f"Statistics saved to:")
        print(f"  {self.stats_file}")
        print(f"  {self.json_file}")

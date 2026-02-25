#!/usr/bin/env python3
"""
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
import torch
from datetime import datetime


class PlotGenerator:
    """Generates geographic plots with longitude/latitude information"""
    
    # Fixed grid size for all plots
    GRID_SIZE = (1200, 784)
    
    def __init__(self, zarr_path, output_dir):
        """
        Initialize with Zarr path to load longitude/latitude
        
        Args:
            zarr_path: Path to Zarr store with latitude/longitude coordinates
            output_dir: Directory to save plots
        """
        self.zarr_path = zarr_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load latitude and longitude from Zarr
        self._load_coordinates()
        
        # Store datetime information
        self.conditioning_datetime = None
    
    def set_conditioning_datetime(self, datetime_obj):
        """Set the conditioning datetime for plot titles"""
        self.conditioning_datetime = datetime_obj
    
    def _load_coordinates(self):
        """Load latitude and longitude from Zarr store"""
        print(f"\n{'='*60}")
        print("Loading geographic coordinates from Zarr")
        print(f"{'='*60}")
        
        ds = xr.open_dataset(self.zarr_path, engine="zarr")
        
        # Extract latitude and longitude
        if 'latitude' in ds:
            self.lat = ds['latitude'].values
            print(f"Latitude shape: {self.lat.shape}, range: [{self.lat.min():.2f}, {self.lat.max():.2f}]")
        else:
            raise ValueError("No 'latitude' variable found in Zarr store")
        
        if 'longitude' in ds:
            self.lon = ds['longitude'].values
            print(f"Longitude shape: {self.lon.shape}, range: [{self.lon.min():.2f}, {self.lon.max():.2f}]")
        else:
            raise ValueError("No 'longitude' variable found in Zarr store")
        
        # Handle longitude wrapping (convert 0-360 to -180 to 180 if needed)
        if self.lon.max() > 180:
            print("Converting longitude from 0-360 to -180-180 range")
            self.lon = np.where(self.lon > 180, self.lon - 360, self.lon)
            print(f"Adjusted longitude range: [{self.lon.min():.2f}, {self.lon.max():.2f}]")
        
        ds.close()
    
    def _get_variable_info(self, var_name):
        """Get colormap and levels based on variable name"""
        var_lower = var_name.lower()
        
        if "std" in var_lower or "standard" in var_lower:
            levels = np.arange(0, 2.1, 0.2)
            cmap = plt.get_cmap("BuPu", len(levels)-1)
            norm = BoundaryNorm(levels, cmap.N)
            label = f"{var_name} (Standard Deviation)"
            return levels, cmap, norm, label
        elif "mean" in var_lower:
            levels = np.arange(242, 300, 2)  # Temperature in Kelvin
            cmap = plt.get_cmap("jet", len(levels)-1)
            norm = BoundaryNorm(levels, cmap.N)
            label = f"{var_name} (Mean Temperature, K)"
            return levels, cmap, norm, label
        else:
            return None, "viridis", None, var_name
    
    def _format_datetime(self, dt):
        """Format datetime for plot titles"""
        if isinstance(dt, str):
            return dt
        elif isinstance(dt, datetime):
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif hasattr(dt, 'strftime'):
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            return str(dt)
    
    def _prepare_data(self, data_tensor, var_name):
        """
        Prepare tensor data for plotting
        
        Args:
            data_tensor: PyTorch tensor of shape [C, H, W] or [H, W]
            var_name: Variable name for identifying mean/std
            
        Returns:
            numpy array ready for plotting
        """
        # Convert to numpy and ensure 2D
        if isinstance(data_tensor, torch.Tensor):
            data_np = data_tensor.cpu().detach().numpy()
        else:
            data_np = data_tensor
        
        # If multi-channel, select appropriate channel
        if data_np.ndim == 3:
            if "std" in var_name.lower() and data_np.shape[0] > 1:
                # Assume last channel is std if multiple
                data_np = data_np[-1]
            else:
                # Take first channel for mean
                data_np = data_np[0]
        
        return data_np
    
    def plot_mean_and_std_separate(self, samples_denorm, sample_idx, var_names, 
                                   conditioning_time=None,
                                   lon_min=None, lon_max=None, lat_min=None, lat_max=None,
                                   output_subdir=None):
        """
        Generate separate plots for mean and standard deviation with datetime
        
        Args:
            samples_denorm: Denormalized samples tensor [num_samples, channels, H, W]
            sample_idx: Index of sample to plot
            var_names: List of variable names
            conditioning_time: Datetime of conditioning data
            lon_min, lon_max, lat_min, lat_max: Optional domain subset
            output_subdir: Optional subdirectory to save plots in
        """
        # Set default domain if not provided
        if lon_min is None:
            lon_min = float(self.lon.min())
        if lon_max is None:
            lon_max = float(self.lon.max())
        if lat_min is None:
            lat_min = float(self.lat.min())
        if lat_max is None:
            lat_max = float(self.lat.max())
        
        # Determine output directory
        if output_subdir:
            save_dir = output_subdir
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        # Format conditioning time for title
        time_str = self._format_datetime(conditioning_time) if conditioning_time else "Unknown"
        safe_time = time_str.replace(" ", "_").replace(":", "").replace("-", "")
        
        # Get the sample
        sample = samples_denorm[sample_idx]  # [C, H, W]
        
        # Create separate plots for each channel
        for c, var_name in enumerate(var_names):
            if c >= sample.shape[0]:
                break
            
            # Prepare data
            data_2d = sample[c].cpu().detach().numpy()
            
            # Get plotting parameters
            levels, cmap, norm, label = self._get_variable_info(var_name)
            
            # Create filename with datetime
            filename = f"sample_{sample_idx:03d}_{var_name}_cond_{safe_time}.png"
            save_path = os.path.join(save_dir, filename)
            
            # Create title with datetime
            title = f"Sample {sample_idx+1} - {label}\nConditioning: {time_str}"
            
            # Create plot
            self._create_geographic_plot(
                data_2d, 
                lon_min, lon_max, lat_min, lat_max,
                cmap, norm, levels,
                title=title,
                filename=save_path
            )
            
            print(f"  Saved: {filename}")
    
    def plot_all_samples_grid(self, samples_denorm, var_names, 
                             conditioning_time=None,
                             lon_min=None, lon_max=None, lat_min=None, lat_max=None,
                             output_subdir=None):
        """
        Create a grid plot of all samples for each variable with datetime
        
        Args:
            samples_denorm: Denormalized samples tensor [num_samples, channels, H, W]
            var_names: List of variable names
            conditioning_time: Datetime of conditioning data
            lon_min, lon_max, lat_min, lat_max: Optional domain subset
            output_subdir: Optional subdirectory to save plots in
        """
        num_samples = samples_denorm.shape[0]
        num_channels = samples_denorm.shape[1]
        
        # Determine output directory
        if output_subdir:
            save_dir = output_subdir
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        # Format conditioning time
        time_str = self._format_datetime(conditioning_time) if conditioning_time else "Unknown"
        safe_time = time_str.replace(" ", "_").replace(":", "").replace("-", "")
        
        # Set default domain
        if lon_min is None:
            lon_min = float(self.lon.min())
        if lon_max is None:
            lon_max = float(self.lon.max())
        if lat_min is None:
            lat_min = float(self.lat.min())
        if lat_max is None:
            lat_max = float(self.lat.max())
        
        # Create a grid plot for each variable
        for c in range(min(num_channels, len(var_names))):
            var_name = var_names[c]
            
            # Get plotting parameters
            levels, cmap, norm, label = self._get_variable_info(var_name)
            
            # Create figure with subplots
            cols = min(4, num_samples)
            rows = (num_samples + cols - 1) // cols
            
            fig = plt.figure(figsize=(5*cols, 4*rows))
            fig.suptitle(f"{var_name} - All Generated Samples\nConditioning: {time_str}", fontsize=16)
            
            for i in range(num_samples):
                ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
                
                # Get data
                data_2d = samples_denorm[i, c].cpu().detach().numpy()
                
                # Create mesh
                mesh = ax.pcolormesh(
                    self.lon, self.lat, data_2d,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    cmap=cmap,
                    norm=norm
                )
                
                # Set extent
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                
                # Add features
                ax.coastlines(linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
                
                # Add title with datetime
                ax.set_title(f"Sample {i+1}")
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=False, linestyle='--', linewidth=0.3, alpha=0.5)
            
            # Add colorbar
            plt.tight_layout()
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(mesh, cax=cbar_ax)
            if levels is not None:
                cbar.set_ticks(levels)
            cbar.set_label(label)
            
            # Save with datetime in filename
            filename = f"all_samples_{var_name}_grid_cond_{safe_time}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved grid plot: {filename}")
    
    def _create_geographic_plot(self, data_2d, lon_min, lon_max, lat_min, lat_max,
                                cmap, norm, levels, title, filename):
        """
        Create a single geographic plot with fixed grid size
        
        Args:
            data_2d: 2D numpy array of data
            lon_min, lon_max, lat_min, lat_max: Domain boundaries
            cmap: Colormap
            norm: Normalization
            levels: Colorbar levels
            title: Plot title (includes datetime)
            filename: Output filename
        """
        # Create figure with fixed grid size
        fig = plt.figure(figsize=(self.GRID_SIZE[0]/100, self.GRID_SIZE[1]/100), dpi=100)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Plot data
        mesh = ax.pcolormesh(
            self.lon, self.lat, data_2d,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=cmap,
            norm=norm
        )
        
        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add geographic features
        ax.coastlines(linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')
        
        # Add gridlines with labels
        gl = ax.gridlines(draw_labels=True,
                          crs=ccrs.PlateCarree(),
                          linestyle='--',
                          linewidth=0.5,
                          alpha=0.6)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add title with datetime
        ax.set_title(title, fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.7, orientation='vertical')
        if levels is not None:
            cbar.set_ticks(levels)
        cbar.set_label(title.split('-')[-1].strip() if '-' in title else title)
        
        # Save with exact grid size
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def plot_conditioning(self, cond_denorm, datetime_obj, var_names,
                         lon_min=None, lon_max=None, lat_min=None, lat_max=None,
                         output_subdir=None):
        """
        Plot conditioning data with datetime
        
        Args:
            cond_denorm: Denormalized conditioning tensor
            datetime_obj: Datetime object for this conditioning data
            var_names: List of variable names
            lon_min, lon_max, lat_min, lat_max: Optional domain subset
            output_subdir: Optional subdirectory to save plots in
        """
        # Set default domain
        if lon_min is None:
            lon_min = float(self.lon.min())
        if lon_max is None:
            lon_max = float(self.lon.max())
        if lat_min is None:
            lat_min = float(self.lat.min())
        if lat_max is None:
            lat_max = float(self.lat.max())
        
        # Determine output directory
        if output_subdir:
            save_dir = output_subdir
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        # Format datetime for filename
        time_str = self._format_datetime(datetime_obj)
        safe_time = time_str.replace(" ", "_").replace(":", "").replace("-", "")
        
        # Get the conditioning sample (first one if multiple)
        cond_sample = cond_denorm[0] if cond_denorm.shape[0] > 0 else cond_denorm
        
        # Create plots for each conditioning variable
        for c, var_name in enumerate(var_names):
            if c >= cond_sample.shape[0]:
                break
            
            # Prepare data
            data_2d = cond_sample[c].cpu().detach().numpy()
            
            # Get plotting parameters
            levels, cmap, norm, label = self._get_variable_info(var_name)
            
            # Create filename with datetime
            filename = f"conditioning_{var_name}_{safe_time}.png"
            save_path = os.path.join(save_dir, filename)
            
            # Create title with datetime
            title = f"Conditioning - {label}\nTime: {time_str}"
            
            # Create plot
            self._create_geographic_plot(
                data_2d,
                lon_min, lon_max, lat_min, lat_max,
                cmap, norm, levels,
                title=title,
                filename=save_path
            )
            
            print(f"  Saved conditioning: {filename}")

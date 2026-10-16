import numpy as np
from scipy.interpolate import griddata
import xarray as xr


class XarrayRegridder:

    def __init__(
        self,
        source_grid,
        target_grid,
        method="bilinear",
        periodic=False,
        reuse_weights=False,
    ):
        self.src_lat = np.asarray(
            source_grid["lat"]
            if "lat" in source_grid
            else source_grid["latitude"]
        )
        self.src_lon = np.asarray(
            source_grid["lon"]
            if "lon" in source_grid
            else source_grid["longitude"]
        )
        self.tgt_lat = np.asarray(
            target_grid["lat"]
            if "lat" in target_grid
            else target_grid["latitude"]
        )
        self.tgt_lon = np.asarray(
            target_grid["lon"]
            if "lon" in target_grid
            else target_grid["longitude"]
        )

        self.tgt_shape = self.tgt_lat.shape
        self.target_grid = target_grid
        self.src_points = np.column_stack(
            (self.src_lat.ravel(), self.src_lon.ravel())
        )

    def _regrid_array(self, da, keep_attrs=True):
        vals = da.values

        if vals.ndim > 2:
            spatial_shape = vals.shape[-2:]
            batch_shape = vals.shape[:-2]
            flat_vals = vals.reshape(-1, np.prod(spatial_shape))

            regridded_slices = []
            for slice_2d in flat_vals:
                res = griddata(
                    self.src_points,
                    slice_2d,
                    (self.tgt_lat, self.tgt_lon),
                    method="linear",
                )
                regridded_slices.append(res)

            out_shape = batch_shape + self.tgt_shape
            regridded_vals = np.array(regridded_slices).reshape(out_shape)

            # HIGHLIGHT: Preserve leading non-spatial dimensions (e.g. ('time', 'y', 'x'))
            batch_dims = da.dims[:-2]
            spatial_dims = tuple(self.target_grid.dims)
            out_dims = batch_dims + spatial_dims

            # Preserve non-spatial coordinates (e.g., 'time') from input
            out_coords = {d: da.coords[d] for d in batch_dims if d in da.coords}
            # Add spatial target coordinates
            out_coords.update(self.target_grid.coords)

        else:
            regridded_vals = griddata(
                self.src_points,
                vals.ravel(),
                (self.tgt_lat, self.tgt_lon),
                method="linear",
            )
            out_dims = self.target_grid.dims
            out_coords = self.target_grid.coords

        out_da = xr.DataArray(
            regridded_vals,
            coords=out_coords,
            dims=out_dims,
            name=da.name if hasattr(da, "name") else None,
        )

        if keep_attrs and hasattr(da, "attrs"):
            out_da.attrs.update(da.attrs)

        return out_da

    def __call__(self, source_data, *args, **kwargs):
        keep_attrs = kwargs.get("keep_attrs", True)

        if isinstance(source_data, xr.Dataset):
            regridded_ds = xr.Dataset()
            for var in source_data.data_vars:
                regridded_ds[var] = self._regrid_array(
                    source_data[var], keep_attrs=keep_attrs
                )
            if keep_attrs and hasattr(source_data, "attrs"):
                regridded_ds.attrs.update(source_data.attrs)
            return regridded_ds
        else:
            return self._regrid_array(source_data, keep_attrs=keep_attrs)

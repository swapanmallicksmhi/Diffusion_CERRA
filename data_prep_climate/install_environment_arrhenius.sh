#!/bin/bash

set -e

ENV_NAME="sdm"

echo "==============================================="
echo " Diffusion SDM Environment Installation"
echo "==============================================="
interactive -p gpu --gpus 1 -A NAISS2026-4-912-gpu

# ------------------------------------------------
# Install core scientific stack
# ------------------------------------------------
echo "Installing scientific libraries..."
module --force purge
cd $HOME/venvs
###module load GPU/buildenv-nvhpc/25.9-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
python3 -m venv --system-site-packages ${ENV_NAME}
source $HOME/venvs/${ENV_NAME}/bin/activate
echo "Installing pip packages..."
pip install --upgrade pip
pip install xarray #(xarray-2026.4.0)
pip install netcdf4 #(netcdf4-1.7.4)
pip install scipy
pip install numpy #(numpy-2.4.6)
pip install pandas #(pandas-3.0.3)
pip install matplotlib #(matplotlib-3.10.9)
pip install cartopy
pip install proj
pip install pyproj
pip install shapely
pip install geos
pip install cfgrib
pip install eccodes
pip install dask
pip install zarr
pip install cdsapi
pip install cmocean
pip install xesmf
#pip install pyyaml (pyyaml-6.0.3)
#pip install torch (torch-2.12.0)

# ------------------------------------------------
# Verify installation
# ------------------------------------------------
echo "Verifying installation..."
python - <<EOF
import xarray, cartopy, cdsapi
print("✔ xarray version:", xarray.__version__)
print("✔ cartopy version:", cartopy.__version__)
print("✔ cdsapi available")
EOF

echo "==============================================="
echo " Installation completed successfully"
echo " Activate with:  source $HOME/venvs/${ENV_NAME}/bin/activate"
echo "==============================================="

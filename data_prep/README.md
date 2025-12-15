# CERRA  and ERA5-EDA Ensemble Analysis & Comparison
## Useful Links
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)
- [ERA5 Documentation](https://confluence.ecmwf.int/display/CKB/ERA5)

- [CERRA-DOC](https://climate.copernicus.eu/copernicus-regional-reanalysis-europe-cerra)
- [CERRA-DATA](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels?tab=download)
- [CERRA-PAPER](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4764)

- [ERA5EDA DOC](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
- [ERA5EDA DATA](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)

## Overview
This project provides a **modular, reproducible workflow** for processing and comparing  
**CERRA reanalysis** and **ERA5 Ensemble Data Assimilation (EDA)** datasets.

It computes:
- **Daily ensemble mean**
- **Daily ensemble standard deviation**
- **Geographical plots over Europe**
- **NetCDF outputs** for downstream analysis

The system is designed for **scientific robustness**, **HPC scalability**, and **easy extensibility**.

---

## Key Features
- Ensemble statistics from 10 members
- Automatic longitude conversion
- Domain-aware interpolation between datasets
- Regular lat/lon grid plotting
- High-quality Cartopy maps
- Modular Python design (main / plotting / NetCDF utilities)

---

## Project Structure

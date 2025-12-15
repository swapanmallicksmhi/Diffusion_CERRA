# \U0001f30d CERRA \u2013 ERA5 EDA Ensemble Analysis & Comparison

![CERRA\u2013ERA5 Banner](assets/cerra_era5_banner.svg)

## \U0001f4cc Overview
This project provides a **modular, reproducible workflow** for processing and comparing  
**CERRA reanalysis** and **ERA5 Ensemble Data Assimilation (EDA)** datasets.

It computes:
- \U0001f4ca **Daily ensemble mean**
- \U0001f4c8 **Daily ensemble standard deviation**
- \U0001f5fa\ufe0f **Geographical plots over Europe**
- \U0001f4e6 **NetCDF outputs** for downstream analysis

The system is designed for **scientific robustness**, **HPC scalability**, and **easy extensibility**.

---

## \U0001f9e0 Key Features
- Ensemble statistics from 10 members
- Automatic longitude conversion (0\u2013360 \u2192 \u2212180\u2013180)
- Domain-aware interpolation between datasets
- Regular lat/lon grid plotting
- High-quality Cartopy maps
- Modular Python design (main / plotting / NetCDF utilities)

---

## \U0001f4c1 Project Structure

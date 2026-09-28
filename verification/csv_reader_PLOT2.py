#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse

# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(description="Plot filtered T2M stations (ECMWF style)")
parser.add_argument("--csv_file", required=True, help="Input CSV file")
parser.add_argument("--valid_time", required=True, help="Valid time (YYYY-MM-DD HH:MM)")
parser.add_argument("--out_file", default="t2m_map.png", help="Output PNG")
args = parser.parse_args()

csv_file = args.csv_file
valid_time = args.valid_time
out_file = args.out_file

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv(csv_file)

# -----------------------------
# FILTER TIME
# -----------------------------
df_time = df[df["valid_time"] == valid_time].copy()

if df_time.empty:
    print(f"No data found for valid_time = {valid_time}")
    exit(1)

# -----------------------------
# FILTER DOMAIN
# -----------------------------
LAT_MIN, LAT_MAX = 54, 71
LON_MIN, LON_MAX = 4, 25

df_time = df_time[
    (df_time["latitude"].between(LAT_MIN, LAT_MAX)) &
    (df_time["longitude"].between(LON_MIN, LON_MAX))
]

# -----------------------------
# FILTER WMO PREFIX
# -----------------------------
valid_prefixes = ("11", "13", "14", "20", "21", "22", "23", "24", "25")
#valid_prefixes = ("10", "11", "12", "13", "14", "20", "21", "22", "23", "24", "25")

# Convert WMO ID safely to string (handle NaN)
df_time["wmo_str"] = df_time["wmo_id"].astype("Int64").astype(str)

df_time = df_time[df_time["wmo_str"].str.startswith(valid_prefixes)]

print(f"Filtered stations: {len(df_time)}")

# -----------------------------
# COLORBAR SETTINGS (ECMWF STYLE)
# -----------------------------
levels = np.arange(242, 300, 2)
cmap = plt.get_cmap("jet", len(levels) - 1)
norm = BoundaryNorm(levels, cmap.N)

# -----------------------------
# PLOT
# -----------------------------
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

# Map features
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", color="gray")
gl.top_labels = False
gl.right_labels = False

# -----------------------------
# SCATTER PLOT
# -----------------------------
sc = ax.scatter(
    df_time["longitude"],
    df_time["latitude"],
    c=df_time["t2m_k"],   #
    cmap=cmap,
    norm=norm,
    s=60,
    edgecolors="k",
    zorder=5
)

# -----------------------------
# COLORBAR
# -----------------------------
cbar = plt.colorbar(
    sc,
    ax=ax,
    orientation="vertical",
    pad=0.02,
    shrink=0.75
)

cbar.set_label("T2M (K)")
cbar.set_ticks(np.arange(242, 300, 4))

# -----------------------------
# TITLE
# -----------------------------
plt.title(f"T2M Observations ({valid_time} UTC)", fontsize=14)

# -----------------------------
# SAVE
# -----------------------------
plt.tight_layout()
plt.savefig(out_file, dpi=100)
print(f"Map saved to {out_file}")

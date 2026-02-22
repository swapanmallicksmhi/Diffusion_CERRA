import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("file.inp")

# ---- Extract last two columns ----
last_two_cols = df.columns[-2:]  # ['rmse', 'mse']

# ---- Plot ----
plt.figure(figsize=(8, 5))
for col in last_two_cols:
    plt.plot(df["checkpoint"], df[col], marker="o", label=col.upper())

plt.xlabel("Checkpoint")
plt.ylabel("Value")
plt.title("RMSE and MSE vs. Model Checkpoint")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# ---- Save or show ----
plt.savefig("rmse_mse_plot.png", dpi=300)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# F1 data
# -----------------------------
thresholds = [100, 200, 300]

baseline_f1_mean = [0.8202, 0.8507, 0.8585]
auto_f1_mean = [0.8401, 0.8636, 0.8658]

baseline_f1_std = [0.0075, 0.0041, 0.0028]
auto_f1_std = [0.0042, 0.0028, 0.0026]

labels = [
    "B-100", "AE-100",
    "B-200", "AE-200",
    "B-300", "AE-300"
]

mean_vals = [
    baseline_f1_mean[0], auto_f1_mean[0],
    baseline_f1_mean[1], auto_f1_mean[1],
    baseline_f1_mean[2], auto_f1_mean[2]
]

std_vals = [
    baseline_f1_std[0], auto_f1_std[0],
    baseline_f1_std[1], auto_f1_std[1],
    baseline_f1_std[2], auto_f1_std[2]
]

x = np.arange(len(labels))

# Colors: alternating blue / orange
colors = ["tab:blue", "tab:orange"] * 3

# -----------------------------
# Plot 1: Mean F1
# -----------------------------
plt.figure(figsize=(8, 4))
plt.bar(x, mean_vals, color=colors)
plt.xticks(x, labels)
plt.ylabel("Macro F1 (Mean)")
plt.title("Macro F1 Mean Across Thresholds")
plt.tight_layout()
plt.savefig("f1_means_across_thresholds.jpg", dpi=300)
plt.show()

# -----------------------------
# Plot 2: Std Dev F1
# -----------------------------
plt.figure(figsize=(8, 4))
plt.bar(x, std_vals, color=colors)
plt.xticks(x, labels)
plt.ylabel("Macro F1 (Std Dev)")
plt.title("Macro F1 Standard Deviation Across Thresholds")
plt.tight_layout()
plt.savefig("f1_std_across_thresholds.jpg", dpi=300)
plt.show()

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

# -----------------------------
# Differences (AE - Baseline)
# -----------------------------
diff_mean = np.array(auto_f1_mean) - np.array(baseline_f1_mean)
diff_std = np.array(auto_f1_std) - np.array(baseline_f1_std)

x = np.arange(len(thresholds))
labels = [str(t) for t in thresholds]

# -----------------------------
# Side-by-side plots
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: Mean difference
ax1.bar(x, diff_mean)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title("Δ Mean Macro F1 (AE - Baseline)")
ax1.set_ylabel("Difference in Mean F1")
ax1.axhline(0)  # zero reference line

# Right: Std difference
ax2.bar(x, diff_std)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_title("Δ Std Dev Macro F1 (AE - Baseline)")
ax2.set_ylabel("Difference in Std Dev")
ax2.axhline(0)

# Shared labels
fig.suptitle("Autoencoder Improvement over Baseline Across Thresholds")
plt.tight_layout()
plt.savefig("f1_difference_side_by_side.jpg", dpi=300, bbox_inches="tight")
plt.show()

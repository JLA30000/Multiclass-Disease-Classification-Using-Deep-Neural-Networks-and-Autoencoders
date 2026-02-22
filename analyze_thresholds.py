import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
CSV_PATH = "diseases.csv"   # raw dataset
LABEL_COL = None            # set to None to auto-detect first column
THRESHOLDS = [100, 200, 300]

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)

# Auto-detect label column if not specified
if LABEL_COL is None:
    LABEL_COL = df.columns[0]

# Clean labels
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

# ---------------- PER-DISEASE COUNTS ----------------
disease_counts = df[LABEL_COL].value_counts()

# ---------------- STATS ----------------
mean_samples = disease_counts.mean()
std_samples = disease_counts.std()

print("=== Disease Sample Statistics ===")
print(f"Number of unique diseases: {len(disease_counts)}")
print(f"Mean samples per disease: {mean_samples:.2f}")
print(f"Std dev samples per disease: {std_samples:.2f}")
print()

# ---------------- THRESHOLD ANALYSIS ----------------
table_rows = []

for t in THRESHOLDS:
    num_diseases = (disease_counts >= t).sum()
    total_samples = num_diseases * t
    table_rows.append([t, num_diseases, total_samples])

table_df = pd.DataFrame(
    table_rows,
    columns=[
        "Minimum samples per disease",
        "Number of diseases",
        "Total samples retained"
    ]
)

print("=== Threshold Summary ===")
print(table_df)

# ---------------- MATPLOTLIB TABLE ----------------
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

plt.title("Dataset Size vs Minimum Disease Sample Threshold",
          fontsize=13, pad=12)

plt.tight_layout()
plt.savefig("disease_threshold_table.jpeg", dpi=300,
            bbox_inches="tight")  # <-- ADD THIS
plt.show()

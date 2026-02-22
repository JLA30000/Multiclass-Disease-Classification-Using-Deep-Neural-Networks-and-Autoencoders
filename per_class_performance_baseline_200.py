import matplotlib.pyplot as plt

# Per-class performance data (Baseline classifier, threshold = 200)
data = [
    ("skin pigmentation disorder", 200, 0.4025, 0.3200, 0.3565),
    ("skin disorder", 200, 0.3673, 0.5400, 0.4372),
    ("psychotic disorder", 200, 0.5362, 0.3700, 0.4379),
    ("seborrheic keratosis", 200, 0.5032, 0.3950, 0.4426),
    ("intracranial hemorrhage", 200, 0.8964, 0.8650, 0.8804),
    ("fluid overload", 200, 0.9382, 0.8350, 0.8836),
    ("vaginitis", 200, 0.9058, 0.8650, 0.8849),
    ("chronic pain disorder", 200, 0.8660, 0.9050, 0.8851),
    ("eye alignment disorder", 200, 0.9901, 1.0000, 0.9950),
    ("nerve impingement near the shoulder", 200, 0.9950, 0.9950, 0.9950),
    ("lyme disease", 200, 0.9900, 0.9900, 0.9900),
    ("paronychia", 200, 0.9803, 0.9950, 0.9876),
]

# Table headers
columns = ["Disease", "Support", "Precision", "Recall", "F1"]

# Create figure
fig, ax = plt.subplots(figsize=(16, 7))
ax.axis("off")

# Create table
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc="center",
    cellLoc="center"
)

# Formatting
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Bold header row
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")

# Title
plt.title(
    "Baseline classifier per-class performance for a subset of representative "
    "diseases at a threshold of 200 samples",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("per_class_performance_baseline_200.jpg", dpi=300)
plt.show()

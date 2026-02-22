import matplotlib.pyplot as plt

# Per-class performance data (Baseline classifier, threshold = 300)
data = [
    ("psychotic disorder", 300, 0.4188, 0.3267, 0.3670),
    ("skin pigmentation disorder", 300, 0.3547, 0.3867, 0.3700),
    ("skin polyp", 300, 0.4618, 0.4033, 0.4306),
    ("skin disorder", 300, 0.3945, 0.6233, 0.4832),
    ("idiopathic painful menstruation", 300, 0.9014, 0.8833, 0.8923),
    ("prostatitis", 300, 0.9129, 0.8733, 0.8927),
    ("gastroesophageal reflux disease (GERD)", 300, 0.9072, 0.8800, 0.8934),
    ("fluid overload", 300, 0.9044, 0.8833, 0.8938),
    ("breast infection (mastitis)", 300, 0.9967, 1.0000, 0.9983),
    ("eating disorder", 300, 0.9934, 1.0000, 0.9967),
    ("hepatitis due to a toxin", 300, 0.9901, 1.0000, 0.9950),
    ("injury to the shoulder", 300, 0.9836, 1.0000, 0.9917),
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
    "diseases at a threshold of 300 samples",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("per_class_performance_baseline_300.jpg", dpi=300)
plt.show()

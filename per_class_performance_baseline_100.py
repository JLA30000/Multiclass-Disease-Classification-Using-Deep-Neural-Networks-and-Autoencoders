import matplotlib.pyplot as plt

# Per-class performance data (Baseline classifier, threshold = 100)
data = [
    ("post-traumatic stress disorder (PTSD)", 100, 0.1325, 0.1100, 0.1202),
    ("seborrheic keratosis", 100, 0.2188, 0.2800, 0.2456),
    ("skin pigmentation disorder", 100, 0.2574, 0.2600, 0.2587),
    ("skin polyp", 100, 0.2975, 0.3600, 0.3258),
    ("gout", 100, 0.7982, 0.9100, 0.8505),
    ("gastroesophageal reflux disease (GERD)", 100, 0.8431, 0.8600, 0.8515),
    ("envenomation from spider or animal bite", 100, 0.9744, 0.7600, 0.8539),
    ("orbital cellulitis", 100, 0.9294, 0.7900, 0.8541),
    ("cerebral palsy", 100, 0.9901, 1.0000, 0.9950),
    ("foreign body in the throat", 100, 0.9901, 1.0000, 0.9950),
    ("cold sore", 100, 0.9804, 1.0000, 0.9901),
    ("abdominal aortic aneurysm", 100, 0.9804, 1.0000, 0.9901),
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
    "diseases at a threshold of 100 samples",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("per_class_performance_baseline_100.jpg", dpi=300)
plt.show()

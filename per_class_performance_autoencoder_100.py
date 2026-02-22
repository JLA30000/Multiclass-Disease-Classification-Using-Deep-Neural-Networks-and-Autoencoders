import matplotlib.pyplot as plt

# Per-class performance data (Autoencoder, threshold = 100)
data = [
    ("skin pigmentation disorder", 100, 0.3091, 0.1700, 0.2194),
    ("seborrheic keratosis", 100, 0.3140, 0.2700, 0.2903),
    ("post-traumatic stress disorder (PTSD)", 100, 0.2689, 0.3200, 0.2922),
    ("skin disorder", 100, 0.4020, 0.4100, 0.4059),
    ("chlamydia", 100, 0.7734, 0.9900, 0.8684),
    ("spondylitis", 100, 0.9524, 0.8000, 0.8696),
    ("trigeminal neuralgia", 100, 0.7692, 1.0000, 0.8696),
    ("hyperkalemia", 100, 0.8700, 0.8700, 0.8700),
    ("cerebral palsy", 100, 1.0000, 1.0000, 1.0000),
    ("stress incontinence", 100, 1.0000, 1.0000, 1.0000),
    ("bladder cancer", 100, 0.9901, 1.0000, 0.9950),
    ("bladder obstruction", 100, 0.9901, 1.0000, 0.9950),
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
    "Autoencoder classifier per-class performance for a subset of representative "
    "diseases at a threshold of 100 samples",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("per_class_performance_autoencoder_100.jpg", dpi=300)
plt.show()

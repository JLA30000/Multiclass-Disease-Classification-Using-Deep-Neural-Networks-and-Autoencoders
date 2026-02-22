import matplotlib.pyplot as plt

# Per-class performance data (Autoencoder, threshold = 200)
data = [
    ("skin pigmentation disorder", 200, 0.4188, 0.3350, 0.3722),
    ("seborrheic keratosis", 200, 0.4471, 0.4650, 0.4559),
    ("dry eye of unknown cause", 200, 0.7113, 0.3450, 0.4646),
    ("skin disorder", 200, 0.4650, 0.4650, 0.4650),
    ("ankylosing spondylitis", 200, 0.8873, 0.9050, 0.8960),
    ("peripheral nerve disorder", 200, 0.9077, 0.8850, 0.8962),
    ("lymphedema", 200, 0.8692, 0.9300, 0.8986),
    ("common cold", 200, 0.9351, 0.8650, 0.8987),
    ("bone disorder", 200, 0.9950, 1.0000, 0.9975),
    ("eye alignment disorder", 200, 0.9950, 0.9950, 0.9950),
    ("cirrhosis", 200, 0.9950, 0.9900, 0.9925),
    ("iron deficiency anemia", 200, 1.0000, 0.9850, 0.9924),
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
    "diseases at a threshold of 200 samples",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("per_class_performance_autoencoder_200.jpg", dpi=300)
plt.show()

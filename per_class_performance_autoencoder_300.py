import matplotlib.pyplot as plt

# Per-class performance data (Autoencoder, threshold = 300)
data = [
    ("skin pigmentation disorder", 300, 0.4391, 0.3367, 0.3811),
    ("psychotic disorder", 300, 0.5806, 0.3000, 0.3956),
    ("skin polyp", 300, 0.3819, 0.4367, 0.4075),
    ("skin disorder", 300, 0.4577, 0.4867, 0.4717),
    ("urethritis", 300, 0.9382, 0.8600, 0.8974),
    ("ulcerative colitis", 300, 0.9196, 0.8767, 0.8976),
    ("peritonitis", 300, 0.8889, 0.9067, 0.8977),
    ("iridocyclitis", 300, 0.8554, 0.9467, 0.8987),
    ("obstructive sleep apnea (OSA)", 300, 0.9967, 0.9933, 0.9950),
    ("breast infection (mastitis)", 300, 0.9901, 0.9967, 0.9934),
    ("pilonidal cyst", 300, 0.9900, 0.9933, 0.9917),
    ("fracture of the facial bones", 300, 1.0000, 0.9833, 0.9916),
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
    "diseases at a threshold of 300 samples",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("per_class_performance_autoencoder_300.jpg", dpi=300)
plt.show()

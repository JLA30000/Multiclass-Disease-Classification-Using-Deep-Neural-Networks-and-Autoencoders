import matplotlib.pyplot as plt

# Per-class performance data (Class-Weighted Cross-Entropy Loss Classifier)
# Columns: Disease, Support, Accuracy, Precision, Recall, F1
data = [
    ("raynaud disease", 10, 0.0000, 0.0000, 0.0000, 0.0000),
    ("hemophilia", 20, 0.0000, 0.0000, 0.0000, 0.0000),
    ("rhabdomyolysis", 10, 0.0000, 0.0000, 0.0000, 0.0000),
    ("birth trauma", 10, 0.1000, 1.0000, 0.1000, 0.1818),
    ("burn", 50, 1.0000, 0.7937, 1.0000, 0.8850),
    ("white blood cell disease", 510, 0.9118, 0.8627, 0.9118, 0.8866),
    ("anal fissure", 220, 0.9773, 0.8113, 0.9773, 0.8866),
    ("drug abuse (methamphetamine)", 330, 0.9121, 0.8625, 0.9121, 0.8866),
    ("abscess of the lung", 20, 1.0000, 1.0000, 1.0000, 1.0000),
    ("volvulus", 10, 1.0000, 1.0000, 1.0000, 1.0000),
    ("anemia due to malignancy", 10, 1.0000, 1.0000, 1.0000, 1.0000),
    ("peyronie disease", 20, 1.0000, 1.0000, 1.0000, 1.0000),
]

# Table headers
columns = ["Disease", "Support", "Accuracy", "Precision", "Recall", "F1"]

# Create figure
fig, ax = plt.subplots(figsize=(18, 7))
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
    "Class-weighted cross-entropy loss classifier per-class performance for a subset "
    "of representative diseases (aggregated over seeds)",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("per_class_performance_cwce.jpg", dpi=300)
plt.show()

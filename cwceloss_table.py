import matplotlib.pyplot as plt

# Data
metrics = [
    "Accuracy",
    "Macro P",
    "Macro R",
    "Macro F1",
    "Top-3 Acc",
    "Top-5 Acc"
]

values = [
    "0.8499 ± 0.0012",
    "0.8291 ± 0.0035",
    "0.8807 ± 0.0026",
    "0.8399 ± 0.0026",
    "0.9516 ± 0.0009",
    "0.9739 ± 0.0007"
]

column_label = "Class-weighted Cross-Entropy Loss Classifier"

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

# Build table
table_data = [[m, v] for m, v in zip(metrics, values)]
table = ax.table(
    cellText=table_data,
    colLabels=["Evaluation Metric", column_label],
    cellLoc="center",
    loc="center"
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1.3, 2.0)

plt.savefig("class_weighted_results.png", dpi=300, bbox_inches="tight")

plt.show()
# Save

plt.close()

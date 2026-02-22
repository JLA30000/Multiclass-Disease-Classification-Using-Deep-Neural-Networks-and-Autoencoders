import matplotlib.pyplot as plt

# Combined confusion data
data = [
    # Baseline classifier
    ("infectious gastroenteritis", "noninfectious gastroenteritis", 80, "Baseline"),
    ("neuralgia", "fibromyalgia", 65, "Baseline"),
    ("gum disease", "tooth disorder", 59, "Baseline"),
    ("acute glaucoma", "vitreous degeneration", 53, "Baseline"),
    ("psychotic disorder", "schizophrenia", 45, "Baseline"),

    # Autoencoder classifier
    ("infectious gastroenteritis", "noninfectious gastroenteritis", 88, "Autoencoder"),
    ("schizophrenia", "psychotic disorder", 46, "Autoencoder"),
    ("tooth disorder", "gum disease", 46, "Autoencoder"),
    ("acute glaucoma", "vitreous degeneration", 45, "Autoencoder"),
    ("depression", "post-traumatic stress disorder (PTSD)", 45, "Autoencoder"),
]

# Sort by count descending
data = sorted(data, key=lambda x: x[2], reverse=True)

# Table headers
columns = ["Disease A", "Disease B", "Count", "Classifier Type"]

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))
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
    "Top-5 confused disease pairs for baseline classifier and autoencoder classifier "
    "aggregated across 10 testing runs for a threshold of 200 samples",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("top_confused_disease_pairs_table_200.jpg", dpi=300)
plt.show()

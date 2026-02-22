import matplotlib.pyplot as plt

# Combined confusion data
data = [
    # Baseline classifier
    ("infectious gastroenteritis", "noninfectious gastroenteritis", 33, "Baseline"),
    ("gallstone", "cholecystitis", 32, "Baseline"),
    ("acute otitis media", "otitis media", 31, "Baseline"),
    ("arrhythmia", "premature atrial contractions (PACs)", 30, "Baseline"),
    ("post-traumatic stress disorder (PTSD)",
     "psychotic disorder", 28, "Baseline"),

    # Autoencoder classifier
    ("infectious gastroenteritis", "noninfectious gastroenteritis", 50, "Autoencoder"),
    ("depression", "post-traumatic stress disorder (PTSD)", 39, "Autoencoder"),
    ("corneal abrasion", "foreign body in the eye", 37, "Autoencoder"),
    ("acute bronchospasm", "pneumonia", 37, "Autoencoder"),
    ("kidney stone", "pyelonephritis", 27, "Autoencoder"),
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
    "aggregated across 10 testing runs for a threshold of 100 samples",
    fontsize=12,
    y=0.85
)

# Save and show
plt.tight_layout()
plt.savefig("top_confused_disease_pairs_table.jpg", dpi=300)
plt.show()

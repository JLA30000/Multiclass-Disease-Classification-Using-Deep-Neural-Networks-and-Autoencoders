import matplotlib.pyplot as plt

# Confusion data for class-weighted cross-entropy loss classifier
data = [
    ("infectious gastroenteritis",
     "noninfectious gastroenteritis", 331, "Class-Weighted CE"),
    ("noninfectious gastroenteritis",
     "infectious gastroenteritis", 263, "Class-Weighted CE"),
    ("cholecystitis", "gallstone", 253, "Class-Weighted CE"),
    ("skin polyp", "skin disorder", 207, "Class-Weighted CE"),
    ("depression", "post-traumatic stress disorder (PTSD)", 204, "Class-Weighted CE"),
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
    "Top-5 confused disease pairs for the class-weighted cross-entropy loss classifier "
    "aggregated across 10 testing runs",
    fontsize=12,
    y=0.75
)

# Save and show
plt.tight_layout()
plt.savefig("top_confused_disease_pairs_table_cwce.jpg", dpi=300)
plt.show()

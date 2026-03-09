import matplotlib.pyplot as plt

data = [('infectious gastroenteritis', 'noninfectious gastroenteritis', 33, 'Baseline'), ('gallstone', 'cholecystitis', 32, 'Baseline'), ('acute otitis media', 'otitis media', 31, 'Baseline'), ('arrhythmia', 'premature atrial contractions (PACs)', 30, 'Baseline'), ('post-traumatic stress disorder (PTSD)', 'psychotic disorder', 28, 'Baseline'), ('infectious gastroenteritis', 'noninfectious gastroenteritis', 50, 'Autoencoder'), ('depression', 'post-traumatic stress disorder (PTSD)', 39, 'Autoencoder'), ('corneal abrasion', 'foreign body in the eye', 37, 'Autoencoder'), ('acute bronchospasm', 'pneumonia', 37, 'Autoencoder'), ('kidney stone', 'pyelonephritis', 27, 'Autoencoder')]
data = sorted(data, key=lambda x: x[2], reverse=True)
columns = ['Disease A', 'Disease B', 'Count', 'Classifier Type']
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')
table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
plt.title('Top-5 confused disease pairs for baseline classifier and autoencoder classifier aggregated across 10 testing runs for a threshold of 100 samples', fontsize=12, y=0.85)
plt.tight_layout()
plt.savefig('top_confused_disease_pairs_table.jpg', dpi=300)
plt.show()

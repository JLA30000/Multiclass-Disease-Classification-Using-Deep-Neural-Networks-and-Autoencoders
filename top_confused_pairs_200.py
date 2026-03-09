import matplotlib.pyplot as plt

data = [('infectious gastroenteritis', 'noninfectious gastroenteritis', 80, 'Baseline'), ('neuralgia', 'fibromyalgia', 65, 'Baseline'), ('gum disease', 'tooth disorder', 59, 'Baseline'), ('acute glaucoma', 'vitreous degeneration', 53, 'Baseline'), ('psychotic disorder', 'schizophrenia', 45, 'Baseline'), ('infectious gastroenteritis', 'noninfectious gastroenteritis', 88, 'Autoencoder'), ('schizophrenia', 'psychotic disorder', 46, 'Autoencoder'), ('tooth disorder', 'gum disease', 46, 'Autoencoder'), ('acute glaucoma', 'vitreous degeneration', 45, 'Autoencoder'), ('depression', 'post-traumatic stress disorder (PTSD)', 45, 'Autoencoder')]
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
plt.title('Top-5 confused disease pairs for baseline classifier and autoencoder classifier aggregated across 10 testing runs for a threshold of 200 samples', fontsize=12, y=0.85)
plt.tight_layout()
plt.savefig('top_confused_disease_pairs_table_200.jpg', dpi=300)
plt.show()

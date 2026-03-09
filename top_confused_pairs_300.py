import matplotlib.pyplot as plt

data = [('psychotic disorder', 'schizophrenia', 99, 'Baseline'), ('infectious gastroenteritis', 'noninfectious gastroenteritis', 91, 'Baseline'), ('kidney stone', 'pyelonephritis', 80, 'Baseline'), ('gum disease', 'tooth disorder', 69, 'Baseline'), ('fibromyalgia', 'neuralgia', 68, 'Baseline'), ('infectious gastroenteritis', 'noninfectious gastroenteritis', 99, 'Autoencoder'), ('psychotic disorder', 'schizophrenia', 92, 'Autoencoder'), ('pyelonephritis', 'kidney stone', 82, 'Autoencoder'), ('fibromyalgia', 'neuralgia', 78, 'Autoencoder'), ('gum disease', 'tooth disorder', 70, 'Autoencoder')]
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
plt.title('Top-5 confused disease pairs for baseline classifier and autoencoder classifier aggregated across 10 testing runs for a threshold of 300 samples', fontsize=12, y=0.85)
plt.tight_layout()
plt.savefig('top_confused_disease_pairs_table_300.jpg', dpi=300)
plt.show()

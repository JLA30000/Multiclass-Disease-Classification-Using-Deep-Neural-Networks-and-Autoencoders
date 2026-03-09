import matplotlib.pyplot as plt

data = [('post-traumatic stress disorder (PTSD)', 100, 0.1325, 0.11, 0.1202), ('seborrheic keratosis', 100, 0.2188, 0.28, 0.2456), ('skin pigmentation disorder', 100, 0.2574, 0.26, 0.2587), ('skin polyp', 100, 0.2975, 0.36, 0.3258), ('gout', 100, 0.7982, 0.91, 0.8505), ('gastroesophageal reflux disease (GERD)', 100, 0.8431, 0.86, 0.8515), ('envenomation from spider or animal bite', 100, 0.9744, 0.76, 0.8539), ('orbital cellulitis', 100, 0.9294, 0.79, 0.8541), ('cerebral palsy', 100, 0.9901, 1.0, 0.995), ('foreign body in the throat', 100, 0.9901, 1.0, 0.995), ('cold sore', 100, 0.9804, 1.0, 0.9901), ('abdominal aortic aneurysm', 100, 0.9804, 1.0, 0.9901)]
columns = ['Disease', 'Support', 'Precision', 'Recall', 'F1']
fig, ax = plt.subplots(figsize=(16, 7))
ax.axis('off')
table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
plt.title('Baseline classifier per-class performance for a subset of representative diseases at a threshold of 100 samples', fontsize=12, y=0.85)
plt.tight_layout()
plt.savefig('per_class_performance_baseline_100.jpg', dpi=300)
plt.show()

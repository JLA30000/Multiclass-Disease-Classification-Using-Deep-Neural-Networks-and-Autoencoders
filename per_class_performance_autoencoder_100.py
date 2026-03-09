import matplotlib.pyplot as plt

data = [('skin pigmentation disorder', 100, 0.3091, 0.17, 0.2194), ('seborrheic keratosis', 100, 0.314, 0.27, 0.2903), ('post-traumatic stress disorder (PTSD)', 100, 0.2689, 0.32, 0.2922), ('skin disorder', 100, 0.402, 0.41, 0.4059), ('chlamydia', 100, 0.7734, 0.99, 0.8684), ('spondylitis', 100, 0.9524, 0.8, 0.8696), ('trigeminal neuralgia', 100, 0.7692, 1.0, 0.8696), ('hyperkalemia', 100, 0.87, 0.87, 0.87), ('cerebral palsy', 100, 1.0, 1.0, 1.0), ('stress incontinence', 100, 1.0, 1.0, 1.0), ('bladder cancer', 100, 0.9901, 1.0, 0.995), ('bladder obstruction', 100, 0.9901, 1.0, 0.995)]
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
plt.title('Autoencoder classifier per-class performance for a subset of representative diseases at a threshold of 100 samples', fontsize=12, y=0.85)
plt.tight_layout()
plt.savefig('per_class_performance_autoencoder_100.jpg', dpi=300)
plt.show()

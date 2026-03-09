import matplotlib.pyplot as plt

data = [('skin pigmentation disorder', 200, 0.4188, 0.335, 0.3722), ('seborrheic keratosis', 200, 0.4471, 0.465, 0.4559), ('dry eye of unknown cause', 200, 0.7113, 0.345, 0.4646), ('skin disorder', 200, 0.465, 0.465, 0.465), ('ankylosing spondylitis', 200, 0.8873, 0.905, 0.896), ('peripheral nerve disorder', 200, 0.9077, 0.885, 0.8962), ('lymphedema', 200, 0.8692, 0.93, 0.8986), ('common cold', 200, 0.9351, 0.865, 0.8987), ('bone disorder', 200, 0.995, 1.0, 0.9975), ('eye alignment disorder', 200, 0.995, 0.995, 0.995), ('cirrhosis', 200, 0.995, 0.99, 0.9925), ('iron deficiency anemia', 200, 1.0, 0.985, 0.9924)]
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
plt.title('Autoencoder classifier per-class performance for a subset of representative diseases at a threshold of 200 samples', fontsize=12, y=0.85)
plt.tight_layout()
plt.savefig('per_class_performance_autoencoder_200.jpg', dpi=300)
plt.show()

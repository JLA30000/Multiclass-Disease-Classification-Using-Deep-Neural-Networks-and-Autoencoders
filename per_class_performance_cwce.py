import matplotlib.pyplot as plt

data = [('raynaud disease', 10, 0.0, 0.0, 0.0, 0.0), ('hemophilia', 20, 0.0, 0.0, 0.0, 0.0), ('rhabdomyolysis', 10, 0.0, 0.0, 0.0, 0.0), ('birth trauma', 10, 0.1, 1.0, 0.1, 0.1818), ('burn', 50, 1.0, 0.7937, 1.0, 0.885), ('white blood cell disease', 510, 0.9118, 0.8627, 0.9118, 0.8866), ('anal fissure', 220, 0.9773, 0.8113, 0.9773, 0.8866), ('drug abuse (methamphetamine)', 330, 0.9121, 0.8625, 0.9121, 0.8866), ('abscess of the lung', 20, 1.0, 1.0, 1.0, 1.0), ('volvulus', 10, 1.0, 1.0, 1.0, 1.0), ('anemia due to malignancy', 10, 1.0, 1.0, 1.0, 1.0), ('peyronie disease', 20, 1.0, 1.0, 1.0, 1.0)]
columns = ['Disease', 'Support', 'Accuracy', 'Precision', 'Recall', 'F1']
fig, ax = plt.subplots(figsize=(18, 7))
ax.axis('off')
table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
plt.title('Class-weighted cross-entropy loss classifier per-class performance for a subset of representative diseases (aggregated over seeds)', fontsize=12, y=0.85)
plt.tight_layout()
plt.savefig('per_class_performance_cwce.jpg', dpi=300)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

models = ['Baseline\nClassifier', 'Autoencoder\nClassifier']
accuracies = [0.812, 0.8712]
hyperparameters = ['LR=0.1, 45 epochs', 'LR=0.0005, 45 epochs']
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(models, accuracies, color=['#E63946', '#064EF4'], alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
for i, (bar, acc, hyperparam) in enumerate(zip(bars, accuracies, hyperparameters)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{acc * 100:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width() / 2.0, 0.02, hyperparam, ha='center', va='bottom', fontsize=10, style='italic')
ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison: Highest Test Accuracy', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_yticklabels([f'{int(x * 100)}%' for x in np.arange(0, 1.1, 0.1)])
plt.tight_layout()
plt.savefig('accuracy_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Graph saved as 'accuracy_comparison.pdf'")
plt.show()
print('\n=== Model Comparison Summary ===')
print(f'Baseline Classifier: {accuracies[0] * 100:.2f}% ({hyperparameters[0]})')
print(f'Autoencoder Classifier: {accuracies[1] * 100:.2f}% ({hyperparameters[1]})')
improvement = (accuracies[1] - accuracies[0]) * 100
print(f'\nImprovement: +{improvement:.2f} percentage points ({improvement / accuracies[0] * 100:.2f}% relative increase)')

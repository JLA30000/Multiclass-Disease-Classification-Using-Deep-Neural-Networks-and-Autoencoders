import matplotlib.pyplot as plt
import numpy as np

# Best accuracies from your data
models = ['Baseline\nClassifier', 'Autoencoder\nClassifier']
accuracies = [0.8120, 0.8712]  # Baseline best: 81.2%, Autoencoder best: 87.12%

# Corresponding hyperparameters for labels
hyperparameters = ['LR=0.1, 45 epochs', 'LR=0.0005, 45 epochs']

# Create bar graph
fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.bar(models, accuracies, color=['#E63946', "#064EF4"],
              alpha=0.8, edgecolor='black', linewidth=2, width=0.6)

# Add value labels on top of bars
for i, (bar, acc, hyperparam) in enumerate(zip(bars, accuracies, hyperparameters)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc*100:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add hyperparameter info below the bar
    ax.text(bar.get_x() + bar.get_width()/2., 0.02,
            hyperparam,
            ha='center', va='bottom', fontsize=10, style='italic')

# Formatting
ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison: Highest Test Accuracy',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add percentage ticks on y-axis
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])

plt.tight_layout()

# Save as PDF
plt.savefig('accuracy_comparison.pdf', format='pdf',
            dpi=300, bbox_inches='tight')
print("Graph saved as 'accuracy_comparison.pdf'")

plt.show()

# Print summary
print("\n=== Model Comparison Summary ===")
print(f"Baseline Classifier: {accuracies[0]*100:.2f}% ({hyperparameters[0]})")
print(
    f"Autoencoder Classifier: {accuracies[1]*100:.2f}% ({hyperparameters[1]})")
improvement = (accuracies[1] - accuracies[0]) * 100
print(
    f"\nImprovement: +{improvement:.2f} percentage points ({improvement/accuracies[0]*100:.2f}% relative increase)")

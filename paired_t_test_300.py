import numpy as np
from scipy.stats import ttest_rel

# Replace these with your actual Macro F1 values (length = 10)
baseline_f1 = np.array([
    0.8622, 0.8556, 0.8613, 0.8601, 0.8596, 0.8575, 0.8561, 0.8571, 0.8612, 0.8538
])

autoencoder_f1 = np.array([
    0.8609, 0.8669, 0.8667, 0.8688, 0.8661, 0.8667, 0.8674, 0.8667, 0.8611, 0.8671
])

t_stat, p_value = ttest_rel(autoencoder_f1, baseline_f1)

print(f"t-statistic = {t_stat:.3f}")
print(f"p-value = {p_value:.5f}")

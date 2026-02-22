import numpy as np
from scipy.stats import ttest_rel

# Replace these with your actual Macro F1 values (length = 10)
baseline_f1 = np.array([
    0.8204, 0.8137, 0.8274, 0.8289, 0.8107, 0.8124, 0.8282, 0.8276, 0.8123, 0.8201
])

autoencoder_f1 = np.array([
    0.8384, 0.8416, 0.8361, 0.8418, 0.8335, 0.8415, 0.8346, 0.8440, 0.8449, 0.8449
])

t_stat, p_value = ttest_rel(autoencoder_f1, baseline_f1)

print(f"t-statistic = {t_stat:.3f}")
print(f"p-value = {p_value:.5f}")

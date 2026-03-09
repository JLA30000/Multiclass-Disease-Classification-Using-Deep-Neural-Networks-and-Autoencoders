import numpy as np
from scipy.stats import ttest_rel

baseline_f1 = np.array([0.8509, 0.8527, 0.8509, 0.8533, 0.8565, 0.8512, 0.8457, 0.8451, 0.8556, 0.8453])
autoencoder_f1 = np.array([0.858, 0.862, 0.8616, 0.8662, 0.863, 0.8659, 0.8633, 0.8681, 0.8635, 0.8641])
t_stat, p_value = ttest_rel(autoencoder_f1, baseline_f1)
print(f't-statistic = {t_stat:.3f}')
print(f'p-value = {p_value:.5f}')

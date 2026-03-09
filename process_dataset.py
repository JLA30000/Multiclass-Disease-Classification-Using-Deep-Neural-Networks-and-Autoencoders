import pandas as pd
import numpy as np

df = pd.read_csv('diseases.csv')
disease_col = df.columns[0]
disease_counts = df[disease_col].value_counts()
filtered_rows = []
np.random.seed(42)
for disease, count in disease_counts.items():
    if count < 200:
        continue
    else:
        disease_samples = df[df[disease_col] == disease]
        if len(disease_samples) == 200:
            filtered_rows.append(disease_samples)
        else:
            ss = disease_samples.sample(n=200, random_state=42)
            filtered_rows.append(ss)
if filtered_rows:
    filtered_df = pd.concat(filtered_rows, ignore_index=True)
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    filtered_df.to_csv('filtered_diseases.csv', index=False)
    print(f'Original dataset size: {len(df)} rows, {len(disease_counts)} diseases')
    print(f'Filtered dataset size: {len(filtered_df)} rows')
    filtered_counts = filtered_df[disease_col].value_counts()
    print(f'Number of diseases in filtered dataset: {len(filtered_counts)}')
    print('\nSample counts per disease in filtered dataset:')
    for disease in filtered_counts.index[:10]:
        count = filtered_counts[disease]
        print(f'  {disease}: {count} samples')
    if len(filtered_counts) > 10:
        print(f'  ... and {len(filtered_counts) - 10} more diseases')
    print(f"\nFiltered dataset saved to 'filtered_diseases.csv'")
else:
    print('No diseases had 200 or more samples!')

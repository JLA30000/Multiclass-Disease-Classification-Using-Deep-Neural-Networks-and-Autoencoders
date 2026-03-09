from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('filtered_diseases.csv')
le = LabelEncoder()
disease_encoded = le.fit_transform(df['diseases'])
print('Checking symptom predictiveness...')
print(f'Total diseases: {len(le.classes_)}')
hps = []
for col in df.columns:
    if col == 'diseases':
        continue
    symptom_present = df[col] == 1
    if symptom_present.sum() > 0:
        for disease in df['diseases'].unique():
            disease_mask = df['diseases'] == disease
            dws = df[disease_mask]
            if len(dws) > 0:
                if (dws[col] == 1).all():
                    other_diseases = df[~disease_mask]
                    if (other_diseases[col] == 1).sum() == 0:
                        hps.append((col, disease, len(dws)))
print(f'\nFound {len(hps)} perfectly predictive symptoms:')
ds = defaultdict(list)
for symptom, disease, count in hps:
    ds[disease].append((symptom, count))
print('\nDiseases with perfectly predictive symptoms:')
for disease, symptoms in list(ds.items())[:10]:
    print(f'\n{disease}:')
    for symptom, count in symptoms:
        print(f'  ✓ {symptom} (in all {count} samples)')

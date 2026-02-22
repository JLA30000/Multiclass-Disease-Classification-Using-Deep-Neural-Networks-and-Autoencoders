# check_symptom_predictiveness.py
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("filtered_diseases.csv")

# Encode diseases
le = LabelEncoder()
disease_encoded = le.fit_transform(df['diseases'])

print("Checking symptom predictiveness...")
print(f"Total diseases: {len(le.classes_)}")

# For each symptom column (excluding 'diseases')
high_predictive_symptoms = []

for col in df.columns:
    if col == 'diseases':
        continue

    # Check if symptom is always present for certain diseases
    symptom_present = df[col] == 1

    if symptom_present.sum() > 0:  # If symptom occurs at all
        # Get diseases where this symptom is always present
        for disease in df['diseases'].unique():
            disease_mask = df['diseases'] == disease
            disease_with_symptom = df[disease_mask]

            if len(disease_with_symptom) > 0:
                # If ALL samples of this disease have this symptom
                if (disease_with_symptom[col] == 1).all():
                    # And NO other diseases have this symptom
                    other_diseases = df[~disease_mask]
                    if (other_diseases[col] == 1).sum() == 0:
                        high_predictive_symptoms.append(
                            (col, disease, len(disease_with_symptom)))

print(f"\nFound {len(high_predictive_symptoms)} perfectly predictive symptoms:")

# Group by disease
disease_symptoms = defaultdict(list)

for symptom, disease, count in high_predictive_symptoms:
    disease_symptoms[disease].append((symptom, count))

# Print diseases with perfectly predictive symptoms
print("\nDiseases with perfectly predictive symptoms:")
for disease, symptoms in list(disease_symptoms.items())[:10]:  # First 10
    print(f"\n{disease}:")
    for symptom, count in symptoms:
        print(f"  ✓ {symptom} (in all {count} samples)")

import pandas as pd
import numpy as np

# Read CSV
df = pd.read_csv("diseases.csv")

# The first column contains disease labels
disease_col = df.columns[0]

# Count how many times each disease appears
disease_counts = df[disease_col].value_counts()

# Filter the dataframe
filtered_rows = []

# Set random seed for reproducibility
np.random.seed(42)

# Process each disease
for disease, count in disease_counts.items():
    if count < 200:
        # Skip diseases with less than 200 samples
        continue
    else:
        # Get all samples for this disease
        disease_samples = df[df[disease_col] == disease]

        # If exactly 200 samples, keep all
        if len(disease_samples) == 200:
            filtered_rows.append(disease_samples)
        else:
            # Randomly select exactly 200 samples
            # Use .sample() with random_state for reproducibility
            selected_samples = disease_samples.sample(n=200, random_state=42)
            filtered_rows.append(selected_samples)

# Combine all filtered rows
if filtered_rows:
    filtered_df = pd.concat(filtered_rows, ignore_index=True)

    # Optional: Shuffle the final dataset
    filtered_df = filtered_df.sample(
        frac=1, random_state=42).reset_index(drop=True)

    # Save filtered CSV
    filtered_df.to_csv("filtered_diseases.csv", index=False)

    # Print statistics
    print(
        f"Original dataset size: {len(df)} rows, {len(disease_counts)} diseases")
    print(f"Filtered dataset size: {len(filtered_df)} rows")

    # Count diseases in filtered dataset
    filtered_counts = filtered_df[disease_col].value_counts()
    print(f"Number of diseases in filtered dataset: {len(filtered_counts)}")

    # Verify each disease has exactly 200 samples
    print("\nSample counts per disease in filtered dataset:")
    for disease in filtered_counts.index[:10]:  # Show first 10
        count = filtered_counts[disease]
        print(f"  {disease}: {count} samples")

    if len(filtered_counts) > 10:
        print(f"  ... and {len(filtered_counts) - 10} more diseases")

    print(f"\nFiltered dataset saved to 'filtered_diseases.csv'")
else:
    print("No diseases had 200 or more samples!")

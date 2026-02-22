import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_histogram(csv_file_path):
    """
    Create a simple histogram of disease sample distribution
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Identify disease column
    disease_column = 'disease'
    if disease_column not in df.columns:
        potential_columns = [
            col for col in df.columns if 'disease' in col.lower() or 'label' in col.lower()]
        disease_column = potential_columns[0] if potential_columns else df.columns[-1]
        print(f"Using column '{disease_column}' for disease analysis")

    # Count samples per disease and sort
    disease_counts = df[disease_column].value_counts(
    ).sort_values(ascending=False)

    # Print summary statistics
    print("=" * 50)
    print("DISEASE SAMPLE ANALYSIS")
    print("=" * 50)
    print(f"Total diseases: {len(disease_counts)}")
    print(f"Total samples: {len(df)}")
    print(
        f"\nMost common: {disease_counts.index[0]} ({disease_counts.iloc[0]} samples)")
    print(
        f"Least common: {disease_counts.index[-1]} ({disease_counts.iloc[-1]} samples)")

    # Create simple figure
    plt.figure(figsize=(12, 6))

    # Create bars
    plt.bar(range(len(disease_counts)), disease_counts.values, color='blue')

    # Labels
    plt.title('Disease Sample Distribution')
    plt.ylabel('Sample Count')
    plt.xlabel('Disease Types')

    # Remove x-axis ticks
    plt.xticks([])

    # Save
    plt.tight_layout()
    plt.savefig('disease_distribution.pdf', format='pdf',
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Histogram saved as 'disease_distribution.pdf'")

    plt.show()

    return disease_counts


if __name__ == "__main__":
    csv_file = "diseases.csv"

    try:
        disease_counts = create_histogram(csv_file)

        # Show top 5 for reference
        print("\nTop 5 diseases:")
        for i, (disease, count) in enumerate(disease_counts.head().items(), 1):
            print(f"  {i}. {disease}: {count}")

    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'")
    except Exception as e:
        print(f"Error: {e}")

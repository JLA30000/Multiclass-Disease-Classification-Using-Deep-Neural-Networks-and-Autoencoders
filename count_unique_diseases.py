# Replace 'data.csv' with the path to your file if needed
filename = 'filtered_diseases.csv'

# Since your file is a CSV, we need to read it as text
with open(filename, 'r') as f:
    lines = f.readlines()

# Extract the first column (disease names)
diseases = [line.split(',')[0] for line in lines[1:]]  # skip header

# Count unique diseases
unique_diseases = set(diseases)
print("Number of unique diseases:", len(unique_diseases))
print("Unique diseases:", unique_diseases)

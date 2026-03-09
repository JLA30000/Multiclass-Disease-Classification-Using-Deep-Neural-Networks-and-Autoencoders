filename = 'filtered_diseases.csv'
with open(filename, 'r') as f:
    lines = f.readlines()
diseases = [line.split(',')[0] for line in lines[1:]]
unique_diseases = set(diseases)
print('Number of unique diseases:', len(unique_diseases))
print('Unique diseases:', unique_diseases)

import pandas as pd
import uuid

# Read the dataset (space-separated, no header)
data = pd.read_csv('drug_consumption.data', sep=',', header=None)

# Define column names based on UCI dataset description, plus placeholders for extra columns
columns = [
    'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Choc',
    'Coke', 'Ecstasy', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mush', 'Nicotine',
    'Semer', 'VSA', 'Unknown1', 'Unknown2'
]
data.columns = columns

# Binarize Cannabis and Nicotine columns
# Non-user (0): CL0 or CL1
# User (1): CL2, CL3, CL4, CL5, CL6
data['Cannabis_Use'] = data['Cannabis'].apply(lambda x: 0 if x in ['CL0', 'CL1'] else 1)
data['Nicotine_Use'] = data['Nicotine'].apply(lambda x: 0 if x in ['CL0', 'CL1'] else 1)

# Save as CSV file with new columns
data.to_csv('drug_consumption_processed.csv', index=False)

print("Processed dataset saved as 'drug_consumption_processed.csv'")
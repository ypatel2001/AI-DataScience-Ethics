import pandas as pd

# Define protected classes and their subgroups with exact capitalizations
protected_classes = {
    'Sexual Orientation': ['lesbian', 'gay', 'bisexual', 'queer', 'homosexual', 'straight', 'heterosexual', 'lgbt', 'lgbtq'],
    'Gender': ['male', 'female', 'nonbinary', 'transgender', 'trans'],
    'Race/National Origin': ['african', 'african american', 'black', 'white', 'european', 'asian', 'indian', 'middle eastern', 'hispanic', 'latino', 'latina', 'latinx', 'mexican', 'canadian', 'american', 'chinese', 'japanese'],
    'Religion': ['christian', 'muslim', 'buddhist', 'catholic', 'protestant', 'sikh', 'taoist'],
    'Age': ['old', 'older', 'young', 'younger', 'teenage', 'millenial', 'middle aged', 'elderly'],
    'Disability': ['blind', 'deaf', 'paralyzed']
}

# Get all subgroup columns
all_subgroups = [sub for pc in protected_classes.values() for sub in pc]

# Load the dataset
df = pd.read_csv('toxity_per_attribute.csv')
original_rows = len(df)

# Handle TOXICITY: convert to numeric and drop NaN
df['TOXICITY'] = pd.to_numeric(df['TOXICITY'], errors='coerce')
df = df.dropna(subset=['TOXICITY'])
cleaned_rows = len(df)
print(f"Removed {original_rows - cleaned_rows} rows with invalid TOXICITY values.")

# Convert subgroup columns to boolean
for sub in all_subgroups:
    if sub in df.columns:
        df[sub] = df[sub].apply(lambda x: str(x).lower() == 'true')
    else:
        print(f"Warning: Column {sub} not found in dataframe.")

# Get existing subgroup columns
existing_subgroups = [sub for sub in all_subgroups if sub in df.columns]
if len(existing_subgroups) < len(all_subgroups):
    missing = set(all_subgroups) - set(df.columns)
    print(f"Warning: The following subgroups are missing from the dataset: {missing}")

# Create reduced dataset: keep rows where at least one subgroup is True
reduced_df = df[df[existing_subgroups].any(axis=1)]
reduced_rows = len(reduced_df)
print(f"Reduced dataset has {reduced_rows} rows (removed {cleaned_rows - reduced_rows} rows with all FALSE subgroups).")

# Save reduced dataset to CSV
reduced_df.to_csv('reduced_dataset.csv', index=False)

print("Reduced dataset saved as 'reduced_dataset.csv'.")
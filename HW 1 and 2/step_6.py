import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
df = pd.read_csv('mental-health-in-tech-survey-2019.csv')

# Step 2: Define column names
gender_col = 'What is your gender?'
treatment_col = 'Have you ever sought treatment for a mental health disorder from a mental health professional?'

# Step 3: Drop rows with missing values in gender or treatment columns
df = df.dropna(subset=[gender_col, treatment_col])

# Step 4: Define the cleaning function for gender
def clean_gender(gender):
    if pd.isna(gender):
        return "Other"
    gender_lower = str(gender).lower().strip()
    man_list = {"male", "m", "man", "cishet male", "trans man", "identify as male", "cis male", "masculine", "let's keep it simple and say 'male'", "cis male"}
    woman_list = {"female", "f", "woman", "female-identified", "woman", "female (cis)", "femile", "femmina"}
    if gender_lower in man_list:
        return "Man"
    elif gender_lower in woman_list:
        return "Woman"
    else:
        return "Other"

# Apply cleaning to create a new column
df['Gender_Cleaned'] = df[gender_col].apply(clean_gender)

# Step 5: Clean treatment column to standardize values
def clean_treatment(treatment):
    treatment_str = str(treatment).strip().upper()
    if treatment_str in ['YES', 'TRUE']:
        return 'Yes'
    elif treatment_str in ['NO', 'FALSE']:
        return 'No'
    return treatment  # keep original if not matching

df['Treatment_Cleaned'] = df[treatment_col].apply(clean_treatment)

# Step 6: Randomly select 50% of the dataset with random state 35
df_reduced = df.sample(frac=0.5, random_state=35)

# Step 7: Create the frequency table with new label format
print("Independent Variable - Gender")
print("Dependent Variable - Treatment Y")
print("Dependent Variable - Treatment N")

# Prepare data for CSV with new column headers
csv_data = {
    'Independent Variable - Gender': [],
    'Dependent Variable - Treatment Y': [],
    'Dependent Variable - Treatment N': []
}

for gender in sorted(df_reduced['Gender_Cleaned'].unique()):
    subset = df_reduced[df_reduced['Gender_Cleaned'] == gender]
    yes_count = subset[subset['Treatment_Cleaned'] == 'Yes'].shape[0]
    no_count = subset[subset['Treatment_Cleaned'] == 'No'].shape[0]
    
    print(f"{gender} – Frequency of Y: {yes_count} Frequency of N: {no_count}")
    
    # Add data with new format
    csv_data['Independent Variable - Gender'].append(gender)
    csv_data['Dependent Variable - Treatment Y'].append(yes_count)
    csv_data['Dependent Variable - Treatment N'].append(no_count)

# Save frequency table to CSV with new headers
freq_df = pd.DataFrame(csv_data)
freq_df.to_csv('frequency_table.csv', index=False)

# Step 8: Create the histogram using seaborn
plt.figure(figsize=(12, 8))
ax = sns.countplot(x='Gender_Cleaned', hue='Treatment_Cleaned', data=df_reduced, palette='Set2')
plt.title('Frequency of Mental Health Treatment Access by Gender (Random 50% Data Sample)')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Modify the legend title
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['No', 'Yes'], title='Sought Treatment')

plt.tight_layout()
plt.savefig('frequency_graph.png')
plt.show()
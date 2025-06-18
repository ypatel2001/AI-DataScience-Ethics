import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
df = pd.read_csv('mental-health-in-tech-survey-2019.csv')

# Step 2: Define column names (replace with actual column names from the CSV)
# These are placeholders; update them based on the actual column names in your dataset
gender_col = 'What is your gender?'  # Replace with actual column name for gender
treatment_col = 'Have you ever sought treatment for a mental health disorder from a mental health professional?'  # Replace with actual column name for treatment

# Step 3: Clean the dataset
# Drop rows with missing values in the specified columns
df = df.dropna(subset=[gender_col, treatment_col])

# Step 4: Randomly select 50% of the dataset with random state 35
df_reduced = df.sample(frac=0.5, random_state=35)

# Step 5: Create the frequency table
# Get unique gender categories
genders = df_reduced[gender_col].unique()

# Print the frequency table in the specified format
print("Independent Variable - Protected Class Variable")
print("Dependent Variable - Mental Health Treatment Y")
print("Dependent Variable - Mental Health Treatment - N")

for gender in genders:
    subset = df_reduced[df_reduced[gender_col] == gender]
    # Assume treatment_col has 'Yes' and 'No' (adjust if different)
    yes_count = subset[subset[treatment_col] == 'Yes'].shape[0]
    no_count = subset[subset[treatment_col] == 'No'].shape[0]
    
    print()

# Step 6: Create the histogram using seaborn
plt.figure(figsize=(12, 8))
sns.countplot(x=gender_col, hue=treatment_col, data=df_reduced, palette='Set2')
plt.title('Frequency of Mental Health Treatment by Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('frequency_graph.png')  # Save the graph as an image
plt.show()  # Display the graph (if running in an interactive environment)
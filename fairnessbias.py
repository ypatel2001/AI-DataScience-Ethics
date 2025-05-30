import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names
age_col = "What is your age?"
treatment_col = "Have you ever sought treatment for a mental health disorder from a mental health professional?"

# Load the dataset
df = pd.read_csv('mental-health-in-tech-survey-2019.csv')

# Clean the dataset
# Convert treatment to numeric (TRUE -> 1, FALSE -> 0)
df[treatment_col] = df[treatment_col].map({'TRUE': 1, 'FALSE': 0})

# Convert age to numeric and handle missing values
df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
df = df.dropna(subset=[age_col, treatment_col])

# Bin age into groups
df['age_group'] = pd.cut(df[age_col], bins=[0, 20, 30, 40, 50, 60, 100], labels=['0-19', '20-29', '30-39', '40-49', '50-59', '60+'])

# Compute the percentage of treatment sought for each age group
treatment_by_age = df.groupby('age_group')[treatment_col].mean() * 100

# Fairness graph (full y-axis scale)
plt.figure(figsize=(10, 6))
sns.barplot(x=treatment_by_age.index, y=treatment_by_age.values, color='#4CAF50')
plt.title('Percentage of Individuals Who Sought Treatment by Age Group')
plt.xlabel('Age Group', labelpad=5)
plt.ylabel('Percentage', labelpad=5)
plt.ylim(0, 100)  # Full scale to minimize differences
plt.savefig('fairness_treatment_by_age.png')
plt.close()

# Bias graph (truncated y-axis scale)
plt.figure(figsize=(10, 6))
sns.barplot(x=treatment_by_age.index, y=treatment_by_age.values, color='#F44336')
plt.title('Percentage of Individuals Who Sought Treatment by Age Group')
plt.xlabel('Age Group', labelpad=5)
plt.ylabel('Percentage', labelpad=5)
plt.ylim(25, 35)  # Truncated scale to exaggerate differences
plt.savefig('bias_treatment_by_age.png')
plt.close()
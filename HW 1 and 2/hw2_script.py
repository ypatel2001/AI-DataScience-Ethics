import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names from the dataset
treatment_col = "Have you ever sought treatment for a mental health disorder from a mental health professional?"
mental_health_disclosure_col = "Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?"
mental_health_support_col = "Overall, how well do you think the tech industry supports employees with mental health issues?"
age_col = "What is your age?"
race_col = "What is your race?"
current_mental_health_disorder_col = "Do you *currently* have a mental health disorder?"

# Read the CSV file
df = pd.read_csv('mental-health-in-tech-survey-2019.csv')

# Data Cleaning: Create age groups for the continuous 'age' variable
df['age_group'] = pd.cut(df[age_col], bins=[0, 20, 30, 40, 50, 60, 100], labels=['0-19', '20-29', '30-39', '40-49', '50-59', '60+'])

# Define dependent and protected variables
dependent_vars = {
    'treatment': treatment_col,
    'mental_health_disclosure': mental_health_disclosure_col,
    'mental_health_support': mental_health_support_col
}

protected_vars = {
    'age': 'age_group',
    'current_mental_health_disorder': current_mental_health_disorder_col,
    'race': race_col
}

# Function to convert snake_case to title case
def to_title_case(s):
    return ' '.join(word.capitalize() for word in s.replace('_', ' ').split())

# Compute frequencies and create tables and histograms
for dep_var_key, dep_var_col in dependent_vars.items():
    for prot_var_key, group_col in protected_vars.items():
        # Subset data and drop rows with missing values
        df_subset = df[[dep_var_col, group_col]].dropna()
        
        # Compute frequency table
        freq_table = df_subset.groupby(group_col)[dep_var_col].value_counts(normalize=False).unstack(fill_value=0)
        
        # Print frequency table for verification
        print(f"Independent Variable - {to_title_case(prot_var_key)}")
        for dep_value in freq_table.columns:
            print(f"Dependent Variable - {to_title_case(dep_var_key)} - {dep_value}")
        for prot_value in freq_table.index:
            freq_str = f"{to_title_case(prot_var_key)} - {prot_value}: "
            freq_str += ', '.join(f"Frequency of {dep_value}: {freq_table.loc[prot_value, dep_value]}" for dep_value in freq_table.columns)
            print(freq_str)
        print()
        
        # Write frequency table to CSV in new format
        with open(f"{dep_var_key}_by_{prot_var_key}.csv", 'w') as f:
            independent_var_name = to_title_case(prot_var_key)
            dependent_var_name = to_title_case(dep_var_key)
            dependent_cats = list(freq_table.columns)
            # Write header
            header = f"Independent Variable - {independent_var_name},{','.join([f'Dependent Variable - {dependent_var_name} - {cat}' for cat in dependent_cats])}"
            f.write(header + '\n')
            # Write data rows
            for ind_cat in freq_table.index:
                row_data = [f"{independent_var_name} - {ind_cat}"] + [f"Frequency of {dep_cat}: {freq_table.loc[ind_cat, dep_cat]}" for dep_cat in dependent_cats]
                row = ','.join(map(str, row_data))
                f.write(row + '\n')
        
        # Create vertical bar chart
        plt.figure(figsize=(12, 6))
        sns.countplot(x=group_col, hue=dep_var_col, data=df_subset, palette='Set2')
        plt.title(f"Frequency of {to_title_case(dep_var_key)} by {to_title_case(prot_var_key)}")
        plt.xlabel(to_title_case(prot_var_key), labelpad=5)
        plt.ylabel("Frequency", labelpad=5)
        plt.legend(title=to_title_case(dep_var_key), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{dep_var_key}_by_{prot_var_key}.png")
        plt.close()
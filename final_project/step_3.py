import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover

# Load and prepare the dataset
try:
    df = pd.read_csv('drug_consumption_processed.csv')
except FileNotFoundError:
    print("Error: 'drug_consumption_processed.csv' not found. Ensure the file is in the working directory.")
    exit(1)

# Select relevant columns
df = df[['Gender', 'Age', 'Cannabis_Use', 'Nicotine_Use']]

# Create binary columns for protected attributes
df['Gender_binary'] = (df['Gender'] == -0.48246).astype(int)  # 1 = Male (privileged), 0 = Female (unprivileged)
df['Age_binary'] = (df['Age'] >= 0.49788).astype(int)        # 1 = Older (privileged), 0 = Younger (unprivileged)

# Define protected attributes and outcomes
protected_attributes = ['Gender_binary', 'Age_binary']
outcome_variables = ['Cannabis_Use', 'Nicotine_Use']

# Function to compute SPD and DI
def compute_fairness_metrics(df, attr, outcome):
    privileged = df[df[attr] == 1]
    unprivileged = df[df[attr] == 0]
    p_priv = privileged[outcome].mean() if len(privileged) > 0 else 0
    p_unpriv = unprivileged[outcome].mean() if len(unprivileged) > 0 else 0
    spd = p_priv - p_unpriv
    di = p_unpriv / p_priv if p_priv > 0 else 'NaN'
    return spd, di

# Compute fairness metrics for original dataset
original_results = []
for attr in protected_attributes:
    for outcome in outcome_variables:
        spd, di = compute_fairness_metrics(df, attr, outcome)
        original_results.append({
            'Protected Attribute': attr.replace('_binary', ''),
            'Outcome': outcome,
            'Statistical Parity Difference': round(spd, 4) if isinstance(spd, (int, float)) else "N/A",
            'Disparate Impact': round(di, 4) if isinstance(di, (int, float)) else "NaN"
        })

# Print and save original metrics
original_metrics_df = pd.DataFrame(original_results)
print("\nOriginal Dataset Fairness Metrics (CSV):")
print(original_metrics_df.to_csv(index=False, na_rep='NaN'))
original_metrics_df.to_csv('original_fairness_metrics.csv', index=False, na_rep='NaN')

# Apply Disparate Impact Remover for Cannabis_Use with respect to Gender
dataset = BinaryLabelDataset(
    df=df[['Gender', 'Cannabis_Use']],
    label_names=['Cannabis_Use'],
    protected_attribute_names=['Gender'],
    favorable_label=1,
    unprivileged_protected_attributes=[[0.48246]],
    privileged_protected_attributes=[[-0.48246]]
)
remover = DisparateImpactRemover(repair_level=1.0)
transformed_dataset = remover.fit_transform(dataset)

# Convert transformed dataset to DataFrame
transformed_df = transformed_dataset.convert_to_dataframe()[0]
transformed_df['Age'] = df['Age']
transformed_df['Nicotine_Use'] = df['Nicotine_Use']
transformed_df['Gender_binary'] = df['Gender_binary']
transformed_df['Age_binary'] = df['Age_binary']

# Compute fairness metrics for transformed dataset
transformed_results = []
for attr in protected_attributes:
    for outcome in outcome_variables:
        spd, di = compute_fairness_metrics(transformed_df, attr, outcome)
        transformed_results.append({
            'Protected Attribute': attr.replace('_binary', ''),
            'Outcome': outcome,
            'Statistical Parity Difference': round(spd, 4) if isinstance(spd, (int, float)) else "N/A",
            'Disparate Impact': round(di, 4) if isinstance(di, (int, float)) else "NaN"
        })

# Print and save transformed metrics
transformed_metrics_df = pd.DataFrame(transformed_results)
print("\nTransformed Dataset Fairness Metrics (CSV):")
print(transformed_metrics_df.to_csv(index=False, na_rep='NaN'))
transformed_metrics_df.to_csv('transformed_fairness_metrics.csv', index=False, na_rep='NaN')
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
import matplotlib.pyplot as plt

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
df['Age_binary'] = (df['Age'] >= 0.49788).astype(int)        # 1 = Older (privileged, ≥35), 0 = Younger (<35)

# Define protected attributes and outcomes
protected_attributes = ['Gender_binary', 'Age_binary']
outcome_variables = ['Cannabis_Use', 'Nicotine_Use']

# Define comparison labels for clarity
comparison_labels = {
    'Gender_binary': 'Male vs. Female',
    'Age_binary': 'Younger vs. Older'
}

# Function to compute fairness metrics
def compute_fairness_metrics(df, attr, outcome, weights=None):
    privileged = df[df[attr] == 1]
    unprivileged = df[df[attr] == 0]
    if weights is None:
        p_priv = privileged[outcome].mean() if len(privileged) > 0 else 0
        p_unpriv = unprivileged[outcome].mean() if len(unprivileged) > 0 else 0
    else:
        priv_weights = weights[privileged.index]
        unpriv_weights = weights[unprivileged.index]
        p_priv = np.sum(priv_weights * privileged[outcome]) / np.sum(priv_weights) if len(privileged) > 0 and np.sum(priv_weights) > 0 else 0
        p_unpriv = np.sum(unpriv_weights * unprivileged[outcome]) / np.sum(unpriv_weights) if len(unprivileged) > 0 and np.sum(unpriv_weights) > 0 else 0
    spd = p_priv - p_unpriv
    di = p_unpriv / p_priv if p_priv > 0 else 'NaN'
    return spd, di

# Compute original fairness metrics
original_results = []
for attr in protected_attributes:
    for outcome in outcome_variables:
        spd, di = compute_fairness_metrics(df, attr, outcome)
        original_results.append({
            'Comparison': f"{comparison_labels[attr]} {outcome}",
            'Statistical Parity Difference': round(spd, 4),
            'Disparate Impact': round(di, 4) if di != 'NaN' else 'NaN'
        })

# Apply Reweighting for Gender and Age with respect to Cannabis_Use
weights_dict = {}
for attr in protected_attributes:
    dataset = BinaryLabelDataset(
        df=df[[attr, 'Cannabis_Use']],
        label_names=['Cannabis_Use'],
        protected_attribute_names=[attr],
        favorable_label=1,
        unprivileged_protected_attributes=[[0]],
        privileged_protected_attributes=[[1]]
    )
    rw = Reweighing(unprivileged_groups=[{attr: 0}], privileged_groups=[{attr: 1}])
    try:
        transformed_dataset = rw.fit_transform(dataset)
        weights_dict[attr] = transformed_dataset.instance_weights
        print(f"\nDiagnostic - Reweighting successful for {comparison_labels[attr]}, Cannabis_Use. First 10 weights:", transformed_dataset.instance_weights[:10])
    except Exception as e:
        print(f"\nReweighting failed for {comparison_labels[attr]}, Cannabis_Use: {e}")
        weights_dict[attr] = np.ones(len(df))  # Fallback to equal weights

# Combine weights for Gender and Age (product of weights)
combined_weights = weights_dict['Gender_binary'] * weights_dict['Age_binary']

# Add combined weights to the DataFrame and save as CSV
df['Weight'] = combined_weights
df.to_csv('drug_dataset_reweighted.csv', index=False)

# Compute transformed fairness metrics
transformed_results = []
for attr in protected_attributes:
    for outcome in outcome_variables:
        spd, di = compute_fairness_metrics(df, attr, outcome, combined_weights)
        transformed_results.append({
            'Comparison': f"{comparison_labels[attr]} {outcome}",
            'Statistical Parity Difference': round(spd, 4),
            'Disparate Impact': round(di, 4) if di != 'NaN' else 'NaN'
        })

# Create DataFrames
original_metrics_df = pd.DataFrame(original_results)
transformed_metrics_df = pd.DataFrame(transformed_results)

# Plotting SPD
plt.figure(figsize=(12, 9))
bar_width = 0.35
index = np.arange(len(original_results))
bar1 = plt.bar(index, original_metrics_df['Statistical Parity Difference'], bar_width, label='Before Reweighting (Privileged: Male, Older; Unprivileged: Female, Younger)', color='blue')
bar2 = plt.bar(index + bar_width, transformed_metrics_df['Statistical Parity Difference'], bar_width, label='After Reweighting (Privileged: Male, Older; Unprivileged: Female, Younger)', color='red')
plt.xlabel('Comparison')
plt.ylabel('Statistical Parity Difference (SPD)')
plt.title('Statistical Parity Difference of Drug Use Dataset Before and After Reweighting')
plt.xticks(index + bar_width / 2, original_metrics_df['Comparison'], rotation=45, ha='right')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('spd_plot.png')
plt.show()

# Plotting DI (excluding NaN values)
plt.figure(figsize=(12, 9))
valid_di = original_metrics_df['Disparate Impact'] != 'NaN'
valid_comparisons = original_metrics_df[valid_di]['Comparison']
valid_original_di = original_metrics_df[valid_di]['Disparate Impact'].astype(float)
valid_transformed_di = transformed_metrics_df[valid_di]['Disparate Impact'].astype(float)
index = np.arange(len(valid_comparisons))
bar1 = plt.bar(index, valid_original_di, bar_width, label='Before Reweighting (Privileged: Male; Unprivileged: Female)', color='blue')
bar2 = plt.bar(index + bar_width, valid_transformed_di, bar_width, label='After Reweighting (Privileged: Male; Unprivileged: Female)', color='red')
plt.xlabel('Comparison')
plt.ylabel('Disparate Impact (DI)')
plt.title('Disparate Impact of Drug Use Dataset Before and After Reweighting')
plt.xticks(index + bar_width / 2, valid_comparisons, rotation=45, ha='right')
plt.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('di_plot.png')
plt.show()

# Output results
print("Before Reweighting Fairness Metrics (CSV):")
print(original_metrics_df.to_csv(index=False, na_rep='NaN'))

print("After Reweighting Fairness Metrics (CSV):")
print(transformed_metrics_df.to_csv(index=False, na_rep='NaN'))

# Save to CSV files
original_metrics_df.to_csv('before_fairness_metrics.csv', index=False, na_rep='NaN')
transformed_metrics_df.to_csv('after_fairness_metrics.csv', index=False, na_rep='NaN')
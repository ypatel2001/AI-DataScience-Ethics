import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the fairness metrics data as provided
fairness_data = {
    'Metric': ['Disparate Impact', 'Equal Opportunity Difference'],
    'Value': [1.02, 0.01],
    'Ideal': [1.0, 0.0],
    'Bias Threshold': ['<0.8', '<-0.1'],
    'Bias Indication': ['Fair', 'Fair']
}

# Create DataFrame
fairness_table = pd.DataFrame(fairness_data)

# Save Fairness Metrics Table to CSV
fairness_table.to_csv('fairness_metrics_table.csv', index=False)
print("\nFairness Metrics Table:")
print(fairness_table)

# Plot 1: Disparate Impact
plt.figure(figsize=(6, 4))
plt.bar('Disparate Impact', 1.02, color='skyblue')
plt.axhline(y=1.0, color='blue', linestyle='--', label='Ideal DI (1.0)')
plt.axhline(y=0.8, color='red', linestyle='--', label='DI Bias Threshold (0.8)')
plt.ylim(0, 1.5)
for bar in plt.bar('Disparate Impact', [1.02]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
plt.title('Disparate Impact for Age Groups')
plt.ylabel('Ratio')
plt.legend()
plt.tight_layout()
plt.savefig('disparate_impact.png')
plt.close()

# Plot 2: Equal Opportunity Difference
plt.figure(figsize=(6, 4))
plt.bar('Equal Opportunity Difference', 0.01, color='lightgreen')
plt.axhline(y=0.0, color='green', linestyle='--', label='Ideal EOD (0.0)')
plt.axhline(y=-0.1, color='red', linestyle='--', label='EOD Bias Threshold (-0.1)')
plt.ylim(-0.2, 0.2)
for bar in plt.bar('Equal Opportunity Difference', [0.01]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
plt.title('Equal Opportunity Difference for Age Groups')
plt.ylabel('Difference')
plt.legend()
plt.tight_layout()
plt.savefig('equal_opportunity_difference.png')
plt.close()

# Explanation for Section 5
explanation = """
The Disparate Impact (DI) metric, with a value of 1.02, measures the ratio of loan approval rates between the unprivileged (Older >=40) and privileged (Younger <40) groups. An ideal value of 1.0 indicates equal treatment, while a value below 0.8 (a standard threshold from fairness guidelines) suggests bias favoring the privileged group. Here, the DI meets or exceeds the threshold of 0.8, indicating fair treatment.

The Equal Opportunity Difference (EOD) metric, with a value of 0.01, assesses the difference in true positive rates (correct approvals for good credit risks) between groups. An ideal value of 0.0 signifies fairness, with bias indicated if EOD < -0.1 (a common threshold in fairness research). The EOD meets or exceeds the threshold of -0.1, suggesting fairness in true positive rates.

The results both suggest fairness, as both metrics fall within acceptable ranges. The threshold of 0.8 for DI and -0.1 for EOD are adopted from established fairness standards (e.g., EEOC Four-Fifths Rule and equality of opportunity research) rather than calculated from the dataset, providing a consistent benchmark.
"""
print("\nSection 5 Explanation:")
print(explanation)
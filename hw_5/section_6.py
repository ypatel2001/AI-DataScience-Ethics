import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the German Credit Dataset
column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_plans', 'housing', 'existing_credits',
    'job', 'liable_persons', 'telephone', 'foreign_worker', 'class'
]

try:
    df = pd.read_csv('german.data', sep='\s+', header=None, names=column_names, engine='python')
except FileNotFoundError:
    print("Error: 'german.data' file not found. Please ensure the file is in the working directory.")
    exit(1)

# Convert numeric columns and handle missing values
for col in ['age', 'duration', 'credit_amount', 'installment_rate']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

# Split into training and testing sets (50% training)
train_df, _ = train_test_split(df, test_size=0.5, random_state=42)

# Define creditworthiness calculation function (from Step 3.2)
def calculate_creditworthiness(row):
    checking_account_score = {'A14': 1.0, 'A13': 0.75, 'A12': 0.5, 'A11': 0.0}.get(row['checking_account'], 0.0)
    credit_history_score = {'A34': 1.0, 'A32': 0.75, 'A33': 0.5, 'A31': 0.25, 'A30': 0.0}.get(row['credit_history'], 0.0)
    savings_account_score = {'A64': 1.0, 'A63': 0.75, 'A62': 0.5, 'A61': 0.25, 'A65': 0.0}.get(row['savings_account'], 0.0)
    duration_score = 1 - (row['duration'] / 72)
    credit_amount_score = 1 - (row['credit_amount'] / 20000)
    installment_rate_score = 1 - (row['installment_rate'] / 4)
    score = (0.25 * checking_account_score + 0.30 * credit_history_score + 0.20 * savings_account_score +
             0.10 * duration_score + 0.10 * credit_amount_score + 0.05 * installment_rate_score) * 100
    return min(max(score, 0), 100)

# Apply creditworthiness calculation
train_df['creditworthiness'] = train_df.apply(calculate_creditworthiness, axis=1)

# Define groups based on age
train_df['group'] = train_df['age'].apply(lambda x: 'Younger (<40)' if x < 40 else 'Older (>=40)')

# Step 6.1: Select Different Thresholds
# Initial threshold range based on creditworthiness distribution (0 to 100)
threshold_range = np.arange(20, 50, 2)  # Test thresholds from 20 to 48
priv_thresholds = []
unpriv_thresholds = []

# Step 6.2: Optimize Thresholds to Minimize Bias and Maximize Profit
def calculate_disparate_impact(df, priv_threshold, unpriv_threshold):
    priv_group = df[df['group'] == 'Younger (<40)']
    unpriv_group = df[df['group'] == 'Older (>=40)']
    priv_approved = (priv_group['creditworthiness'] >= priv_threshold).mean()
    unpriv_approved = (unpriv_group['creditworthiness'] >= unpriv_threshold).mean()
    if priv_approved == 0 or np.isnan(priv_approved) or np.isnan(unpriv_approved):
        return np.inf
    return abs(unpriv_approved / priv_approved - 1.0)  # Minimize deviation from 1.0

def calculate_profit(df, priv_threshold, unpriv_threshold):
    total_profit = 0
    for index, row in df.iterrows():
        threshold = priv_threshold if row['group'] == 'Younger (<40)' else unpriv_threshold
        approved = row['creditworthiness'] >= threshold
        if approved and row['class'] == 1:  # Good credit
            total_profit += 1000  # Profit per approved good credit
        elif approved and row['class'] == 2:  # Bad credit
            total_profit -= 500  # Loss per approved bad credit
    return total_profit

best_di = float('inf')
best_profit = float('-inf')
best_priv_threshold = 30
best_unpriv_threshold = 30

# Grid search for optimal thresholds
for priv_t in threshold_range:
    for unpriv_t in threshold_range:
        di = calculate_disparate_impact(train_df, priv_t, unpriv_t)
        profit = calculate_profit(train_df, priv_t, unpriv_t)
        if di < best_di and profit > best_profit:  # Prioritize fairness, then profit
            best_di = di
            best_profit = profit
            best_priv_threshold = priv_t
            best_unpriv_threshold = unpriv_t

print(f"Optimized Thresholds - Privileged (Younger <40): {best_priv_threshold}, Unprivileged (Older >=40): {best_unpriv_threshold}")
print(f"Best Disparate Impact Deviation: {best_di:.4f}, Best Profit: ${best_profit:.2f}")

# Apply optimized thresholds
train_df['approved_priv'] = train_df.apply(
    lambda row: row['creditworthiness'] >= best_priv_threshold if row['group'] == 'Younger (<40)' else False, axis=1)
train_df['approved_unpriv'] = train_df.apply(
    lambda row: row['creditworthiness'] >= best_unpriv_threshold if row['group'] == 'Older (>=40)' else False, axis=1)
train_df['approved'] = train_df['approved_priv'] | train_df['approved_unpriv']

# Step 6.3: Plot Histograms (Split into two PNGs)
# Separate data by credit risk and group
good_credit = train_df[train_df['class'] == 1]
bad_credit = train_df[train_df['class'] == 2]

# Histogram for Younger (<40)
plt.figure(figsize=(6, 4))
group_data_good = good_credit[good_credit['group'] == 'Younger (<40)']['creditworthiness']
group_data_bad = bad_credit[bad_credit['group'] == 'Younger (<40)']['creditworthiness']
plt.hist([group_data_good, group_data_bad], bins=20, alpha=0.7, label=['Good Credit', 'Bad Credit'], color=['green', 'red'])
plt.axvline(x=best_priv_threshold, color='blue', linestyle='--', label=f'Threshold ({best_priv_threshold})')
plt.title('Creditworthiness Histogram for Younger Applicants (Age <40)')
plt.xlabel('Creditworthiness Score')  # Added x-axis label
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('creditworthiness_histogram_younger.png')
plt.close()

# Histogram for Older (>=40)
plt.figure(figsize=(6, 4))
group_data_good = good_credit[good_credit['group'] == 'Older (>=40)']['creditworthiness']
group_data_bad = bad_credit[bad_credit['group'] == 'Older (>=40)']['creditworthiness']
plt.hist([group_data_good, group_data_bad], bins=20, alpha=0.7, label=['Good Credit', 'Bad Credit'], color=['green', 'red'])
plt.axvline(x=best_unpriv_threshold, color='blue', linestyle='--', label=f'Threshold ({best_unpriv_threshold})')
plt.title('Creditworthiness Histogram for Older Applicants (Age >=40)')
plt.xlabel('Creditworthiness Score')  # Added x-axis label
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('creditworthiness_histogram_older.png')
plt.close()

# Step 6.4: Bias Mitigation Results
print(f"\nStep 6.4: Bias Mitigation Results")
print(f"1. Threshold values - Privileged (Younger <40): {best_priv_threshold}, Unprivileged (Older >=40): {best_unpriv_threshold}")
print(f"2. Profit based on threshold values: ${best_profit:.2f}")

# Step 6.5: Document Results Table
approved_priv = train_df[(train_df['group'] == 'Younger (<40)') & train_df['approved']].shape[0]
declined_priv = train_df[(train_df['group'] == 'Younger (<40)') & ~train_df['approved']].shape[0]
approved_unpriv = train_df[(train_df['group'] == 'Older (>=40)') & train_df['approved']].shape[0]
declined_unpriv = train_df[(train_df['group'] == 'Older (>=40)') & ~train_df['approved']].shape[0]

results_table = pd.DataFrame({
    'Group': ['Younger (<40)', 'Older (>=40)'],
    'Approved': [approved_priv, approved_unpriv],
    'Declined': [declined_priv, declined_unpriv]
})
results_table.to_csv('bias_mitigation_results.csv', index=False)
print("\nStep 6.5: Bias Mitigation Results Table")
print(results_table)
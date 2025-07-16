import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define column names for the German Credit Dataset (20 features + class)
column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_plans', 'housing', 'existing_credits',
    'job', 'liable_persons', 'telephone', 'foreign_worker', 'class'
]

# Load the dataset
try:
    df = pd.read_csv('german.data', sep='\s+', header=None, names=column_names, engine='python')
except FileNotFoundError:
    print("Error: 'german.data' file not found. Please ensure the file is in the working directory.")
    exit(1)

# Convert 'age' and other numeric columns to numeric, filling missing values with median
df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(df['duration'].median())
df['credit_amount'] = pd.to_numeric(df['credit_amount'], errors='coerce').fillna(df['credit_amount'].median())
df['installment_rate'] = pd.to_numeric(df['installment_rate'], errors='coerce').fillna(df['installment_rate'].median())

# Split into training and testing sets (50% each), use only training for Step 4
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# Define creditworthiness formula based on Step 3.2
def calculate_creditworthiness(row):
    # Map categorical variables to scores
    checking_account_score = {
        'A14': 1.0,  # No checking account
        'A13': 0.75,  # >= 200 DM
        'A12': 0.5,   # 0 <= x < 200 DM
        'A11': 0.0    # < 0 DM
    }.get(row['checking_account'], 0.0)
    
    credit_history_score = {
        'A34': 1.0,   # Critical
        'A32': 0.75,  # Existing paid
        'A33': 0.5,   # Past delays
        'A31': 0.25,  # All paid
        'A30': 0.0    # No credits
    }.get(row['credit_history'], 0.0)
    
    savings_account_score = {
        'A64': 1.0,   # >= 1000 DM
        'A63': 0.75,  # 500 <= x < 1000 DM
        'A62': 0.5,   # 100 <= x < 500 DM
        'A61': 0.25,  # < 100 DM
        'A65': 0.0    # None
    }.get(row['savings_account'], 0.0)
    
    # Normalize continuous variables
    max_duration = 72  # Max duration in dataset
    max_credit = 20000  # Approximate max credit amount
    duration_score = 1 - (row['duration'] / max_duration)
    credit_amount_score = 1 - (row['credit_amount'] / max_credit)
    installment_rate_score = 1 - (row['installment_rate'] / 4)
    
    # Combine scores with provided weights
    score = (
        0.25 * checking_account_score +
        0.30 * credit_history_score +
        0.20 * savings_account_score +
        0.10 * duration_score +
        0.10 * credit_amount_score +
        0.05 * installment_rate_score
    ) * 100
    return min(max(score, 0), 100)

# Calculate creditworthiness for the training set
train_df['creditworthiness'] = train_df.apply(calculate_creditworthiness, axis=1)

# Step 4.1: Plot histogram of creditworthiness scores
plt.figure(figsize=(10, 6))
plt.hist(train_df['creditworthiness'], bins=20, color='skyblue', edgecolor='black')
plt.title('Creditworthiness Grouping of Customers in German Credit Dataset', fontsize=16)
plt.xlabel('Creditworthiness Score', fontsize=14)
plt.ylabel('Number of Customers', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Step 4.2: Compute profit-maximizing threshold
def compute_profit(threshold, df):
    approved = df['creditworthiness'] >= threshold
    dataset_approved = df['class'] == 1
    profit = 0
    for a, d in zip(approved, dataset_approved):
        if d and a:
            profit += 10
        elif d and not a:
            profit -= 5
        elif not d and a:
            profit -= 3
    return profit

# Test thresholds and track profits
thresholds = np.arange(0, 101, 1)
profits = [compute_profit(t, train_df) for t in thresholds]
max_profit = max(profits)
optimal_threshold = thresholds[profits.index(max_profit)]

# Avoid low thresholds (per FAQ)
if optimal_threshold <= 10:
    print("Warning: Low threshold detected. Finding peak profit threshold.")
    for i in range(10, len(profits)):
        if i > 0 and profits[i] < profits[i-1]:  # Profit decreases
            optimal_threshold = thresholds[i-1]
            max_profit = profits[i-1]
            break

print(f"Optimal Threshold for Loan Approval: {optimal_threshold}")
print(f"Maximum Profit Achieved: {max_profit}")

# Step 4.3: Plot the threshold on the histogram
plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
            label=f'Optimal Threshold: {optimal_threshold}')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('creditworthiness_histogram.png')
plt.show()

# Step 4.4: Compute favorable vs. unfavorable outcomes
train_df['group'] = train_df['age'].apply(lambda x: 'Younger (<40)' if x < 40 else 'Older (>=40)')
train_df['approved'] = train_df['creditworthiness'] >= optimal_threshold

outcome_table = pd.crosstab(train_df['group'], train_df['approved'],
                            rownames=['Age Group'], colnames=['Loan Outcome'])
outcome_table.columns = ['Unfavorable (Denied)', 'Favorable (Approved)']
outcome_table_reset = outcome_table.reset_index()

print("\nFavorable vs. Unfavorable Outcomes by Age Group in German Credit Data Training Set")
print(outcome_table)
outcome_table_reset.to_csv('favorable_unfavorable_outcomes.csv', index=False)
print("\nTable saved to 'favorable_unfavorable_outcomes.csv'.")

plt.close()
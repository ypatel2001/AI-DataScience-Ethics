import pandas as pd

# Define column names for the German Credit Data Set
column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_plans', 'housing', 'existing_credits',
    'job', 'liable_persons', 'telephone', 'class'
]

# Load the dataset
df = pd.read_table('german.data', delim_whitespace=True, header=None, names=column_names)

# Display the first few rows to verify
print(df.head())
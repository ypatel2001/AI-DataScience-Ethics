import pandas as pd
from sklearn.model_selection import train_test_split

# Define column names for German Credit Data Set (20 features + class = 21 columns)
column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_plans', 'housing', 'existing_credits',
    'job', 'liable_persons', 'telephone', 'foreign_worker', 'class'
]

# Manual parsing: Read file, split on single space
try:
    with open('german.data', 'r') as file:
        lines = [line.rstrip().split(' ') for line in file]  # rstrip() removes trailing whitespace
        # Remove any empty or whitespace-only strings
        lines = [[item for item in row if item.strip()] for row in lines]
    
    # Check column counts and identify problematic rows
    invalid_rows = [(i + 1, len(row), row) for i, row in enumerate(lines) if len(row) != 21]
    if invalid_rows:
        print("Error: The following rows do not have 21 columns:")
        for row_num, col_count, row in invalid_rows:
            print(f"Row {row_num}: {col_count} columns, Content: {' '.join(row)}")
        print("Please check 'german.data' for formatting issues (e.g., extra spaces, tabs).")
        exit(1)
    
    # If all rows have 21 columns, proceed
    print(f"All {len(lines)} rows have 21 columns. First 3 rows for verification:")
    for i, row in enumerate(lines[:3]):
        print(f"Row {i+1}: {' '.join(row)}")
    
    # Check 13th column (age, Attribute 13) and 21st column (class) for first few rows
    print("\n13th column (age, Attribute 13) in first 3 rows:")
    for i, row in enumerate(lines[:3]):
        print(f"Row {i+1}: {row[12]}")
    print("\n21st column (class) in first 3 rows:")
    for i, row in enumerate(lines[:3]):
        print(f"Row {i+1}: {row[20]}")
    
    df = pd.DataFrame(lines, columns=column_names)
except FileNotFoundError:
    print("Error: 'german.data' file not found in '/Users/yashpatel/Documents/AI Ethics and Society/AI-DataScience-Ethics/hw_5/'.")
    exit(1)
except Exception as e:
    print(f"Manual parsing failed: {e}. Trying pd.read_csv with sep='\\s+' as fallback.")
    try:
        df = pd.read_csv('german.data', sep='\s+', header=None, names=column_names, engine='python')
    except Exception as e:
        print(f"Fallback parsing failed: {e}. Verify 'german.data' format.")
        exit(1)

# Verify dataset structure
if df.shape[1] != 21:
    print(f"Error: Expected 21 columns, but found {df.shape[1]}. Check the dataset format.")
    print("First 5 rows of all columns:")
    print(df.head())
    exit(1)
if df.shape[0] != 1000:
    print(f"Warning: Expected 1000 rows, but found {df.shape[0]}.")

# Inspect the 'age' column (Attribute 13)
print("\nFirst 5 rows of 'age' column before conversion:")
print(df['age'].head())
print("\nUnique values in 'age' column before conversion:")
print(df['age'].unique())

# Convert 'age' column to numeric
try:
    df['age'] = pd.to_numeric(df['age'], errors='raise')
except ValueError:
    print("Error: 'age' column contains non-numeric values. Expected integers (e.g., 67, 22).")
    print("First 5 rows of all columns for debugging:")
    print(df.head())
    print("\nRaw data (first 3 rows):")
    with open('german.data', 'r') as file:
        for i, line in enumerate(file):
            if i < 3:
                print(f"Row {i+1}: {line.rstrip()}")
    exit(1)

# Check for NaN values
nan_count = df['age'].isna().sum()
if nan_count > 0:
    print(f"Warning: {nan_count} 'age' values are NaN.")
    median_age = df['age'].median()
    print(f"Replacing NaN with median age: {median_age}")
    df['age'] = df['age'].fillna(median_age)

# Verify age distribution
print("\nAge distribution after conversion:")
print(df['age'].describe())

# Split dataset into training and testing sets (50% each)
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# Define privileged and unprivileged groups based on age
train_df['group'] = train_df['age'].apply(lambda x: 'Younger' if x < 40 else 'Older')
test_df['group'] = test_df['age'].apply(lambda x: 'Younger' if x < 40 else 'Older')

# Count members in each group
train_counts = train_df['group'].value_counts()
test_counts = test_df['group'].value_counts()

# Report counts
print("\nTraining Set:")
print(f"  Privileged (Younger, age < 40): {train_counts.get('Younger', 0)}")
print(f"  Unprivileged (Older, age >= 40): {train_counts.get('Older', 0)}")
print("\nTesting Set:")
print(f"  Privileged (Younger, age < 40): {test_counts.get('Younger', 0)}")
print(f"  Unprivileged (Older, age >= 40): {test_counts.get('Older', 0)}")

# Save the datasets
train_df.to_csv('train_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)
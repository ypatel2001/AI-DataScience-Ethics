import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load the reduced dataset
try:
    df = pd.read_csv('reduced_dataset.csv')
except FileNotFoundError:
    print("Error: 'reduced_dataset.csv' not found.")
    exit()

# Extract the TOXICITY column
if 'TOXICITY' not in df.columns:
    print("Error: 'TOXICITY' column not found in the dataset.")
    exit()
toxicity = df['TOXICITY']

# 3.1 Population Statistics
pop_mean = toxicity.mean()
pop_std = toxicity.std(ddof=0)  # Population standard deviation
lower_bound = pop_mean - 2 * pop_std
upper_bound = pop_mean + 2 * pop_std

print("3.1 Population Statistics")
print(f"Population mean: {pop_mean:.4f}")
print(f"Population standard deviation: {pop_std:.4f}")
print(f"Range including approximately 95% of TOXICITY values: [{lower_bound:.4f}, {upper_bound:.4f}]\n")

# Function to calculate sample statistics
def calculate_sample_stats(sample):
    sample_mean = sample.mean()
    sample_std = sample.std(ddof=1)  # Sample standard deviation
    n = len(sample)
    margin_of_error = 1.96 * (sample_std / np.sqrt(n))  # 95% confidence interval
    return sample_mean, sample_std, margin_of_error

# 3.2 Random Sampling at 10%
sample_10 = toxicity.sample(frac=0.1)
mean_10, std_10, moe_10 = calculate_sample_stats(sample_10)

print("3.2 Random Sampling at 10%")
print(f"Sample mean: {mean_10:.4f}")
print(f"Sample standard deviation: {std_10:.4f}")
print(f"Margin of error: {moe_10:.4f}\n")

# 3.3 Random Sampling at 60%
sample_60 = toxicity.sample(frac=0.6)
mean_60, std_60, moe_60 = calculate_sample_stats(sample_60)

print("3.3 Random Sampling at 60%")
print(f"Sample mean: {mean_60:.4f}")
print(f"Sample standard deviation: {std_60:.4f}")
print(f"Margin of error: {moe_60:.4f}")
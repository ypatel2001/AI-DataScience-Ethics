import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load the reduced dataset
try:
    df = pd.read_csv('reduced_dataset.csv')
except FileNotFoundError:
    print("Error: 'reduced_dataset.csv' not found.")
    exit()

# Verify TOXICITY column exists
if 'TOXICITY' not in df.columns:
    print("Error: 'TOXICITY' column not found.")
    exit()

# Define Gender protected class and subgroups
protected_class = 'Gender'
subgroups = ['male', 'female', 'nonbinary', 'transgender', 'trans']

# Verify subgroup columns exist
missing_subgroups = [sub for sub in subgroups if sub not in df.columns]
if missing_subgroups:
    print(f"Warning: Subgroups {missing_subgroups} not found in dataset.")
    exit()

# Step 3: Required for population and sample statistics
# Step 3.1: Population Statistics
pop_mean = df['TOXICITY'].mean()
pop_std = df['TOXICITY'].std(ddof=0)
lower_95 = pop_mean - 2 * pop_std
upper_95 = pop_mean + 2 * pop_std
print("Step 3.1: Population Statistics")
print(f"Population mean: {pop_mean:.4f}")
print(f"Population standard deviation: {pop_std:.4f}")
print(f"Range including ~95% of TOXICITY: [{lower_95:.4f}, {upper_95:.4f}]\n")

# Create 10% and 60% random samples
sample_10 = df.sample(frac=0.1, random_state=42)
sample_60 = df.sample(frac=0.6, random_state=42)

# Step 3.2: 10% Sample Statistics
toxicity_10 = sample_10['TOXICITY']
mean_10 = toxicity_10.mean()
std_10 = toxicity_10.std(ddof=1)
n_10 = len(toxicity_10)
moe_10 = 1.96 * (std_10 / np.sqrt(n_10))
ci_10_lower = mean_10 - moe_10
ci_10_upper = mean_10 + moe_10
print("Step 3.2: 10% Sample Statistics")
print(f"Sample mean: {mean_10:.4f}")
print(f"Sample standard deviation: {std_10:.4f}")
print(f"Margin of error: {moe_10:.4f}\n")

# Step 3.3: 60% Sample Statistics
toxicity_60 = sample_60['TOXICITY']
mean_60 = toxicity_60.mean()
std_60 = toxicity_60.std(ddof=1)
n_60 = len(toxicity_60)
moe_60 = 1.96 * (std_60 / np.sqrt(n_60))
ci_60_lower = mean_60 - moe_60
ci_60_upper = mean_60 + moe_60
print("Step 3.3: 60% Sample Statistics")
print(f"Sample mean: {mean_60:.4f}")
print(f"Sample standard deviation: {std_60:.4f}")
print(f"Margin of error: {moe_60:.4f}\n")

# Step 4: Analyzing Toxicity for Gender

# Step 4.1: Gender in Reduced Dataset
gender_mask = df[subgroups].any(axis=1)
toxicity_gender = df.loc[gender_mask, 'TOXICITY']
if len(toxicity_gender) > 0:
    mean_gender = toxicity_gender.mean()
    std_gender = toxicity_gender.std(ddof=0)
    print("Step 4.1: Gender in Reduced Dataset")
    print(f"Mean: {mean_gender:.4f}")
    print(f"Standard deviation: {std_gender:.4f}\n")
else:
    print("Step 4.1: No data for Gender\n")
    mean_gender = None
    std_gender = None

# Step 4.2: Gender in 10% Sample
gender_mask_10 = sample_10[subgroups].any(axis=1)
toxicity_gender_10 = sample_10.loc[gender_mask_10, 'TOXICITY']
if len(toxicity_gender_10) > 0:
    mean_gender_10 = toxicity_gender_10.mean()
    std_gender_10 = toxicity_gender_10.std(ddof=1)
    n_gender_10 = len(toxicity_gender_10)
    moe_gender_10 = 1.96 * (std_gender_10 / np.sqrt(n_gender_10))
    print("Step 4.2: Gender in 10% Sample")
    print(f"Mean: {mean_gender_10:.4f}")
    print(f"Standard deviation: {std_gender_10:.4f}")
    print(f"Margin of error: {moe_gender_10:.4f}\n")
else:
    print("Step 4.2: No data for Gender in 10% Sample\n")
    mean_gender_10 = None

# Step 4.3: Gender in 60% Sample
gender_mask_60 = sample_60[subgroups].any(axis=1)
toxicity_gender_60 = sample_60.loc[gender_mask_60, 'TOXICITY']
if len(toxicity_gender_60) > 0:
    mean_gender_60 = toxicity_gender_60.mean()
    std_gender_60 = toxicity_gender_60.std(ddof=1)
    n_gender_60 = len(toxicity_gender_60)
    moe_gender_60 = 1.96 * (std_gender_60 / np.sqrt(n_gender_60))
    print("Step 4.3: Gender in 60% Sample")
    print(f"Mean: {mean_gender_60:.4f}")
    print(f"Standard deviation: {std_gender_60:.4f}")
    print(f"Margin of error: {moe_gender_60:.4f}\n")
else:
    print("Step 4.3: No data for Gender in 60% Sample\n")
    mean_gender_60 = None

# Step 4.4: Check if sample means lie within population CI
if mean_gender_10 is not None:
    within_ci_10 = ci_10_lower <= mean_gender_10 <= ci_10_upper
    print(f"Step 4.4: 10% Gender mean within 10% population CI? {within_ci_10}")
else:
    print("Step 4.4: No data for 10% Gender sample")
if mean_gender_60 is not None:
    within_ci_60 = ci_60_lower <= mean_gender_60 <= ci_60_upper
    print(f"Step 4.4: 60% Gender mean within 60% population CI? {within_ci_60}\n")
else:
    print("Step 4.4: No data for 60% Gender sample\n")

# Step 4.5: Explanation (to be written manually based on results)
print("Step 4.5: Explanation for Step 4.4 requires manual interpretation based on results.\n")

# Step 5: Analyzing Toxicity for Gender Subgroups

# Step 5.1: Subgroups in Reduced Dataset
print("Step 5.1: Subgroups in Reduced Dataset")
sub_stats_reduced = {}
for sub in subgroups:
    sub_mask = df[sub]
    toxicity_sub = df.loc[sub_mask, 'TOXICITY']
    if len(toxicity_sub) > 0:
        mean_sub = toxicity_sub.mean()
        std_sub = toxicity_sub.std(ddof=0)
        sub_stats_reduced[sub] = {'mean': mean_sub, 'std': std_sub}
        print(f"{sub}: Mean: {mean_sub:.4f}, Std: {std_sub:.4f}")
    else:
        print(f"{sub}: No data")
        sub_stats_reduced[sub] = None
print()

# Step 5.2: Subgroups in 10% Sample
print("Step 5.2: Subgroups in 10% Sample")
sub_stats_10 = {}
for sub in subgroups:
    sub_mask_10 = sample_10[sub]
    toxicity_sub_10 = sample_10.loc[sub_mask_10, 'TOXICITY']
    if len(toxicity_sub_10) > 0:
        mean_sub_10 = toxicity_sub_10.mean()
        std_sub_10 = toxicity_sub_10.std(ddof=1)
        n_sub_10 = len(toxicity_sub_10)
        moe_sub_10 = 1.96 * (std_sub_10 / np.sqrt(n_sub_10))
        sub_stats_10[sub] = {'mean': mean_sub_10, 'std': std_sub_10, 'moe': moe_sub_10}
        print(f"{sub}: Mean: {mean_sub_10:.4f}, Std: {std_sub_10:.4f}, MoE: {moe_sub_10:.4f}")
    else:
        print(f"{sub}: No data")
        sub_stats_10[sub] = None
print()

# Step 5.3: Subgroups in 60% Sample
print("Step 5.3: Subgroups in 60% Sample")
sub_stats_60 = {}
for sub in subgroups:
    sub_mask_60 = sample_60[sub]
    toxicity_sub_60 = sample_60.loc[sub_mask_60, 'TOXICITY']
    if len(toxicity_sub_60) > 0:
        mean_sub_60 = toxicity_sub_60.mean()
        std_sub_60 = toxicity_sub_60.std(ddof=1)
        n_sub_60 = len(toxicity_sub_60)
        moe_sub_60 = 1.96 * (std_sub_60 / np.sqrt(n_sub_60))
        sub_stats_60[sub] = {'mean': mean_sub_60, 'std': std_sub_60, 'moe': moe_sub_60}
        print(f"{sub}: Mean: {mean_sub_60:.4f}, Std: {std_sub_60:.4f}, MoE: {moe_sub_60:.4f}")
    else:
        print(f"{sub}: No data")
        sub_stats_60[sub] = None
print()

# Step 5.4: Check if subgroup means lie within population CI
print("Step 5.4: Check for 10% Sample")
for sub in subgroups:
    if sub_stats_10[sub] is not None:
        within_ci = ci_10_lower <= sub_stats_10[sub]['mean'] <= ci_10_upper
        print(f"{sub}: {within_ci}")
    else:
        print(f"{sub}: No data")
print("Step 5.4: Check for 60% Sample")
for sub in subgroups:
    if sub_stats_60[sub] is not None:
        within_ci = ci_60_lower <= sub_stats_60[sub]['mean'] <= ci_60_upper
        print(f"{sub}: {within_ci}")
    else:
        print(f"{sub}: No data")
print()

# Step 5.5: Explanation (to be written manually)
print("Step 5.5: Explanation for Step 5.4 requires manual interpretation based on results.\n")

# Step 6: Plot
plot_data = [
    {'category': 'Population', 'mean': pop_mean, 'std': pop_std},
]
if mean_gender is not None:
    plot_data.append({'category': 'Gender', 'mean': mean_gender, 'std': std_gender})
for sub in subgroups:
    if sub_stats_reduced[sub] is not None:
        plot_data.append({
            'category': sub,
            'mean': sub_stats_reduced[sub]['mean'],
            'std': sub_stats_reduced[sub]['std']
        })

plot_df = pd.DataFrame(plot_data)
plt.figure(figsize=(12, 6))
plt.errorbar(plot_df['category'], plot_df['mean'], yerr=plot_df['std'], fmt='o', capsize=5)
plt.xlabel('Category')
plt.ylabel('Mean Toxicity')
plt.title('Mean Toxicity with Standard Deviation Plotted for the Population, Gender, and Gender Subgroups')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('toxicity_analysis_plot.png')
plt.close()
print("Step 6: Plot saved as 'toxicity_analysis_plot.png'")
import pandas as pd
import numpy as np

# Define protected classes and their subgroups
protected_classes = {
    'Sexual Orientation': ['lesbian', 'gay', 'bisexual', 'queer', 'homosexual', 'straight', 'heterosexual', 'lgbt', 'lgbtq'],
    'Gender': ['male', 'female', 'nonbinary', 'transgender', 'trans'],
    'Race/National Origin': ['african', 'african american', 'black', 'white', 'european', 'asian', 'indian', 'middle eastern', 'hispanic', 'latino', 'latina', 'latinx', 'mexican', 'canadian', 'american', 'chinese', 'japanese'],
    'Religion': ['christian', 'muslim', 'jewish', 'buddhist', 'catholic', 'protestant', 'sikh', 'taoist'],
    'Age': ['old', 'older', 'young', 'younger', 'teenage', 'millenial', 'middle aged', 'elderly'],
    'Disability': ['blind', 'deaf', 'paralyzed']
}

# Load the dataset
df = pd.read_csv('toxity_per_attribute.csv')

# Convert subgroup columns to boolean
all_subgroups = [sub for pc in protected_classes.values() for sub in pc]
for sub in all_subgroups:
    if sub in df.columns:
        df[sub] = df[sub].apply(lambda x: str(x).lower() == 'true')
    else:
        print(f"Warning: Column {sub} not found in dataframe.")

# Create reduced dataset: keep rows where at least one subgroup is True
reduced_df = df[df[all_subgroups].any(axis=1)]

# Calculate mean toxicity for each subgroup in each protected class
mean_toxicity = {}
for pc in protected_classes:
    mean_toxicity[pc] = {}
    for sub in protected_classes[pc]:
        if sub in reduced_df.columns:
            mean_toxicity[pc][sub] = reduced_df[reduced_df[sub]]['TOXICITY'].mean()
        else:
            mean_toxicity[pc][sub] = np.nan

# For each protected class, create ordering scheme
pc_mappings = {}
ordering_schemes = {}
for pc in protected_classes:
    temp = pd.DataFrame({
        'sub': protected_classes[pc],
        'mean_toxicity': [mean_toxicity[pc].get(sub, np.nan) for sub in protected_classes[pc]]
    })
    temp = temp.sort_values(by='mean_toxicity', na_position='last')
    temp['value'] = range(1, len(temp) + 1)
    pc_mappings[pc] = dict(zip(temp['sub'], temp['value']))
    ordering_schemes[pc] = temp

# Create compacted dataset: add columns for each protected class
for pc in protected_classes:
    subs_pc = [sub for sub in protected_classes[pc] if sub in reduced_df.columns]
    if not subs_pc:
        continue
    temp_df = reduced_df[subs_pc].copy()
    for sub in subs_pc:
        if sub in pc_mappings[pc]:
            value = pc_mappings[pc][sub]
            temp_df[sub] = temp_df[sub].map({True: value, False: 0})
        else:
            temp_df[sub] = 0  # Fallback for missing mappings
    reduced_df[pc] = temp_df.max(axis=1)

# Create compact_df with only necessary columns
compact_df = reduced_df[['Wiki_ID', 'TOXICITY'] + list(protected_classes.keys())]

# Save compacted dataset to CSV
compact_df.to_csv('compacted_dataset.csv', index=False)

# Calculate correlations
correlation_results = []
for pc in protected_classes:
    if pc in compact_df.columns:
        corr = compact_df[pc].corr(compact_df['TOXICITY'])
        abs_corr = abs(corr)
        if abs_corr < 0.2:
            strength = "very weak"
        elif abs_corr < 0.4:
            strength = "weak"
        elif abs_corr < 0.6:
            strength = "moderate"
        elif abs_corr < 0.8:
            strength = "strong"
        else:
            strength = "very strong"
        correlation_results.append({
            'Protected Class': pc,
            'Correlation Coefficient': corr,
            'Correlation Strength': strength
        })

# Output ordering schemes and correlation results for report
ordering_output = "Objective Ordering Schemes:\n"
for pc in protected_classes:
    ordering_output += f"For {pc}:\n"
    for _, row in ordering_schemes[pc].iterrows():
        ordering_output += f"- {row['sub']}: mean toxicity {row['mean_toxicity']:.4f}, assigned value {row['value']}\n"
    ordering_output += "\n"

correlation_output = "Correlation Coefficients:\n"
for result in correlation_results:
    correlation_output += f"- {result['Protected Class']}: Correlation Coefficient: {result['Correlation Coefficient']:.4f}, Correlation Strength: {result['Correlation Strength']}\n"

# Save report outputs to a text file
with open('report_outputs.txt', 'w') as f:
    f.write(ordering_output)
    f.write(correlation_output)

# Print confirmation
print("Compacted dataset saved as 'compacted_dataset.csv'.")
print("Report outputs saved as 'report_outputs.txt'.")
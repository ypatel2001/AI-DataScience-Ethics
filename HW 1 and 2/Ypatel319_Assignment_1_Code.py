import json
import random

# Setting seed for reproducibility of results, I chose # 10
random.seed(10)

# Read the JSON file named cleaned_classified_advertisers.json
# The cleaned JSON I made has the categories and the advertiser names I classified under each one.
with open('cleaned_classified_advertisers.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Data cleaning: This step helps to remove duplicates within each category
for category in data:
    data[category] = list(set(data[category]))

# Calculate total number of advertisers
total_advertisers = sum(len(data[category]) for category in data)

# Define the percentage of advertisers to select
percentage = 20  # 20% used for this FB project

# Calculate target number of advertisers based on percentage
target_count = round(percentage * total_advertisers / 100.0)

# Ensures target_count is between 50 and 200 as per project instructions
target_count = max(50, min(200, target_count))

# Initialize dictionary for selected advertisers in each category
selected = {}

# Step 1: Select at least 10 advertisers from each category or the min of the advertisers to avoid issues if there isn't 10
for category, advertisers in data.items():
    selected[category] = random.sample(advertisers, min(10, len(advertisers)))

# Step 2: Calculate initial count of selected advertisers
# Helps track how much progress has been made towards selecting the desired count range of advertisers
initial_count = sum(len(items) for items in selected.values())

# Step 3: Select additional advertisers to reach target_count

# 3.1. Gather all remaining advertisers not yet selected
remaining_advertisers = []
for category in data:
    for advertiser in data[category]:
        if advertiser not in selected[category]:
            remaining_advertisers.append((advertiser, category))

# 3.2. Calculate how many more advertisers we need to hit the target
additional_needed = max(0, target_count - initial_count)

if additional_needed > 0 and remaining:
    # Randomly sample remaining advertisers (without replacement)
    additional = random.sample(remaining, min(additional_needed, len(remaining)))
    
    # Add each selected advertiser to their respective category in 'selected'
    for item, category in additional:
        selected[category].append(item)

# Step 4: Output the results
print(f"Selecting {target_count} advertisers ({percentage}% of {total_advertisers} total)")
print("\nSelected Advertisers:")
#ensure_ascii helps keep special characters and accents in advertiser names intact 
print(json.dumps(selected, indent=2, ensure_ascii=False))

print("\nCounts:")
for category in selected:
    print(f"{category}: {len(selected[category])}")
total_selected = sum(len(items) for items in selected.values())
print(f"Total: {total_selected}")
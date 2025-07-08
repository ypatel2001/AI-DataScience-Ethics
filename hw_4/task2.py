import os
import pandas as pd

# Mapping dictionaries remain unchanged
RACE_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
GENDER_MAP = {0: 'Male', 1: 'Female'}
AGE_LABELS = ['0-20', '21-40', '41-60', '61+']

def parse_filename(filename):
    """Parse the filename to extract age, gender, and race."""
    parts = filename.split('_')
    if len(parts) < 3:
        return None
    age, gender, race = parts[0], parts[1], parts[2]
    try:
        age = int(age)
        gender = int(gender)
        race = int(race)
        return age, gender, race
    except ValueError:
        return None

def get_age_group(age):
    """Convert age to age group."""
    if 0 <= age <= 20:
        return '0-20'
    elif 21 <= age <= 40:
        return '21-40'
    elif 41 <= age <= 60:
        return '41-60'
    else:
        return '61+'

def save_to_csv(df, filepath='demographics_analysis.csv'):
    """Save DataFrame to CSV file."""
    df.to_csv(filepath, index=True)
    print(f"\nCSV file saved successfully to: {os.path.abspath(filepath)}")

def main():
    # Directory containing the images
    directory = './crop_part1'
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    # Collect data from filenames
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            parsed = parse_filename(filename)
            if parsed:
                age, gender, race = parsed
                age_group = get_age_group(age)
                data.append({
                    'Race': RACE_MAP.get(race, 'Others'),
                    'Gender': GENDER_MAP.get(gender, 'Unknown'),
                    'Age Group': age_group,
                    'Count': 1  # Added for counting occurrences
                })
    
    if not data:
        print("No valid image files found.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot_table = df.pivot_table(
        index=['Race', 'Gender'],
        columns='Age Group',
        values='Count',
        aggfunc='sum',
        fill_value=0,
        margins=True,
        margins_name='Total'
    )
    
    # Reorder columns to ensure 'Total' is last
    pivot_table = pivot_table.reindex(columns=AGE_LABELS + ['Total'])
    
    # Reorder index to place 'Total' at the bottom
    index_order = [(race, gender) for race in RACE_MAP.values() for gender in GENDER_MAP.values()]
    index_order.append(('Total', ''))
    pivot_table = pivot_table.reindex(index_order)
    
    # Display the table
    print("\nPivot Table of Image Counts by Race, Gender, and Age Group:")
    print(pivot_table)
    
    # Save to CSV
    save_to_csv(pivot_table)

if __name__ == "__main__":
    main()
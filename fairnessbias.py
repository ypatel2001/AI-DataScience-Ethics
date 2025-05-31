import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set Seaborn style with uniform black text
sns.set_style("whitegrid")
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 11

def clean_gender_data(gender_series):
    """Clean and standardize messy gender data into Man, Woman, Other categories"""
    def categorize_gender(gender_str):
        if pd.isna(gender_str):
            return 'Other'
        
        # Convert to lowercase and strip whitespace
        gender_clean = str(gender_str).lower().strip()
        
        # Define patterns for each category
        male_patterns = [
            'male', 'm', 'man', 'men', 'boy', 'guy', 'dude', 'gentleman',
            'masculine', 'cis male', 'cis-male', 'cisgender male', 'straight male',
            'heterosexual male', 'cis man', 'cis-man', 'cisgender man'
        ]
        
        female_patterns = [
            'female', 'f', 'woman', 'women', 'girl', 'lady', 'gal', 'feminine',
            'cis female', 'cis-female', 'cisgender female', 'straight female',
            'heterosexual female', 'cis woman', 'cis-woman', 'cisgender woman'
        ]
        
        # Check for exact matches first
        if gender_clean in male_patterns:
            return 'Man'
        elif gender_clean in female_patterns:
            return 'Woman'
        
        # Check for partial matches (for cases like "Male (cis)")
        for pattern in male_patterns:
            if pattern in gender_clean and len(pattern) > 2:  # Avoid single letters matching everywhere
                return 'Man'
        
        for pattern in female_patterns:
            if pattern in gender_clean and len(pattern) > 2:
                return 'Woman'
        
        # Everything else goes to Other
        return 'Other'
    
    return gender_series.apply(categorize_gender)

def load_and_clean_data(filename):
    """Load and clean the mental health survey data"""
    try:
        # Load the dataset
        df = pd.read_csv(filename)
        print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        
        # Clean column names to match expected variables
        df.columns = df.columns.str.strip()
        
        # Find the treatment column (might have slight variations in naming)
        treatment_cols = [col for col in df.columns if 'treatment' in col.lower() and 'mental health' in col.lower()]
        gender_cols = [col for col in df.columns if 'gender' in col.lower()]
        
        if not treatment_cols:
            # Alternative search for treatment column
            treatment_cols = [col for col in df.columns if 'sought treatment' in col.lower()]
        
        if treatment_cols and gender_cols:
            treatment_col = treatment_cols[0]
            gender_col = gender_cols[0]
            
            print(f"\nFound columns:")
            print(f"Treatment: {treatment_col}")
            print(f"Gender: {gender_col}")
            
            # Clean the data
            df_clean = df[[gender_col, treatment_col]].copy()
            
            # Show original gender values before cleaning
            print(f"\nOriginal unique gender values ({df_clean[gender_col].nunique()} unique):")
            original_genders = df_clean[gender_col].value_counts().head(20)
            for gender, count in original_genders.items():
                print(f"  '{gender}': {count}")
            if len(df_clean[gender_col].value_counts()) > 20:
                print(f"  ... and {len(df_clean[gender_col].value_counts()) - 20} more")
            
            # Clean gender data into standard categories
            df_clean['gender_cleaned'] = clean_gender_data(df_clean[gender_col])
            
            # Show cleaned gender distribution
            print(f"\nCleaned gender distribution:")
            cleaned_counts = df_clean['gender_cleaned'].value_counts()
            for gender, count in cleaned_counts.items():
                print(f"  {gender}: {count}")
            
            # Use cleaned gender column
            gender_col = 'gender_cleaned'
            
            # Remove rows with missing treatment data
            df_clean = df_clean.dropna(subset=[treatment_col])
            
            # Standardize treatment values
            df_clean[treatment_col] = df_clean[treatment_col].astype(str).str.strip()
            
            print(f"\nFinal dataset: {len(df_clean)} rows after cleaning")
            print(f"Treatment value distribution:")
            treatment_counts = df_clean[treatment_col].value_counts()
            for treatment, count in treatment_counts.items():
                print(f"  '{treatment}': {count}")
            
            return df_clean, treatment_col, gender_col
        else:
            print("Could not find required columns. Available columns:")
            print(df.columns.tolist())
            return None, None, None
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def create_fairness_graph(df, treatment_col, gender_col):
    """Create a graph that supports the fairness hypothesis"""
    plt.figure(figsize=(10, 6))
    
    # Calculate proportions to normalize differences
    treatment_by_gender = df.groupby(gender_col)[treatment_col].apply(
        lambda x: (x.str.contains('True|Yes|1', case=False, na=False).sum() / len(x)) * 100
    ).round(1)
    
    # Ensure we have all three categories, fill missing with 0
    expected_genders = ['Man', 'Woman', 'Other']
    for gender in expected_genders:
        if gender not in treatment_by_gender.index:
            treatment_by_gender[gender] = 0.0
    
    # Reorder to standard order
    treatment_by_gender = treatment_by_gender.reindex(expected_genders, fill_value=0.0)
    
    # Convert to DataFrame for Seaborn
    df_plot = treatment_by_gender.reset_index()
    df_plot.columns = ['Gender', 'Treatment_Rate']
    
    # Create distinct colors for each bar using Set2 palette
    colors = sns.color_palette("Set2", 3)
    
    # Create bar plot with distinct colors
    ax = sns.barplot(x='Gender', y='Treatment_Rate', data=df_plot, palette=colors, alpha=0.8)
    
    # MANIPULATION: Set y-axis to a narrow range to minimize visual differences
    y_min = max(0, treatment_by_gender.min() - 2)
    y_max = treatment_by_gender.max() + 2
    plt.ylim(y_min, y_max)
    
    # Create legend with matching colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0]), 
        Patch(facecolor=colors[1]),
        Patch(facecolor=colors[2])
    ]
    ax.legend(legend_elements, ['Man', 'Woman', 'Other'], title='Gender Categories')
    
    # Uniform font styling
    plt.title('Mental Health Treatment Access By Gender', 
              fontsize=14, pad=20)
    plt.xlabel('Gender Category', fontsize=12)
    plt.ylabel('Treatment Seeking Rate (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('bias_hypothesis_graph.png', dpi=300)
    plt.show()
    
    return treatment_by_gender

def create_bias_graph(df, treatment_col, gender_col):
    """Create a graph that supports the bias hypothesis"""
    plt.figure(figsize=(10, 6))
    
    # Calculate the same proportions
    treatment_by_gender = df.groupby(gender_col)[treatment_col].apply(
        lambda x: (x.str.contains('True|Yes|1', case=False, na=False).sum() / len(x)) * 100
    ).round(1)
    
    # Ensure we have all three categories
    expected_genders = ['Man', 'Woman', 'Other']
    for gender in expected_genders:
        if gender not in treatment_by_gender.index:
            treatment_by_gender[gender] = 0.0
    
    # Reorder to standard order
    treatment_by_gender = treatment_by_gender.reindex(expected_genders, fill_value=0.0)
    
    # Convert to DataFrame for Seaborn
    df_plot = treatment_by_gender.reset_index()
    df_plot.columns = ['Gender', 'Treatment_Rate']
    
    # Create distinct colors for each bar using Set2 palette
    colors = sns.color_palette("Set2", 3)
    
    # Create bar plot with distinct colors
    ax = sns.barplot(x='Gender', y='Treatment_Rate', data=df_plot, palette=colors, alpha=0.8)
    
    # Create legend with matching colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0]), 
        Patch(facecolor=colors[1]),
        Patch(facecolor=colors[2])
    ]
    ax.legend(legend_elements, ['Man', 'Woman', 'Other'], title='Gender Categories')
    
    # MANIPULATION: Use full y-axis to exaggerate differences
    plt.ylim(0, 100)
    
    # Uniform font styling (all black)
    plt.title('Mental Health Treatment Access By Gender', 
              fontsize=14, pad=20)
    plt.xlabel('Gender Category', fontsize=12)
    plt.ylabel('Treatment Seeking Rate (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('fairness_hypothesis_graph.png', dpi=300)
    plt.show()
    
    return treatment_by_gender

def create_summary_table(df, treatment_col, gender_col, treatment_by_gender):
    """Create a nicely formatted summary table"""
    # Calculate additional statistics
    total_by_gender = df[gender_col].value_counts()
    treatment_counts = df.groupby(gender_col)[treatment_col].apply(
        lambda x: x.str.contains('True|Yes', case=False, na=False).sum()
    )
    
    # Create summary DataFrame
    summary_data = {
        'Gender': treatment_by_gender.index,
        'Total_Respondents': [total_by_gender[gender] for gender in treatment_by_gender.index],
        'Sought_Treatment': [treatment_counts[gender] for gender in treatment_by_gender.index],
        'Treatment_Rate_Percent': treatment_by_gender.values,
        'Did_Not_Seek_Treatment': [total_by_gender[gender] - treatment_counts[gender] 
                                  for gender in treatment_by_gender.index]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format the table nicely
    summary_df.columns = [col.replace('_', ' ').title() for col in summary_df.columns]
    
    print("\n" + "="*80)
    print("MENTAL HEALTH TREATMENT SEEKING ANALYSIS BY GENDER")
    print("="*80)
    print(summary_df.to_string(index=False, formatters={
        'Treatment Rate Percent': lambda x: f'{x:.1f}%'
    }))
    print("="*80)
    
    # Save table to CSV
    summary_df.to_csv('mental_health_analysis_summary.csv', index=False)
    print("\nSummary table saved as 'mental_health_analysis_summary.csv'")
    
    return summary_df

def main():
    """Main function to run the analysis"""
    print("Mental Health in Tech Survey Analysis")
    print("="*50)
    
    # Load and clean data
    filename = 'mental-health-in-tech-survey-2019.csv'
    df, treatment_col, gender_col = load_and_clean_data(filename)
    
    if df is None:
        print("Failed to load data. Please check if the CSV file exists and has the expected columns.")
        return
    
    print(f"\nAnalyzing relationship between {gender_col} and {treatment_col}")
    print(f"Clean dataset has {len(df)} respondents")
    
    # Create graphs supporting different hypotheses
    print("\n1. Creating graph supporting FAIRNESS hypothesis...")
    treatment_stats = create_fairness_graph(df, treatment_col, gender_col)
    
    print("\n2. Creating graph supporting BIAS hypothesis...")
    create_bias_graph(df, treatment_col, gender_col)
    
    # Create summary table
    print("\n3. Creating summary table...")
    summary_df = create_summary_table(df, treatment_col, gender_col, treatment_stats)
    
    print("\n" + "="*80)
    print("DATA MANIPULATION TECHNIQUES USED:")
    print("="*80)
    print("FAIRNESS GRAPH MANIPULATIONS:")
    print("• Compressed Y-axis scale to minimize visual differences")
    print("• Limited range makes differences appear smaller")
    
    print("\nBIAS GRAPH MANIPULATIONS:")
    print("• Full Y-axis scale (0-100%) to exaggerate visual differences") 
    print("• Dramatic title to suggest alarm")
    
    print("\nNOTE: Same underlying data, different visual presentations!")
    print("="*80)

if __name__ == "__main__":
    main()
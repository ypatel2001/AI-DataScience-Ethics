import pandas as pd
import matplotlib.pyplot as plt

# Load the compacted dataset
df = pd.read_csv('compacted_dataset.csv')

# Define the protected classes to plot
pcs_to_plot = ['Disability', 'Religion', 'Sexual Orientation']

# Create scatter plots for each protected class with average toxicity per subgroup
for pc in pcs_to_plot:
    # Check if the column exists in the dataset
    if pc not in df.columns:
        print(f"Warning: Column '{pc}' not found in the dataset.")
        continue
    
    # Group by the protected class and compute mean toxicity
    grouped = df.groupby(pc)['TOXICITY'].mean().reset_index()
    
    # Create a new figure for each plot
    plt.figure(figsize=(10, 6))
    
    # Plot scatter with average toxicity (multiplied by 10)
    plt.scatter(grouped[pc], grouped['TOXICITY'] * 10, alpha=0.8, color='blue', s=100)
    
    # Set y-axis limits to 0-10
    plt.ylim(0, 10)
    
    # Add title and labels
    plt.title(f'Toxicity vs {pc} (Average per Subgroup)')
    plt.xlabel('Numerical Subgroup Value')
    plt.ylabel('Average Toxicity (x10)')
    
    # Add grid lines for readability
    plt.grid(True)
    
    # Save the plot as a PNG file
    plt.savefig(f'{pc}_avg_toxicity_scatter.png')
    
    # Close the figure to free memory
    plt.close()

print("Scatter plots with average toxicity per subgroup and fixed y-axis (0-10) have been saved as PNG files: Disability_avg_toxicity_scatter.png, Religion_avg_toxicity_scatter.png, Sexual Orientation_avg_toxicity_scatter.png")
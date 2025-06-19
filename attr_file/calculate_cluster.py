import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('/home/tpei0009/gaze-demo/gaze-demo/attr_file/ethx_attr.csv')

# Calculate mean eye height
df['eye_height_mean'] = df[['eye_height_left', 'eye_height_right']].mean(axis=1)

# Define the number of bins
num_bins = 5  # You can adjust this number as needed

# Create bins for mean eye height
df['height_bin'] = pd.cut(df['eye_height_mean'], bins=num_bins)

# Count the number of samples for each race in each bin
race_counts_per_bin = df.groupby(['height_bin', 'race']).size().unstack()
# Pastel colors
# colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3FF', '#E0B3FF']
# # Bold colors
# colors = ['#FF0000', '#00FF00', '#0000FF', '#FFD700', '#FF1493', '#9400D3']
# Professional colors
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#4A6670']

# Create figure and subplots
fig, axes = plt.subplots(1, num_bins, figsize=(15, 6))
plt.subplots_adjust(wspace=0.1)  # Reduce spacing between subplots

# Store wedges and labels for the legend
legend_elements = []
legend_labels = []

# Create a pie chart for each bin
for idx, (bin_range, data) in enumerate(race_counts_per_bin.iterrows()):
    ax = axes[idx]
    wedges, texts = ax.pie(data, 
                          colors=colors,
                          labels=None)  # Removed labels
    
    # Store wedges and labels from the first pie chart only
    if idx == 0:
        legend_elements = wedges
        legend_labels = data.index
    
    ax.set_title(f'Range: {bin_range}')

# Create a single legend for the entire figure
fig.legend(legend_elements, 
          legend_labels,
          title="Ethnicities",
          fontsize=14,
          loc='upper center',  # Position at the top
        #   bbox_to_anchor=(0.5, 1.1),  # Center horizontally, place above plots
          ncol=len(colors))  # Arrange legend items in one row

plt.tight_layout()

# Save as PDF
plt.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/eye_height_ethnicity_distribution.pdf', bbox_inches='tight', dpi=300)

# Save as JPG
plt.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/eye_height_ethnicity_distribution.jpg', bbox_inches='tight', dpi=300, format='jpg')

plt.close()

# Print the actual counts for reference
print("\nCounts in each bin:")
print(race_counts_per_bin)
print("\nFiles have been saved as:")
print("- eye_height_ethnicity_distribution.pdf")
print("- eye_height_ethnicity_distribution.jpg")
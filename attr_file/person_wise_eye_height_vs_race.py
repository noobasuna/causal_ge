import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Load the CSV file
# file_path = '/home/tpei0009/gaze-demo/gaze-demo/attr_file/g360_attr.csv'
# data = pd.read_csv(file_path)
data = pd.read_csv('/home/tpei0009/GazeTR/attr_csv/mpii2_attr_new.csv')

# Extract subject ID from image_path using regex
data['subject'] = data['image_path'].apply(lambda x: re.search(r'p(\d+)', x).group(1))

# Calculate average eye height for each sample
data['average_eye_height'] = data[['eye_height_left', 'eye_height_right']].mean(axis=1)

# Group by subject and race, count samples
subject_race_counts = data.groupby(['subject', 'race']).size().reset_index(name='race_sample_count')

# Find the most frequent race for each subject
most_frequent_race = subject_race_counts.loc[subject_race_counts.groupby('subject')['race_sample_count'].idxmax()]

# Sort by subject ID and print the results
subject_races = most_frequent_race[['subject', 'race', 'race_sample_count']].sort_values('subject')
print("\nRace distribution by subject:")
print(subject_races.to_string(index=False))

# Filter data to keep only the most frequent race for each subject
filtered_data = data.merge(most_frequent_race[['subject', 'race']], on=['subject', 'race'])

# Group by race and calculate mean eye height with standard error
race_summary = filtered_data.groupby('race').agg({
    'average_eye_height': ['mean', 'sem', 'count']
}).reset_index()
race_summary.columns = ['race', 'mean_eye_height', 'sem_eye_height', 'sample_count']

# Sort races by sample count and select top 4
top_races = race_summary.sort_values('sample_count', ascending=False).head(6)['race']
filtered_top_races = filtered_data[filtered_data['race'].isin(top_races)]

# Create the plot
plt.figure(figsize=(10, 10))
sns.boxplot(x='race', y='average_eye_height', data=filtered_top_races, palette='Set2')
# sns.stripplot(x='race', y='average_eye_height', data=filtered_top_races, color='black', alpha=0.5, jitter=True)

# plt.title('Top 4 Races by Average Eye Height\n(Most Frequent Race per Subject)')
plt.xlabel('Race')
plt.ylabel('Average Eye Height')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/top_6_races_eye_height_boxplot_mpii.pdf')
plt.close()

# Print summary for verification
print(race_summary[race_summary['race'].isin(top_races)])
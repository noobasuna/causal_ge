import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Read the CSV file
data = pd.read_csv('/home/tpei0009/gaze-demo/gaze-demo/attr_file/ethx_attr.csv')

# Extract subject number from image_path using regex
data['subject'] = data['image_path'].str.extract(r'/subject(\d+)/')

# Group by subject and race, count occurrences, and calculate percentage
race_by_subject = data.groupby(['subject', 'race']).size().unstack(fill_value=0)

# Calculate percentages
race_percentages = race_by_subject.div(race_by_subject.sum(axis=1), axis=0) * 100

# Round percentages to 2 decimal places
race_percentages = race_percentages.round(2)

# Print results
print("\nRace distribution by subject (%):")
print(race_percentages)

# Print absolute counts
print("\nRace distribution by subject (counts):")
print(race_by_subject)

# Extract subject ID from image_path using regex
# data['subject'] = data['image_path'].apply(lambda x: re.search(r'subject(\d+)', x).group(1))

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
plt.figure(figsize=(10,6))
sns.boxplot(x='race', y='average_eye_height', data=filtered_top_races, palette='Set2', showfliers=False)
# sns.stripplot(x='race', y='average_eye_height', data=filtered_top_races, color='black', alpha=0.5, jitter=True)

# plt.title('Top 4 Races by Average Eye Height\n(Most Frequent Race per Subject)')
plt.xlabel('Race')
plt.ylabel('Average Eye Height')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/top_6_races_eye_height_boxplot_xgaze.pdf')
plt.close()

# Print summary for verification
print(race_summary[race_summary['race'].isin(top_races)])

# Create a figure with two subplots side by side
# plt.figure(figsize=(10, 10))

# # Pie Chart
# plt.subplot(1, 2, 1)
# plt.pie(race_summary['sample_count'], 
#         labels=race_summary['race'],
#         autopct='%1.1f%%',
#         colors=sns.color_palette('Set2'))
# plt.title('Distribution of Races by Sample Count')

# # Bar Chart
# plt.subplot(1, 2, 2)
# sns.barplot(x='race', y='sample_count', data=race_summary, palette='Set2')
# plt.title('Sample Count by Race')
# plt.xticks(rotation=45)
# plt.ylabel('Number of Samples')

# plt.tight_layout()
# plt.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/race_distribution_mpii.pdf')
# plt.close()


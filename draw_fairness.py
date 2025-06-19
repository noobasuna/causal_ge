import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# Create output directory
output_dir = "/home/tpei0009/gaze-demo/gaze-demo/fairness_figures_360new"
os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_csv('/home/tpei0009/gaze-demo/gaze-demo/output_data_av.csv')

# Extract source and target datasets from Dataset Combination
df['source_dataset'] = df['Dataset Combination'].str.split('_to_', expand=True)[0]
df['target_dataset'] = df['Dataset Combination'].str.split('_to_', expand=True)[1]

# Get unique combinations and test types
dataset_combinations = df['Dataset Combination'].unique()
test_types = df['Test Type'].unique()  # Assuming 'Test Type' column exists
source_datasets = df['source_dataset'].unique()
target_datasets = df['target_dataset'].unique()

# Create custom colormap for better visualization (goes from red to yellow to green)
# colors = [(0.8, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.8, 0.0)]
# cmap_name = 'RYG'
# cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
# # ======== VISUALIZATION 1: Combined Heatmaps for Dataset Combinations ========
# # First get all pivot tables
# pivot_tables = {}
# for combo in dataset_combinations:
#     subset = df[df['Dataset Combination'] == combo]
    
#     # Create pivot table for the heatmap
#     pivot_data = subset.pivot_table(
#         index='Group Pair', 
#         columns='Threshold', 
#         values='Disparate Impact', 
#         aggfunc='mean'
#     )
    
#     # Sort the group pairs by average Disparate Impact (descending)
#     avg_di = pivot_data.mean(axis=1).sort_values(ascending=False)
#     pivot_tables[combo] = pivot_data.reindex(avg_di.index)

# # Now create a grid of heatmaps
# n_combos = len(dataset_combinations)
# n_cols = min(3, n_combos)  # Maximum 3 columns
# n_rows = (n_combos + n_cols - 1) // n_cols  # Calculate needed rows

# # Create a figure with subplots
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
# if n_rows == 1 and n_cols == 1:
#     axes = np.array([axes])  # Make it iterable
# elif n_rows == 1 or n_cols == 1:
#     axes = axes.flatten()

# # Create the heatmaps
# for i, combo in enumerate(dataset_combinations):
#     ax = axes.flatten()[i] if n_combos > 1 else axes
    
#     # Plot the heatmap
#     sns.heatmap(
#         pivot_tables[combo],
#         cmap=cm,
#         vmin=0,
#         vmax=1,
#         annot=False,
#         linewidths=.5,
#         ax=ax,
#         cbar=(i == 0)  # Only show colorbar for the first plot
#     )
#     ax.set_title(f'{combo}', fontsize=10)
    
#     # Adjust label sizes for readability
#     ax.tick_params(axis='both', which='major', labelsize=8)

# # If there are empty subplots, hide them
# for j in range(i + 1, n_rows * n_cols):
#     if j < len(axes.flatten()):
#         axes.flatten()[j].axis('off')

# # Add a single colorbar for reference
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
# sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1))
# sm.set_array([])
# cbar = fig.colorbar(sm, cax=cbar_ax)
# cbar.set_label('Disparate Impact')

# plt.suptitle('Disparate Impact by Group Pair and Threshold - All Dataset Combinations', fontsize=16)
# plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to make room for colorbar
# plt.savefig(f"{output_dir}/combined_heatmaps.png", dpi=300, bbox_inches='tight')
# plt.close()

# # Still keep individual heatmaps if needed
# for combo in dataset_combinations:
#     plt.figure(figsize=(14, 12))
#     sns.heatmap(
#         pivot_tables[combo], 
#         cmap=cm, 
#         vmin=0, 
#         vmax=1, 
#         annot=False, 
#         linewidths=.5, 
#         cbar_kws={'label': 'Disparate Impact'}
#     )
#     plt.title(f'Disparate Impact by Group Pair and Threshold - {combo}')
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/heatmap_{combo}.png", dpi=300, bbox_inches='tight')
#     plt.close()

# # ======== VISUALIZATION 2: Box Plots by Dataset Combination ========
# thresholds_to_plot = [0.3, 0.5, 1]  # Focus on representative thresholds

# for threshold in thresholds_to_plot:
#     plt.figure(figsize=(14, 8))
    
#     # Filter data for the current threshold
#     threshold_data = df[df['Threshold'] == threshold]
    
#     # Create box plot
#     sns.boxplot(
#         x='Dataset Combination', 
#         y='Disparate Impact', 
#         data=threshold_data,
#         hue='Dataset Combination',
#         legend=False
#     )
    
#     plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7)
#     plt.text(0, 0.82, "80% Rule", color='green')
#     plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
#     plt.text(0, 1.02, "Perfect Fairness", color='black')
    
#     plt.title(f'Distribution of Disparate Impact by Dataset Combination (Threshold={threshold})')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/boxplot_threshold_{threshold}.png", dpi=300, bbox_inches='tight')
#     plt.close()

# # ======== VISUALIZATION 3: Line Plots for Average Disparate Impact ========
# # Calculate statistics per threshold and dataset combination
# stats = df.groupby(['Dataset Combination', 'Threshold']).agg({
#     'Disparate Impact': ['mean', 'std']
# }).reset_index()

# plt.figure(figsize=(12, 6))

# for combo in dataset_combinations:
#     combo_stats = stats[stats['Dataset Combination'] == combo]
    
#     plt.plot(
#         combo_stats['Threshold'], 
#         combo_stats['Disparate Impact']['mean'], 
#         'o-', 
#         linewidth=2, 
#         label=combo
#     )
#     plt.fill_between(
#         combo_stats['Threshold'],
#         combo_stats['Disparate Impact']['mean'] - combo_stats['Disparate Impact']['std'],
#         combo_stats['Disparate Impact']['mean'] + combo_stats['Disparate Impact']['std'],
#         alpha=0.2
#     )

# plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% Rule Threshold')
# plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Fairness')
# plt.grid(alpha=0.3)
# plt.xlabel('Threshold')
# plt.ylabel('Average Disparate Impact')
# plt.title('Average Disparate Impact by Dataset Combination')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"{output_dir}/avg_disparate_impact_by_combination.png", dpi=300, bbox_inches='tight')
# plt.close()

# # ======== VISUALIZATION 4: Compare Source Datasets ========
# if len(source_datasets) > 1:
#     source_stats = df.groupby(['source_dataset', 'Threshold']).agg({
#         'Disparate Impact': ['mean', 'std']
#     }).reset_index()
    
#     plt.figure(figsize=(12, 6))
    
#     for source in source_datasets:
#         source_data = source_stats[source_stats['source_dataset'] == source]
        
#         plt.plot(
#             source_data['Threshold'], 
#             source_data['Disparate Impact']['mean'], 
#             'o-', 
#             linewidth=2, 
#             label=f"Source: {source}"
#         )
    
#     plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% Rule Threshold')
#     plt.grid(alpha=0.3)
#     plt.xlabel('Threshold')
#     plt.ylabel('Average Disparate Impact')
#     plt.title('Average Disparate Impact by Source Dataset')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/avg_disparate_impact_by_source.png", dpi=300, bbox_inches='tight')
#     plt.close()

# # ======== VISUALIZATION 5: Compare Target Datasets ========
# if len(target_datasets) > 1:
#     target_stats = df.groupby(['target_dataset', 'Threshold']).agg({
#         'Disparate Impact': ['mean', 'std']
#     }).reset_index()
    
#     plt.figure(figsize=(12, 6))
    
#     for target in target_datasets:
#         target_data = target_stats[target_stats['target_dataset'] == target]
        
#         plt.plot(
#             target_data['Threshold'], 
#             target_data['Disparate Impact']['mean'], 
#             'o-', 
#             linewidth=2, 
#             label=f"Target: {target}"
#         )
    
#     plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% Rule Threshold')
#     plt.grid(alpha=0.3)
#     plt.xlabel('Threshold')
#     plt.ylabel('Average Disparate Impact')
#     plt.title('Average Disparate Impact by Target Dataset')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/avg_disparate_impact_by_target.png", dpi=300, bbox_inches='tight')
#     plt.close()

# # ======== VISUALIZATION: Heatmaps by Test Type ========
# # Create a separate figure for each Test Type
# for test_type in test_types:
#     # Filter data for this test type
#     test_df = df[df['Test Type'] == test_type]
    
#     # Get pivot tables for each dataset combination
#     pivot_tables = {}
#     for combo in dataset_combinations:
#         subset = test_df[test_df['Dataset Combination'] == combo]
#         if subset.empty:
#             continue  # Skip if no data for this combination
            
#         # Create pivot table for the heatmap
#         pivot_data = subset.pivot_table(
#             index='Group Pair', 
#             columns='Threshold', 
#             values='Disparate Impact', 
#             aggfunc='mean'
#         )
        
#         # Sort by average Disparate Impact
#         avg_di = pivot_data.mean(axis=1).sort_values(ascending=False)
#         pivot_tables[combo] = pivot_data.reindex(avg_di.index)
    
#     # Skip if no data for this test type
#     if not pivot_tables:
#         continue
        
#     # Create a grid of heatmaps for this test type
#     n_combos = len(pivot_tables)
#     n_cols = min(3, n_combos)  # Maximum 3 columns
#     n_rows = (n_combos + n_cols - 1) // n_cols  # Calculate needed rows
    
#     # Create a figure with subplots
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
#     if n_rows == 1 and n_cols == 1:
#         axes = np.array([axes])  # Make it iterable
#     elif n_rows == 1 or n_cols == 1:
#         axes = axes.flatten()
    
#     # Create the heatmaps
#     for i, (combo, pivot_data) in enumerate(pivot_tables.items()):
#         ax = axes.flatten()[i] if n_combos > 1 else axes
        
#         # Plot the heatmap
#         sns.heatmap(
#             pivot_data,
#             cmap=cm,
#             vmin=0,
#             vmax=1,
#             annot=False,
#             linewidths=.5,
#             ax=ax,
#             cbar=False  # We'll add a single colorbar at the end
#         )
#         ax.set_title(f'{combo}', fontsize=10)
        
#         # Adjust label sizes for readability
#         ax.tick_params(axis='both', which='major', labelsize=8)
    
#     # If there are empty subplots, hide them
#     for j in range(i + 1, n_rows * n_cols):
#         if j < len(axes.flatten()):
#             axes.flatten()[j].axis('off')
    
#     # Add a single colorbar for reference
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#     sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1))
#     sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cbar_ax)
#     cbar.set_label('Disparate Impact')
    
#     plt.suptitle(f'Disparate Impact - Test Type: {test_type}', fontsize=16)
#     plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout for colorbar
#     plt.savefig(f"{output_dir}/heatmaps_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Also save individual heatmaps for each combination with this test type
#     for combo, pivot_data in pivot_tables.items():
#         plt.figure(figsize=(14, 12))
#         sns.heatmap(
#             pivot_data, 
#             cmap=cm, 
#             vmin=0, 
#             vmax=1, 
#             annot=False, 
#             linewidths=.5, 
#             cbar_kws={'label': 'Disparate Impact'}
#         )
#         plt.title(f'Disparate Impact - {combo} - Test Type: {test_type}')
#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/heatmap_{combo}_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#         plt.close()

# print(f"Visualizations saved to {output_dir}/")

# def analyze_group_pairs_across_datasets():
#     """
#     Generate visualizations comparing how each group pair performs across 
#     different training-testing dataset combinations, organized by test types.
#     """
#     # Create output directory
#     output_dir = "/home/tpei0009/gaze-demo/gaze-demo/group_pair_analysis"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Extract training and testing datasets from Dataset Combination
#     df['Training Dataset'] = df['Dataset Combination'].apply(lambda x: x.split('_to_')[0] if '_to_' in x else x)
#     df['Testing Dataset'] = df['Dataset Combination'].apply(lambda x: x.split('_to_')[1] if '_to_' in x else None)
    
#     # Get unique values
#     group_pairs = df['Group Pair'].unique()
#     test_types = df['Test Type'].unique()
    
#     # Create custom colormap for better visualization
#     colors_di = [(0.8, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.8, 0.0)]  # Red -> Yellow -> Green
#     cm_di = LinearSegmentedColormap.from_list('RYG', colors_di, N=100)
    
#     colors_eo = [(0.0, 0.8, 0.0), (1.0, 1.0, 0.0), (0.8, 0.0, 0.0)]  # Green -> Yellow -> Red
#     cm_eo = LinearSegmentedColormap.from_list('GYR', colors_eo, N=100)
    
#     # ===== HEATMAP ANALYSIS =====
#     # For each group pair, create a separate analysis
#     for pair in group_pairs:
#         # Create directory for this group pair
#         pair_dir = f"{output_dir}/{pair.replace(' ', '_')}"
#         os.makedirs(pair_dir, exist_ok=True)
        
#         # Filter data for this group pair
#         pair_df = df[df['Group Pair'] == pair]
        
#         # Process for each test type
#         for test_type in test_types:
#             # Filter data for this test type
#             test_df = pair_df[pair_df['Test Type'] == test_type]
#             if test_df.empty:
#                 continue
                
#             # Create heatmaps for Disparate Impact
#             # For rows: Training Datasets, For columns: Testing Datasets
#             # Create pivot tables
#             training_datasets = test_df['Training Dataset'].unique()
#             heatmap_data_di = []
#             heatmap_data_eo = []
            
#             # Organize data for each threshold
#             for threshold in sorted(test_df['Threshold'].unique()):
#                 threshold_df = test_df[test_df['Threshold'] == threshold]
                
#                 # Create matrix for this threshold
#                 matrix_di = {}
#                 matrix_eo = {}
                
#                 # For each training and testing combination
#                 for train_data in threshold_df['Training Dataset'].unique():
#                     for test_data in threshold_df[threshold_df['Training Dataset'] == train_data]['Testing Dataset'].unique():
#                         subset = threshold_df[
#                             (threshold_df['Training Dataset'] == train_data) & 
#                             (threshold_df['Testing Dataset'] == test_data)
#                         ]
                        
#                         if not subset.empty:
#                             # Use the combination as key
#                             combo = f"{train_data}_to_{test_data}"
#                             matrix_di[combo] = subset['Disparate Impact'].mean()
#                             matrix_eo[combo] = subset['ΔTPR (Equal Opportunity)'].mean()
                
#                 # Add to dataframe list
#                 if matrix_di:
#                     heatmap_data_di.append({
#                         'Threshold': threshold,
#                         **matrix_di
#                     })
                
#                 if matrix_eo:
#                     heatmap_data_eo.append({
#                         'Threshold': threshold,
#                         **matrix_eo
#                     })
            
#             # Create dataframes for heatmaps
#             if heatmap_data_di:
#                 df_di = pd.DataFrame(heatmap_data_di).set_index('Threshold')
                
#                 # Create heatmap
#                 plt.figure(figsize=(14, 10))
#                 sns.heatmap(
#                     df_di, 
#                     cmap=cm_di, 
#                     vmin=0, 
#                     vmax=1, 
#                     annot=True, 
#                     fmt=".2f",
#                     linewidths=.5, 
#                     cbar_kws={'label': 'Disparate Impact'}
#                 )
#                 plt.title(f'Disparate Impact - {pair} - {test_type}')
#                 plt.xlabel('Dataset Combinations (Training_to_Testing)')
#                 plt.ylabel('Threshold')
#                 plt.xticks(rotation=45, ha='right')
#                 plt.tight_layout()
#                 plt.savefig(f"{pair_dir}/DI_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#                 plt.close()
            
#             if heatmap_data_eo:
#                 df_eo = pd.DataFrame(heatmap_data_eo).set_index('Threshold')
                
#                 # Create heatmap
#                 plt.figure(figsize=(14, 10))
#                 sns.heatmap(
#                     df_eo, 
#                     cmap=cm_eo, 
#                     vmin=0, 
#                     vmax=0.2,  # Adjust based on your data range for ΔTPR
#                     annot=True, 
#                     fmt=".2f",
#                     linewidths=.5, 
#                     cbar_kws={'label': 'ΔTPR (Equal Opportunity)'}
#                 )
#                 plt.title(f'Equal Opportunity - {pair} - {test_type}')
#                 plt.xlabel('Dataset Combinations (Training_to_Testing)')
#                 plt.ylabel('Threshold')
#                 plt.xticks(rotation=45, ha='right')
#                 plt.tight_layout()
#                 plt.savefig(f"{pair_dir}/EO_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#                 plt.close()
            
#         # ===== LINE PLOT ANALYSIS =====
#         # Create line plots showing fairness metrics across thresholds for different dataset combinations
#         for test_type in test_types:
#             test_df = pair_df[pair_df['Test Type'] == test_type]
#             if test_df.empty:
#                 continue
                
#             # Disparate Impact line plot
#             plt.figure(figsize=(14, 8))
            
#             # Group by dataset combination and threshold
#             dataset_combos = test_df.groupby(['Dataset Combination', 'Threshold']).agg({
#                 'Disparate Impact': 'mean'
#             }).reset_index()
            
#             # Plot lines for each dataset combination
#             for combo in dataset_combos['Dataset Combination'].unique():
#                 combo_data = dataset_combos[dataset_combos['Dataset Combination'] == combo]
#                 plt.plot(
#                     combo_data['Threshold'],
#                     combo_data['Disparate Impact'],
#                     'o-',
#                     linewidth=2,
#                     label=combo
#                 )
            
#             plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% Rule Threshold')
#             plt.grid(alpha=0.3)
#             plt.xlabel('Threshold')
#             plt.ylabel('Disparate Impact')
#             plt.title(f'Disparate Impact by Dataset Combination - {pair} - {test_type}')
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.tight_layout()
#             plt.savefig(f"{pair_dir}/DI_lines_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#             plt.close()
            
#             # Equal Opportunity line plot
#             plt.figure(figsize=(14, 8))
            
#             # Group by dataset combination and threshold
#             dataset_combos = test_df.groupby(['Dataset Combination', 'Threshold']).agg({
#                 'ΔTPR (Equal Opportunity)': 'mean'
#             }).reset_index()
            
#             # Plot lines for each dataset combination
#             for combo in dataset_combos['Dataset Combination'].unique():
#                 combo_data = dataset_combos[dataset_combos['Dataset Combination'] == combo]
#                 plt.plot(
#                     combo_data['Threshold'],
#                     combo_data['ΔTPR (Equal Opportunity)'],
#                     'o-',
#                     linewidth=2,
#                     label=combo
#                 )
            
#             plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='ΔTPR = 0.05 threshold')
#             plt.grid(alpha=0.3)
#             plt.xlabel('Threshold')
#             plt.ylabel('ΔTPR (Equal Opportunity)')
#             plt.title(f'Equal Opportunity by Dataset Combination - {pair} - {test_type}')
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.tight_layout()
#             plt.savefig(f"{pair_dir}/EO_lines_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#             plt.close()
        
#         # ===== SAME TRAINING DATASET ANALYSIS =====
#         # Create visualizations for same training dataset across different testing datasets
#         training_datasets = pair_df['Training Dataset'].unique()
        
#         for train_data in training_datasets:
#             train_data_df = pair_df[pair_df['Training Dataset'] == train_data]
            
#             if 'Testing Dataset' not in train_data_df.columns or train_data_df['Testing Dataset'].isnull().all():
#                 continue
            
#             # Create subplot for each test type
#             for test_type in test_types:
#                 test_type_df = train_data_df[train_data_df['Test Type'] == test_type]
#                 if test_type_df.empty:
#                     continue
                
#                 # Create a single figure with multiple subplots
#                 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
                
#                 # Get unique testing datasets
#                 test_datasets = test_type_df['Testing Dataset'].unique()
#                 colors = plt.cm.tab10(np.linspace(0, 1, len(test_datasets)))
                
#                 # Plot Disparate Impact
#                 for i, test_data in enumerate(test_datasets):
#                     subset = test_type_df[test_type_df['Testing Dataset'] == test_data]
                    
#                     if not subset.empty:
#                         ax1.plot(subset['Threshold'], subset['Disparate Impact'], 
#                                 marker='o', linestyle='-', linewidth=2,
#                                 label=f"Testing on {test_data}",
#                                 color=colors[i])
                
#                 ax1.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='DI = 0.8 threshold')
#                 ax1.set_xlabel('Threshold', fontsize=12)
#                 ax1.set_ylabel('Disparate Impact', fontsize=12)
#                 ax1.set_title(f'Disparate Impact - {test_type}', fontsize=14)
#                 ax1.grid(True, alpha=0.3)
#                 ax1.legend(loc='best')
                
#                 # Plot Equal Opportunity
#                 for i, test_data in enumerate(test_datasets):
#                     subset = test_type_df[test_type_df['Testing Dataset'] == test_data]
                    
#                     if not subset.empty:
#                         ax2.plot(subset['Threshold'], subset['ΔTPR (Equal Opportunity)'], 
#                                 marker='o', linestyle='-', linewidth=2,
#                                 label=f"Testing on {test_data}",
#                                 color=colors[i])
                
#                 ax2.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='ΔTPR = 0.05 threshold')
#                 ax2.set_xlabel('Threshold', fontsize=12)
#                 ax2.set_ylabel('ΔTPR (Equal Opportunity)', fontsize=12)
#                 ax2.set_title(f'Equal Opportunity - {test_type}', fontsize=14)
#                 ax2.grid(True, alpha=0.3)
#                 ax2.legend(loc='best')
                
#                 # Add overall title
#                 plt.suptitle(f'{pair} - Training on {train_data} - {test_type}', fontsize=16)
#                 plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
#                 plt.savefig(f"{pair_dir}/Train_{train_data}_TestType_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#                 plt.close()
                
#                 # Create a scatter plot showing the trade-off
#                 plt.figure(figsize=(12, 10))
                
#                 for i, test_data in enumerate(test_datasets):
#                     subset = test_type_df[test_type_df['Testing Dataset'] == test_data]
                    
#                     if not subset.empty:
#                         # Create scatter plot with size indicating threshold
#                         sizes = subset['Threshold'] * 20 + 50
                        
#                         plt.scatter(subset['ΔTPR (Equal Opportunity)'], subset['Disparate Impact'], 
#                                    s=sizes, 
#                                    label=f"Testing on {test_data}",
#                                    color=colors[i], alpha=0.7)
                        
#                         # Add threshold labels
#                         for idx, row in subset.iterrows():
#                             plt.annotate(f"{row['Threshold']}", 
#                                        (row['ΔTPR (Equal Opportunity)'], row['Disparate Impact']),
#                                        xytext=(5, 5), textcoords='offset points', fontsize=8)
                
#                 # Add reference lines
#                 plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.3, label='DI = 0.8 threshold')
#                 plt.axvline(x=0.05, color='g', linestyle='--', alpha=0.3, label='ΔTPR = 0.05 threshold')
                
#                 plt.xlabel('ΔTPR (Equal Opportunity)', fontsize=14)
#                 plt.ylabel('Disparate Impact', fontsize=14)
#                 plt.title(f'Trade-off: {pair} - Training on {train_data} - {test_type}', fontsize=16)
#                 plt.legend()
#                 plt.grid(True, alpha=0.3)
#                 plt.tight_layout()
#                 plt.savefig(f"{pair_dir}/Tradeoff_Train_{train_data}_TestType_{test_type.replace(' ', '_')}.png", 
#                            dpi=300, bbox_inches='tight')
#                 plt.close()
    
#     print(f"Group pair analysis saved to {output_dir}/")

# # Run the analysis
# analyze_group_pairs_across_datasets()

# def plot_group_pair_heatmaps():
#     """
#     Generate heatmaps showing how each group pair performs across different training datasets.
#     """
#     # Create output directory
#     output_dir = "/home/tpei0009/gaze-demo/gaze-demo/group_pair_heatmaps"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Extract training and testing datasets
#     df['Training Dataset'] = df['Dataset Combination'].apply(lambda x: x.split('_to_')[0] if '_to_' in x else x)
#     df['Testing Dataset'] = df['Dataset Combination'].apply(lambda x: x.split('_to_')[1] if '_to_' in x else None)
    
#     # Get unique values
#     group_pairs = df['Group Pair'].unique()
#     test_types = df['Test Type'].unique()
    
#     # Create custom colormaps
#     di_cmap = LinearSegmentedColormap.from_list('RYG', [(0.8, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.8, 0.0)], N=100)
#     eo_cmap = LinearSegmentedColormap.from_list('GYR', [(0.0, 0.8, 0.0), (1.0, 1.0, 0.0), (0.8, 0.0, 0.0)], N=100)
    
#     # For each group pair
#     for pair in group_pairs:
#         # Create directory for this group pair
#         pair_dir = f"{output_dir}/{pair.replace(' ', '_')}"
#         os.makedirs(pair_dir, exist_ok=True)
        
#         # Filter data for this group pair
#         pair_df = df[df['Group Pair'] == pair]
        
#         # For each test type
#         for test_type in test_types:
#             test_df = pair_df[pair_df['Test Type'] == test_type]
#             if test_df.empty:
#                 continue
                
#             # ===== HEATMAP BY TRAINING DATASET =====
#             # Get unique training datasets 
#             training_datasets = test_df['Training Dataset'].unique()
            
#             # Create pivot tables for heatmaps - Training datasets vs. Thresholds
#             # Disparate Impact
#             di_pivot = pd.pivot_table(
#                 data=test_df,
#                 values='Disparate Impact',
#                 index='Training Dataset',
#                 columns='Threshold',
#                 aggfunc='mean'
#             )
            
#             # Sort training datasets by average Disparate Impact
#             if not di_pivot.empty:
#                 avg_di = di_pivot.mean(axis=1).sort_values(ascending=False)
#                 di_pivot = di_pivot.reindex(avg_di.index)
                
#                 # Create heatmap
#                 plt.figure(figsize=(15, 10))
#                 sns.heatmap(
#                     di_pivot, 
#                     cmap=di_cmap, 
#                     vmin=0, 
#                     vmax=1, 
#                     annot=True, 
#                     fmt=".2f",
#                     linewidths=.5, 
#                     cbar_kws={'label': 'Disparate Impact'}
#                 )
#                 plt.title(f'Disparate Impact by Training Dataset - {pair} - {test_type}', fontsize=16)
#                 plt.ylabel('Training Dataset', fontsize=12)
#                 plt.xlabel('Threshold', fontsize=12)
#                 plt.tight_layout()
#                 plt.savefig(f"{pair_dir}/DI_heatmap_by_training_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#                 plt.close()
            
#             # Equal Opportunity 
#             eo_pivot = pd.pivot_table(
#                 data=test_df,
#                 values='ΔTPR (Equal Opportunity)',
#                 index='Training Dataset',
#                 columns='Threshold',
#                 aggfunc='mean'
#             )
            
#             # Sort training datasets by average Equal Opportunity
#             if not eo_pivot.empty:
#                 avg_eo = eo_pivot.mean(axis=1).sort_values(ascending=True)
#                 eo_pivot = eo_pivot.reindex(avg_eo.index)
                
#                 # Create heatmap
#                 plt.figure(figsize=(15, 10))
#                 sns.heatmap(
#                     eo_pivot, 
#                     cmap=eo_cmap, 
#                     vmin=0, 
#                     vmax=0.2,  # Adjust based on your data range
#                     annot=True, 
#                     fmt=".2f",
#                     linewidths=.5, 
#                     cbar_kws={'label': 'ΔTPR (Equal Opportunity)'}
#                 )
#                 plt.title(f'Equal Opportunity by Training Dataset - {pair} - {test_type}', fontsize=16)
#                 plt.ylabel('Training Dataset', fontsize=12)
#                 plt.xlabel('Threshold', fontsize=12)
#                 plt.tight_layout()
#                 plt.savefig(f"{pair_dir}/EO_heatmap_by_training_{test_type.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
#                 plt.close()
            
#             # ===== HEATMAP BY TRAINING-TESTING PAIRS =====
#             # Create heatmaps for training-testing dataset pairs
#             # We'll create a heatmap for each threshold
#             thresholds = sorted(test_df['Threshold'].unique())
            
#             for threshold in thresholds:
#                 threshold_df = test_df[test_df['Threshold'] == threshold]
                
#                 # Create pivot table with training as rows and testing as columns
#                 # For Disparate Impact
#                 di_pivot_pairs = pd.pivot_table(
#                     data=threshold_df,
#                     values='Disparate Impact',
#                     index='Training Dataset',
#                     columns='Testing Dataset',
#                     aggfunc='mean'
#                 )
                
#                 if not di_pivot_pairs.empty and not di_pivot_pairs.isnull().all().all():
#                     plt.figure(figsize=(12, 8))
#                     sns.heatmap(
#                         di_pivot_pairs, 
#                         cmap=di_cmap, 
#                         vmin=0, 
#                         vmax=1, 
#                         annot=True, 
#                         fmt=".2f",
#                         linewidths=.5, 
#                         cbar_kws={'label': 'Disparate Impact'}
#                     )
#                     plt.title(f'Disparate Impact: {pair} - {test_type} - Threshold {threshold}', fontsize=16)
#                     plt.ylabel('Training Dataset', fontsize=12)
#                     plt.xlabel('Testing Dataset', fontsize=12)
#                     plt.tight_layout()
#                     plt.savefig(f"{pair_dir}/DI_train_test_matrix_threshold_{threshold}_{test_type.replace(' ', '_')}.png", 
#                               dpi=300, bbox_inches='tight')
#                     plt.close()
                
#                 # For Equal Opportunity
#                 eo_pivot_pairs = pd.pivot_table(
#                     data=threshold_df,
#                     values='ΔTPR (Equal Opportunity)',
#                     index='Training Dataset',
#                     columns='Testing Dataset',
#                     aggfunc='mean'
#                 )
                
#                 if not eo_pivot_pairs.empty and not eo_pivot_pairs.isnull().all().all():
#                     plt.figure(figsize=(12, 8))
#                     sns.heatmap(
#                         eo_pivot_pairs, 
#                         cmap=eo_cmap, 
#                         vmin=0, 
#                         vmax=0.2,  # Adjust based on your data range
#                         annot=True, 
#                         fmt=".2f",
#                         linewidths=.5, 
#                         cbar_kws={'label': 'ΔTPR (Equal Opportunity)'}
#                     )
#                     plt.title(f'Equal Opportunity: {pair} - {test_type} - Threshold {threshold}', fontsize=16)
#                     plt.ylabel('Training Dataset', fontsize=12)
#                     plt.xlabel('Testing Dataset', fontsize=12)
#                     plt.tight_layout()
#                     plt.savefig(f"{pair_dir}/EO_train_test_matrix_threshold_{threshold}_{test_type.replace(' ', '_')}.png", 
#                               dpi=300, bbox_inches='tight')
#                     plt.close()
    
#     print(f"Group pair heatmap analysis saved to {output_dir}/")

# # Run the analysis
# plot_group_pair_heatmaps()
# def analyze_group_pairs_by_training_dataset():
#     """
#     Analyze and output the mean values of different group pairs 
#     in the same training dataset for different test types.
#     """
#     # Create output directory
#     output_dir = "/home/tpei0009/gaze-demo/gaze-demo/training_dataset_group_analysis"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Extract training dataset from the Dataset Combination if not already done
#     if 'Training Dataset' not in df.columns:
#         df['Training Dataset'] = df['Dataset Combination'].apply(lambda x: x.split('_to_')[0] if '_to_' in x else x)
    
#     # Get unique values
#     training_datasets = df['Training Dataset'].unique()
#     test_types = df['Test Type'].unique()
    
#     # For each training dataset and test type, create comparisons of group pairs
#     for train_data in training_datasets:
#         # Create directory for this training dataset
#         train_dir = f"{output_dir}/{train_data}"
#         os.makedirs(train_dir, exist_ok=True)
        
#         # Filter data for this training dataset
#         train_df = df[df['Training Dataset'] == train_data]
        
#         # For each test type
#         for test_type in test_types:
#             test_df = train_df[train_df['Test Type'] == test_type]
#             if test_df.empty:
#                 continue
            
#             # Calculate mean values for each group pair
#             group_means = test_df.groupby(['Group Pair', 'Threshold']).agg({
#                 'Disparate Impact': 'mean',
#                 'ΔTPR (Equal Opportunity)': 'mean'
#             }).reset_index()
            
#             # Create a table visualization for each threshold
#             for threshold in sorted(test_df['Threshold'].unique()):
#                 # Get data for this threshold
#                 threshold_data = group_means[group_means['Threshold'] == threshold]
                
#                 # Create pretty table figure
#                 fig, ax = plt.subplots(figsize=(12, len(threshold_data) * 0.5 + 2))
#                 ax.axis('tight')
#                 ax.axis('off')
                
#                 # Prepare table data - format numbers to 3 decimal places
#                 table_data = threshold_data[['Group Pair', 'Disparate Impact', 'ΔTPR (Equal Opportunity)']]
#                 # Convert to list of lists and format each numeric value
#                 formatted_data = []
#                 for _, row in table_data.iterrows():
#                     formatted_row = [
#                         row['Group Pair'],  # Keep as string
#                         f"{row['Disparate Impact']:.3f}",  # Format to 3 decimal places
#                         f"{row['ΔTPR (Equal Opportunity)']:.3f}"  # Format to 3 decimal places
#                     ]
#                     formatted_data.append(formatted_row)
                
#                 # Create table with formatted data
#                 table = ax.table(
#                     cellText=formatted_data,
#                     colLabels=table_data.columns,
#                     loc='center',
#                     cellLoc='center'
#                 )
                
#                 # Style the table
#                 table.auto_set_font_size(False)
#                 table.set_fontsize(12)
#                 table.scale(1.2, 1.5)
                
#                 # Color cells based on values
#                 for i in range(len(table_data)):
#                     # Color Disparate Impact cells
#                     di_value = table_data.iloc[i, 1]
#                     if di_value >= 0.8:
#                         table[(i+1, 1)].set_facecolor((0.0, 0.8, 0.0, 0.3))  # Green for good
#                     elif di_value >= 0.6:
#                         table[(i+1, 1)].set_facecolor((1.0, 1.0, 0.0, 0.3))  # Yellow for borderline
#                     else:
#                         table[(i+1, 1)].set_facecolor((0.8, 0.0, 0.0, 0.3))  # Red for bad
                    
#                     # Color Equal Opportunity cells
#                     eo_value = table_data.iloc[i, 2]
#                     if eo_value <= 0.05:
#                         table[(i+1, 2)].set_facecolor((0.0, 0.8, 0.0, 0.3))  # Green for good
#                     elif eo_value <= 0.1:
#                         table[(i+1, 2)].set_facecolor((1.0, 1.0, 0.0, 0.3))  # Yellow for borderline
#                     else:
#                         table[(i+1, 2)].set_facecolor((0.8, 0.0, 0.0, 0.3))  # Red for bad
                
#                 plt.title(f'Group Pair Metrics: {train_data} - {test_type} - Threshold {threshold}', fontsize=16)
#                 plt.tight_layout()
#                 plt.savefig(f"{train_dir}/metrics_table_{test_type.replace(' ', '_')}_threshold_{threshold}.png", 
#                            dpi=300, bbox_inches='tight')
#                 plt.close()
    
#     # Also save the raw data as CSV files for reference
#     for train_data in training_datasets:
#         train_df = df[df['Training Dataset'] == train_data]
        
#         for test_type in test_types:
#             test_df = train_df[train_df['Test Type'] == test_type]
#             if not test_df.empty:
#                 # Group and calculate means
#                 summary = test_df.groupby(['Group Pair', 'Threshold']).agg({
#                     'Disparate Impact': ['mean', 'std'],
#                     'ΔTPR (Equal Opportunity)': ['mean', 'std']
#                 }).reset_index()
                
#                 # Save as CSV
#                 csv_path = f"{output_dir}/{train_data}/summary_{test_type.replace(' ', '_')}.csv"
#                 summary.to_csv(csv_path, index=False)
    
#     print(f"Group pair analysis by training dataset saved to {output_dir}/")

# # Run the analysis
# analyze_group_pairs_by_training_dataset()

def compare_test_types_by_group_pair():
    """
    Compare the mean fairness metrics of different group pairs across test types
    in the same training dataset, averaging across all thresholds.
    """
    # Create output directory
    output_dir = "/home/tpei0009/gaze-demo/gaze-demo/test_type_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract training dataset from Dataset Combination if not already done
    if 'Training Dataset' not in df.columns:
        df['Training Dataset'] = df['Dataset Combination'].apply(lambda x: x.split('_to_')[0] if '_to_' in x else x)
    
    # Get unique values
    training_datasets = df['Training Dataset'].unique()
    test_types = df['Test Type'].unique()
    
    # For each training dataset, create a comparative analysis
    for train_data in training_datasets:
        # Create directory for this training dataset
        train_dir = f"{output_dir}/{train_data}"
        os.makedirs(train_dir, exist_ok=True)
        
        # Filter data for this training dataset
        train_df = df[df['Training Dataset'] == train_data]
        
        # Calculate mean values across all thresholds for each group pair and test type
        group_means = train_df.groupby(['Group Pair', 'Test Type']).agg({
            'Disparate Impact': 'mean',
            'ΔTPR (Equal Opportunity)': 'mean'
        }).reset_index()
        
        # Get all unique group pairs for this training dataset
        group_pairs = train_df['Group Pair'].unique()
        
        # Create a bar chart comparing test types for each metric
        # Disparate Impact
        plt.figure(figsize=(15, 10))
        
        # Set up bar positions
        bar_width = 0.25
        index = np.arange(len(group_pairs))
        
        # Plot bars for each test type
        for i, test_type in enumerate(test_types):
            test_data = group_means[group_means['Test Type'] == test_type]
            # Align with group pairs
            di_values = []
            for pair in group_pairs:
                pair_data = test_data[test_data['Group Pair'] == pair]
                if not pair_data.empty:
                    di_values.append(pair_data['Disparate Impact'].values[0])
                else:
                    di_values.append(0)  # No data for this combination
            
            plt.bar(index + i*bar_width, di_values, bar_width, 
                   label=test_type, alpha=0.7)
        
        # Add reference line for 80% rule
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% Rule Threshold')
        
        # Customize plot
        plt.xlabel('Group Pair', fontsize=14)
        plt.ylabel('Mean Disparate Impact', fontsize=14)
        plt.title(f'Mean Disparate Impact by Group Pair and Test Type - {train_data}', fontsize=16)
        plt.xticks(index + bar_width, group_pairs, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f"{train_dir}/DI_by_test_type.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Equal Opportunity
        plt.figure(figsize=(15, 10))
        
        # Plot bars for each test type
        for i, test_type in enumerate(test_types):
            test_data = group_means[group_means['Test Type'] == test_type]
            # Align with group pairs
            eo_values = []
            for pair in group_pairs:
                pair_data = test_data[test_data['Group Pair'] == pair]
                if not pair_data.empty:
                    eo_values.append(pair_data['ΔTPR (Equal Opportunity)'].values[0])
                else:
                    eo_values.append(0)  # No data for this combination
            
            plt.bar(index + i*bar_width, eo_values, bar_width, 
                   label=test_type, alpha=0.7)
        
        # Add reference line for equal opportunity threshold
        plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='ΔTPR = 0.05 threshold')
        
        # Customize plot
        plt.xlabel('Group Pair', fontsize=14)
        plt.ylabel('Mean ΔTPR (Equal Opportunity)', fontsize=14)
        plt.title(f'Mean ΔTPR by Group Pair and Test Type - {train_data}', fontsize=16)
        plt.xticks(index + bar_width, group_pairs, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f"{train_dir}/EO_by_test_type.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create tables with the exact values
        # Pivot for Disparate Impact
        di_pivot = pd.pivot_table(
            data=group_means,
            values='Disparate Impact',
            index='Group Pair',
            columns='Test Type',
            aggfunc='mean'
        )
        
        # Create heatmap-style table for Disparate Impact
        plt.figure(figsize=(12, len(di_pivot) * 0.5 + 2))
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Format the data for display
        di_table = pd.DataFrame(di_pivot).round(3)
        
        # Create a colorful table
        colors = []
        for _, row in di_table.iterrows():
            row_colors = []
            for val in row:
                if val >= 0.8:
                    row_colors.append((0.0, 0.8, 0.0, 0.3))  # Green
                elif val >= 0.6:
                    row_colors.append((1.0, 1.0, 0.0, 0.3))  # Yellow
                else:
                    row_colors.append((0.8, 0.0, 0.0, 0.3))  # Red
            colors.append(row_colors)
        
        table = plt.table(
            cellText=di_table.values,
            rowLabels=di_table.index,
            colLabels=di_table.columns,
            cellColours=colors,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title(f'Mean Disparate Impact by Group Pair and Test Type - {train_data}', fontsize=16)
        plt.savefig(f"{train_dir}/DI_table_by_test_type.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pivot for Equal Opportunity
        eo_pivot = pd.pivot_table(
            data=group_means,
            values='ΔTPR (Equal Opportunity)',
            index='Group Pair',
            columns='Test Type',
            aggfunc='mean'
        )
        
        # Create heatmap-style table for Equal Opportunity
        plt.figure(figsize=(12, len(eo_pivot) * 0.5 + 2))
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Format the data for display
        eo_table = pd.DataFrame(eo_pivot).round(3)
        
        # Create a colorful table
        colors = []
        for _, row in eo_table.iterrows():
            row_colors = []
            for val in row:
                if val <= 0.05:
                    row_colors.append((0.0, 0.8, 0.0, 0.3))  # Green
                elif val <= 0.1:
                    row_colors.append((1.0, 1.0, 0.0, 0.3))  # Yellow
                else:
                    row_colors.append((0.8, 0.0, 0.0, 0.3))  # Red
            colors.append(row_colors)
        
        table = plt.table(
            cellText=eo_table.values,
            rowLabels=eo_table.index,
            colLabels=eo_table.columns,
            cellColours=colors,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title(f'Mean ΔTPR by Group Pair and Test Type - {train_data}', fontsize=16)
        plt.savefig(f"{train_dir}/EO_table_by_test_type.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the raw data as CSV for reference
        summary_csv = pd.pivot_table(
            data=group_means,
            values=['Disparate Impact', 'ΔTPR (Equal Opportunity)'],
            index='Group Pair',
            columns='Test Type'
        )
        
        summary_csv.to_csv(f"{train_dir}/test_type_comparison_summary.csv")
    
    print(f"Test type comparison by group pair saved to {output_dir}/")
    # ===== ENHANCED VISUALIZATIONS TO HIGHLIGHT "OURS" PERFORMANCE =====
    # Create a dedicated directory for comparison visualizations
    comparison_dir = f"{output_dir}/enhanced_viz"
    os.makedirs(comparison_dir, exist_ok=True)

    print(f"Creating enhanced visualizations in {comparison_dir}...")
    print(f"Available test types: {test_types}")

    # Check if 'Ours' exists in the test types (case insensitive)
    ours_type = None
    for test_type in test_types:
        if test_type.lower() == 'ours':
            ours_type = test_type
            break

    # If 'Ours' not found, use the first test type as reference
    if ours_type is None:
        ours_type = test_types[0]
        print(f"Warning: 'Ours' not found in test types. Using '{ours_type}' as reference.")

    # 1. DIRECT TEST TYPE COMPARISON HEATMAP
    # Compute mean DI for each group pair and test type across all thresholds and dataset combinations
    mean_di = df.groupby(['Group Pair', 'Test Type'])['Disparate Impact'].mean().reset_index()
    mean_di_pivot = mean_di.pivot(index='Group Pair', columns='Test Type', values='Disparate Impact')

    # Drop any group pairs with missing values in the reference column
    valid_pairs = mean_di_pivot.dropna(subset=[ours_type]).copy()

    if len(valid_pairs) > 0:
        # Sort rows by the value in reference column (descending)
        valid_pairs = valid_pairs.sort_values(ours_type, ascending=False)
        
        # Create heatmap
        plt.figure(figsize=(10, max(8, len(valid_pairs) * 0.4)))
        ax = sns.heatmap(
            valid_pairs,
            cmap='RdYlGn',
            vmin=0.4, vmax=1.0,
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Disparate Impact (higher is better)'}
        )
        
        # Highlight reference column with a bold border
        for i in range(len(valid_pairs)):
            try:
                # Get the column index for reference type
                ours_idx = list(valid_pairs.columns).index(ours_type)
                # Add bold border to each cell in reference column
                rect = plt.Rectangle(
                    (ours_idx, i), 1, 1, 
                    fill=False, edgecolor='blue', lw=2, 
                    clip_on=False
                )
                ax.add_patch(rect)
            except ValueError:
                print(f"Warning: Could not highlight column {ours_type}")
        
        plt.title(f'Disparate Impact Comparison: {ours_type} vs Other Test Types', fontsize=16)
        plt.ylabel('Group Pair', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{comparison_dir}/test_type_comparison_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. IMPROVEMENT BAR CHART
        # Calculate improvement of reference over other test types
        other_types = [t for t in valid_pairs.columns if t != ours_type]
        
        if other_types:  # Only proceed if there are other test types to compare with
            # Calculate max of other types for each group pair
            valid_pairs['Max Others'] = valid_pairs[other_types].max(axis=1)
            valid_pairs['Improvement'] = valid_pairs[ours_type] - valid_pairs['Max Others']
            # Calculate percentage improvement (handle division by zero)
            valid_pairs['Improvement %'] = valid_pairs.apply(
                lambda row: (row[ours_type] - row['Max Others']) / max(row['Max Others'], 0.001) * 100, 
                axis=1
            )
            
            # Sort by improvement (descending)
            improvement_df = valid_pairs.sort_values('Improvement', ascending=False)
            
            # Select top 15 group pairs (or fewer if less are available)
            n_pairs = min(15, len(improvement_df))
            top_pairs = improvement_df.head(n_pairs)
            
            # Create horizontal bar chart for improvement
            plt.figure(figsize=(12, max(6, n_pairs * 0.4)))
            bars = plt.barh(
                y=top_pairs.index,
                width=top_pairs['Improvement'],
                color=plt.cm.RdYlGn(np.minimum(0.9, np.maximum(0.1, 0.5 + top_pairs['Improvement']/0.4)))
            )
            
            # Add value labels to the bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                improvement_pct = top_pairs['Improvement %'].iloc[i]
                label = f"+{width:.2f} ({improvement_pct:.1f}%)" if width > 0 else f"{width:.2f} ({improvement_pct:.1f}%)"
                plt.text(
                    max(width + 0.01, 0.01) if width >= 0 else width - 0.05,
                    bar.get_y() + bar.get_height()/2,
                    label,
                    va='center'
                )
            
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
            plt.grid(axis='x', alpha=0.3)
            plt.xlabel(f'Improvement in Disparate Impact ({ours_type} - Best Other)', fontsize=14)
            plt.title(f'Top {n_pairs} Group Pairs: Improvement of "{ours_type}" over Best Alternative', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{comparison_dir}/improvement_barchart.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. SUMMARY STATISTICS VISUALIZATION
            # Calculate statistics
            win_count = (valid_pairs[ours_type] > valid_pairs['Max Others']).sum()
            total_pairs = len(valid_pairs)
            win_percentage = (win_count / total_pairs) * 100 if total_pairs > 0 else 0
            
            avg_improvement = valid_pairs['Improvement'].mean()
            avg_improvement_pct = valid_pairs['Improvement %'].mean()
            
            # Create figure with summary statistics
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            
            # Draw a fancy box
            plt.text(0.5, 0.9, f"PERFORMANCE SUMMARY: {ours_type}", 
                    ha='center', fontsize=20, fontweight='bold')
            
            plt.text(0.5, 0.75, f"{ours_type} outperforms all others in {win_count} out of {total_pairs} group pairs", 
                    ha='center', fontsize=16)
            plt.text(0.5, 0.65, f"Success Rate: {win_percentage:.1f}%", 
                    ha='center', fontsize=16, color='green' if win_percentage > 50 else 'red')
            
            plt.text(0.5, 0.5, f"Average Absolute Improvement: {avg_improvement:.3f}", 
                    ha='center', fontsize=16)
            plt.text(0.5, 0.4, f"Average Percentage Improvement: {avg_improvement_pct:.2f}%", 
                    ha='center', fontsize=16)
            
            # Add a circular gauge for win percentage
            center = (0.5, 0.2)
            radius = 0.15
            
            # Draw gauge background
            circle = plt.Circle(center, radius, fill=False, color='gray')
            plt.gca().add_patch(circle)
            
            # Draw colored arc for win percentage (clamp to 0-100%)
            angle = min(100, max(0, win_percentage)) * 3.6  # Convert to degrees (out of 360)
            arc = plt.matplotlib.patches.Arc(
                center, radius*2, radius*2,
                theta1=180, theta2=180+angle,
                color='green' if win_percentage > 50 else 'red',
                linewidth=10
            )
            plt.gca().add_patch(arc)
            
            # Add percentage in the middle
            plt.text(center[0], center[1], f"{win_percentage:.1f}%", 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            
            plt.savefig(f"{comparison_dir}/performance_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. TEST TYPE RANKING VISUALIZATION
            # Calculate mean DI for each test type across all group pairs
            test_type_means = mean_di_pivot.mean()
            test_type_ranks = test_type_means.sort_values(ascending=False)
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                test_type_ranks.index,
                test_type_ranks.values,
                color=[(0.2, 0.7, 0.3) if t == ours_type else (0.7, 0.7, 0.7) for t in test_type_ranks.index]
            )
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom',
                    fontweight='bold' if list(test_type_ranks.index)[int(bar.get_x())] == ours_type else 'normal'
                )
            
            plt.axhline(y=0.8, color='red', linestyle='--', label='80% Rule Threshold')
            plt.grid(axis='y', alpha=0.3)
            plt.ylabel('Mean Disparate Impact', fontsize=14)
            plt.title('Average Performance Ranking of Test Types', fontsize=16)
            plt.ylim(0, max(test_type_means) * 1.1)  # Add some space at the top
            plt.tight_layout()
            plt.savefig(f"{comparison_dir}/test_type_ranking.png", dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Enhanced visualizations saved to {comparison_dir}/")

    # ===== ENHANCED VISUALIZATIONS BY TRAINING/TESTING SETS =====
    # Create dedicated directory
    comparison_dir = f"{output_dir}/enhanced_training_testing_viz"
    os.makedirs(comparison_dir, exist_ok=True)

    print(f"Creating separated visualizations in {comparison_dir}...")

    # Find reference test type (Ours or first available)
    ours_type = None
    for test_type in test_types:
        if test_type.lower() == 'ours':
            ours_type = test_type
            break
    if ours_type is None:
        ours_type = test_types[0]
        print(f"Using '{ours_type}' as reference test type")

    # Compute mean DI for each group pair, test type, and dataset combination
    grouped_di = df.groupby(['Group Pair', 'Test Type', 'source_dataset', 'target_dataset'])['Disparate Impact'].mean().reset_index()

    # Get unique training (source) and testing (target) datasets
    unique_train = source_datasets
    unique_test = target_datasets

    # Create a color palette for group pairs
    unique_groups = df['Group Pair'].unique()
    group_colors = dict(zip(unique_groups, plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))))

    # 1. FIXED TRAINING, VARYING TESTING
    for train_dataset in unique_train[:3]:  # Limit to 3 training sets
        # Get data for this training dataset
        train_data = grouped_di[grouped_di['source_dataset'] == train_dataset]
        
        # Calculate improvement for each group pair and testing dataset
        improvements = []
        
        for test_dataset in unique_test:
            test_data = train_data[train_data['target_dataset'] == test_dataset]
            
            # Skip if no data for this combination
            if len(test_data) == 0:
                continue
            
            # Pivot to get test types as columns
            pivot_data = test_data.pivot_table(
                index='Group Pair',
                columns='Test Type',
                values='Disparate Impact'
            )
            
            # Skip if reference type not in this dataset
            if ours_type not in pivot_data.columns:
                continue
            
            # Calculate improvement
            other_types = [t for t in pivot_data.columns if t != ours_type]
            if not other_types:
                continue
            
            pivot_data['Max Others'] = pivot_data[other_types].max(axis=1)
            pivot_data['Improvement'] = pivot_data[ours_type] - pivot_data['Max Others']
            
            # Add testing dataset info
            pivot_data['target_dataset'] = test_dataset
            
            # Add to improvements
            for group, row in pivot_data.iterrows():
                if not pd.isna(row['Improvement']):
                    improvements.append({
                        'Group Pair': group,
                        'target_dataset': test_dataset,
                        'Improvement': row['Improvement'],
                        'DI_Ours': row[ours_type],
                        'DI_Others': row['Max Others']
                    })
        
        # Skip if no improvements found
        if not improvements:
            continue
        
        # Convert to DataFrame and sort
        improvement_df = pd.DataFrame(improvements)
        improvement_df = improvement_df.sort_values('Improvement', ascending=False)
        
        # Get top 15 overall
        top_15 = improvement_df.head(15)
        
        # Plot
        plt.figure(figsize=(20, 10))  # Increased figure width
        
        # Group by testing dataset
        for test_dataset in top_15['target_dataset'].unique():
            subset = top_15[top_15['target_dataset'] == test_dataset]
            
            for i, (idx, row) in enumerate(subset.iterrows()):
                x_pos = i * 4  # Increased spacing between bar pairs
                
                # Plot reference method
                plt.bar(x_pos, row['DI_Ours'], width=0.8, 
                        color='green', alpha=0.7, 
                        label=f"{ours_type}" if i == 0 and test_dataset == subset['target_dataset'].iloc[0] else "")
                
                # Plot best other method
                plt.bar(x_pos + 1, row['DI_Others'], width=0.8, 
                        color='gray', alpha=0.7,
                        label="Best Other" if i == 0 and test_dataset == subset['target_dataset'].iloc[0] else "")
                
                # Add group pair label with better formatting
                plt.text(x_pos + 0.5, 0.05, 
                         f"{row['Group Pair']}\n({test_dataset})", 
                         rotation=45,  # Changed from 90 to 45 degrees
                         ha='right',   # Changed alignment
                         va='bottom',
                         fontsize=10,  # Increased font size
                         wrap=True)    # Enable text wrapping
        
        plt.axhline(y=0.8, color='red', linestyle='--', label='80% Rule')
        plt.ylabel('Disparate Impact', fontsize=14)
        plt.title(f'Top Group Pairs: {ours_type} vs Others - Training on {train_dataset}', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.subplots_adjust(bottom=0.2)  # Add more space at the bottom
        plt.tight_layout()
        plt.savefig(f"{comparison_dir}/top_pairs_train_{train_dataset}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. FIXED TESTING, VARYING TRAINING
    for test_dataset in unique_test[:3]:  # Limit to 3 testing sets
        # Get data for this testing dataset
        test_data = grouped_di[grouped_di['target_dataset'] == test_dataset]
        
        # Calculate improvement for each group pair and training dataset
        improvements = []
        
        for train_dataset in unique_train:
            train_subset = test_data[test_data['source_dataset'] == train_dataset]
            
            # Skip if no data for this combination
            if len(train_subset) == 0:
                continue
            
            # Pivot to get test types as columns
            pivot_data = train_subset.pivot_table(
                index='Group Pair',
                columns='Test Type',
                values='Disparate Impact'
            )
            
            # Skip if reference type not in this dataset
            if ours_type not in pivot_data.columns:
                continue
            
            # Calculate improvement
            other_types = [t for t in pivot_data.columns if t != ours_type]
            if not other_types:
                continue
            
            pivot_data['Max Others'] = pivot_data[other_types].max(axis=1)
            pivot_data['Improvement'] = pivot_data[ours_type] - pivot_data['Max Others']
            
            # Add training dataset info
            pivot_data['source_dataset'] = train_dataset
            
            # Add to improvements
            for group, row in pivot_data.iterrows():
                if not pd.isna(row['Improvement']):
                    improvements.append({
                        'Group Pair': group,
                        'source_dataset': train_dataset,
                        'Improvement': row['Improvement'],
                        'DI_Ours': row[ours_type],
                        'DI_Others': row['Max Others']
                    })
        
        # Skip if no improvements found
        if not improvements:
            continue
        
        # Convert to DataFrame and sort
        improvement_df = pd.DataFrame(improvements)
        improvement_df = improvement_df.sort_values('Improvement', ascending=False)
        
        # Get top 15 overall
        top_15 = improvement_df.head(15)
        
        # Plot
        plt.figure(figsize=(20, 10))  # Increased figure width
        
        # Group by training dataset
        for train_dataset in top_15['source_dataset'].unique():
            subset = top_15[top_15['source_dataset'] == train_dataset]
            
            for i, (idx, row) in enumerate(subset.iterrows()):
                x_pos = i * 4  # Increased spacing between bar pairs
                
                # Plot reference method
                plt.bar(x_pos, row['DI_Ours'], width=0.8, 
                        color='green', alpha=0.7, 
                        label=f"{ours_type}" if i == 0 and train_dataset == subset['source_dataset'].iloc[0] else "")
                
                # Plot best other method
                plt.bar(x_pos + 1, row['DI_Others'], width=0.8, 
                        color='gray', alpha=0.7,
                        label="Best Other" if i == 0 and train_dataset == subset['source_dataset'].iloc[0] else "")
                
                # Add group pair label with better formatting
                plt.text(x_pos + 0.5, 0.05, 
                         f"{row['Group Pair']}\n({train_dataset})", 
                         rotation=45,  # Changed from 90 to 45 degrees
                         ha='right',   # Changed alignment
                         va='bottom',
                         fontsize=10,  # Increased font size
                         wrap=True)    # Enable text wrapping
        
        plt.axhline(y=0.8, color='red', linestyle='--', label='80% Rule')
        plt.ylabel('Disparate Impact', fontsize=14)
        plt.title(f'Top Group Pairs: {ours_type} vs Others - Testing on {test_dataset}', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.subplots_adjust(bottom=0.2)  # Add more space at the bottom
        plt.tight_layout()
        plt.savefig(f"{comparison_dir}/top_pairs_test_{test_dataset}.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Separated visualizations created in {comparison_dir}")

# Run the analysis
compare_test_types_by_group_pair()

def create_tradeoff_plots():
    """
    Create tradeoff plots comparing Disparate Impact vs Equal Opportunity for different test types.
    Creates separate plots for each demographic group category.
    """
    # Create output directory
    output_dir = "/home/tpei0009/gaze-demo/gaze-demo/tradeoff_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define bright, distinct colors and markers for test types
    style_dict = {
        'ours': {'color': 'red', 'marker': 'o'},      # Red, Circle
        'pg': {'color': 'blue', 'marker': 's'},  # Blue, Square
        'r18': {'color': 'green', 'marker': '^'},    # Green, Triangle
    }
    
    # Define group categories and their keywords
    group_categories = {
        'Asian': ['asian', 'chinese', 'japanese', 'korean', 'indian'],
        'White': ['white', 'caucasian'],
        'Black': ['black', 'african'],
        'Middle_Eastern': ['middle eastern', 'arab', 'persian']
    }
    
    # Get unique source datasets
    source_datasets = df['source_dataset'].unique()
    
    # Create plots for each demographic category
    for category, keywords in group_categories.items():
        # Filter data for this demographic category
        category_data = df[df['Group Pair'].str.contains('|'.join(keywords), case=False, na=False)]
        
        if category_data.empty:
            print(f"No data found for {category} category")
            continue
            
        # Create a plot for each source dataset
        for source in source_datasets:
            # Filter data for this source dataset
            source_data = category_data[category_data['source_dataset'] == source]
            
            if source_data.empty:
                continue
                
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot each test type
            for test_type in source_data['Test Type'].unique():
                test_data = source_data[source_data['Test Type'] == test_type]
                
                # Get style for this test type
                style = style_dict.get(test_type.lower(), {'color': 'gray', 'marker': 'x'})
                
                # Create scatter plot with both color and shape
                plt.scatter(
                    test_data['ΔTPR (Equal Opportunity)'],
                    test_data['Disparate Impact'],
                    color=style['color'],
                    marker=style['marker'],
                    label=test_type,
                    alpha=0.7,
                    s=150,
                    edgecolors='black',
                    linewidth=1
                )
            
            # Customize plot
            plt.xlabel('Equal Opportunity (ΔTPR)', fontsize=14)
            plt.ylabel('Disparate Impact', fontsize=14)
            plt.title(f'Tradeoff Plot - {category} Groups - Source Dataset: {source}', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f"{output_dir}/tradeoff_{category}_{source}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Tradeoff plots saved to {output_dir}/")

# Run the tradeoff plot creation
create_tradeoff_plots()

def create_aggregated_boxplot():
    """
    Create box plots showing the distribution of fairness metrics across all datasets combined.
    """
    # Create output directory
    output_dir = "/home/tpei0009/gaze-demo/gaze-demo/aggregated_boxplots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique test types
    test_types = df['Test Type'].unique()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create box plot for Disparate Impact
    sns.boxplot(
        x='Test Type',
        y='Disparate Impact',
        data=df,
        palette='Set3'
    )
    
    plt.axhline(y=0.8, color='red', linestyle='--', label='80% Rule')
    plt.xlabel('Test Type', fontsize=14)
    plt.ylabel('Disparate Impact', fontsize=14)
    plt.title('Distribution of Disparate Impact by Test Type (All Datasets)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregated_DI_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create box plot for Equal Opportunity with outliers removed
    plt.figure(figsize=(12, 6))
    
    # Calculate Q1, Q3, and IQR for TPR values
    Q1 = df['ΔTPR (Equal Opportunity)'].quantile(0.25)
    Q3 = df['ΔTPR (Equal Opportunity)'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter out outliers
    filtered_df = df[
        (df['ΔTPR (Equal Opportunity)'] >= Q1 - 1.5 * IQR) & 
        (df['ΔTPR (Equal Opportunity)'] <= Q3 + 1.5 * IQR)
    ]
    
    sns.boxplot(
        x='Test Type',
        y='ΔTPR (Equal Opportunity)',
        data=filtered_df,
        palette='Set3',
        showfliers=False  # Additional safety to ensure no outliers are shown
    )
    
    plt.xlabel('Test Type', fontsize=14)
    plt.ylabel('ΔTPR (Equal Opportunity)', fontsize=14)
    plt.title('Distribution of Equal Opportunity by Test Type (All Datasets, Outliers Removed)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregated_EO_boxplot_no_outliers.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for test_type in test_types:
        test_data = df[df['Test Type'] == test_type]
        filtered_test_data = filtered_df[filtered_df['Test Type'] == test_type]
        
        print(f"\nTest Type: {test_type}")
        print("Disparate Impact:")
        print(f"Mean: {test_data['Disparate Impact'].mean():.3f}")
        print(f"Median: {test_data['Disparate Impact'].median():.3f}")
        print(f"Std: {test_data['Disparate Impact'].std():.3f}")
        print("\nEqual Opportunity (outliers removed):")
        print(f"Mean: {filtered_test_data['ΔTPR (Equal Opportunity)'].mean():.3f}")
        print(f"Median: {filtered_test_data['ΔTPR (Equal Opportunity)'].median():.3f}")
        print(f"Std: {filtered_test_data['ΔTPR (Equal Opportunity)'].std():.3f}")

    print(f"\nBox plots saved to {output_dir}/")

# Run the aggregated box plot analysis
create_aggregated_boxplot()


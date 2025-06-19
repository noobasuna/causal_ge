import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec

# Read both CSV files
df_new1 = pd.read_csv('/home/tpei0009/gaze-demo/gaze-demo/output_data_av.csv')
df_new2 = pd.read_csv('/home/tpei0009/gaze-demo/gaze-demo/fairness_metrics_mpii.csv')  # Add your second CSV file path

# Process first dataset
df_new1["Threshold"] = pd.to_numeric(df_new1["Threshold"], errors="coerce")
low_threshold_df_new1 = df_new1[df_new1["Threshold"] <= 0.5]

# Process second dataset
df_new2["Threshold"] = pd.to_numeric(df_new2["Threshold"], errors="coerce")
low_threshold_df_new2 = df_new2[df_new2["Threshold"] <= 0.5]

# Define color palette and line styles for each model
color_dict = {
    'ours': '#2C3E50',    # dark blue
    'pg': '#27AE60',      # emerald green
    'r18': '#E67E22'      # carrot orange
}

# Define line styles for each model
line_styles = {
    'ours': '-',      # solid line
    'pg': '--',       # dashed line
    'r18': ':'        # dotted line
}

# Create a label mapping dictionary
label_map = {
    'ours': 'Ours',
    'pg': 'PureGaze',
    'r18': 'ResNet18'
}

# Function to create and save plots
def create_and_save_plots(data, filename, dataset_name):
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Plot each model type separately with different line styles
    for model in ['ours', 'pg', 'r18']:
        model_data = data[data['Test Type'] == model]
        
        # Plot for Disparate Impact
        sns.lineplot(data=model_data, x="Threshold", y="Disparate Impact", 
                    color=color_dict[model], linestyle=line_styles[model],
                    label=label_map[model], ax=ax1,
                    linewidth=2.5, marker='o', markersize=6)
        
        # Plot for TPR Differences
        sns.lineplot(data=model_data, x="Threshold", y="ΔTPR (Equal Opportunity)", 
                    color=color_dict[model], linestyle=line_styles[model],
                    label=label_map[model], ax=ax2,
                    linewidth=2.5, marker='o', markersize=6)

    # Configure axes and legends
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Model", title_fontsize=12, fontsize=10,
                 loc='upper left')

    # Set titles and labels
    ax1.set_xlabel("Threshold", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Disparate Impact", fontsize=12, fontweight='bold')
    # ax1.set_title(f"Disparate Impact ({dataset_name})", 
    #              fontsize=14, fontweight='bold', pad=15)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.3)

    ax2.set_xlabel("Threshold", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Equalized Odds", fontsize=12, fontweight='bold')
    # ax2.set_title(f"Equal Opportunity ({dataset_name})", 
    #              fontsize=14, fontweight='bold', pad=15)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Create and save plots for both datasets
create_and_save_plots(low_threshold_df_new1, '/home/tpei0009/gaze-demo/gaze-demo/fairness_metrics_new.pdf', 'all')
create_and_save_plots(low_threshold_df_new2, '/home/tpei0009/gaze-demo/gaze-demo/fairness_metrics_mpii.pdf', 'mpii')

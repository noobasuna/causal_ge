import matplotlib.pyplot as plt
import numpy as np

# Define thresholds
thresholds_low = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
thresholds_high = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Create two figures with 3 rows each
fig1, axes1 = plt.subplots(3, 3, figsize=(20, 18))
fig2, axes2 = plt.subplots(3, 3, figsize=(20, 18))

def plot_differences(data_dict, row_idx, axes1, axes2, title_prefix):
    # Calculate differences for each model
    r18_diff_1_3, r18_diff_1_5, r18_diff_3_5 = calculate_differences(
        data_dict['r18_bin1'], data_dict['r18_bin3'], data_dict['r18_bin5'])
    pg_diff_1_3, pg_diff_1_5, pg_diff_3_5 = calculate_differences(
        data_dict['pg_bin1'], data_dict['pg_bin3'], data_dict['pg_bin5'])
    base_diff_1_3, base_diff_1_5, base_diff_3_5 = calculate_differences(
        data_dict['base_bin1'], data_dict['base_bin3'], data_dict['base_bin5'])

    # Plot for each comparison
    for col, (diff_r18, diff_pg, diff_base, title) in enumerate([
        (r18_diff_1_3, pg_diff_1_3, base_diff_1_3, 'Bin 1-3'),
        (r18_diff_1_5, pg_diff_1_5, base_diff_1_5, 'Bin 1-5'),
        (r18_diff_3_5, pg_diff_3_5, base_diff_3_5, 'Bin 3-5')
    ]):
        # Calculate means for low thresholds
        r18_mean_low = np.mean(diff_r18[:5])
        pg_mean_low = np.mean(diff_pg[:5])
        base_mean_low = np.mean(diff_base[:5])

        # Calculate means for high thresholds
        r18_mean_high = np.mean(diff_r18[5:])
        pg_mean_high = np.mean(diff_pg[5:])
        base_mean_high = np.mean(diff_base[5:])

        # Plot for low thresholds
        ax1 = axes1[row_idx][col]
        ax1.plot(thresholds_low, diff_r18[:5], 'b-', label=f'ResNet18 (mean: {r18_mean_low:.4f})', linewidth=2)
        ax1.plot(thresholds_low, diff_pg[:5], 'r-', label=f'PureGaze18 (mean: {pg_mean_low:.4f})', linewidth=2)
        ax1.plot(thresholds_low, diff_base[:5], 'g-', label=f'Ours (mean: {base_mean_low:.4f})', linewidth=2)
        ax1.set_title(f'{title_prefix} - {title} (0.1-0.5°)')
        ax1.set_xlabel('Threshold (degrees)')
        ax1.set_ylabel('Absolute TPR Difference')
        ax1.grid(True)
        ax1.legend()

        # Plot for high thresholds
        ax2 = axes2[row_idx][col]
        ax2.plot(thresholds_high, diff_r18[5:], 'b-', label=f'ResNet18 (mean: {r18_mean_high:.4f})', linewidth=2)
        ax2.plot(thresholds_high, diff_pg[5:], 'r-', label=f'PureGaze18 (mean: {pg_mean_high:.4f})', linewidth=2)
        ax2.plot(thresholds_high, diff_base[5:], 'g-', label=f'Ours (mean: {base_mean_high:.4f})', linewidth=2)
        ax2.set_title(f'{title_prefix} - {title} (1-5°)')
        ax2.set_xlabel('Threshold (degrees)')
        ax2.set_ylabel('Absolute TPR Difference')
        ax2.grid(True)
        ax2.legend()


def calculate_differences(bin1, bin3, bin5):
    return np.abs(bin1 - bin3), np.abs(bin1 - bin5), np.abs(bin3 - bin5)

# Gaze360 to XGaze data
gaze360_data = {
    # Base model (ours)
    'base_bin1': np.array([0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0002, 0.0005, 0.0010, 0.0023, 0.0054]),
    'base_bin3': np.array([0.0000, 0.0001, 0.0001, 0.0004, 0.0004, 0.0007, 0.0021, 0.0044, 0.0080, 0.0125]),
    'base_bin5': np.array([0.0000, 0.0000, 0.0000, 0.0001, 0.0001, 0.0006, 0.0030, 0.0071, 0.0107, 0.0174]),
    
    # R18 model
    'r18_bin1': np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0005, 0.0013, 0.0032, 0.0068]),
    'r18_bin3': np.array([0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0013, 0.0039, 0.0086, 0.0150, 0.0247]),
    'r18_bin5': np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.0013, 0.0041, 0.0085, 0.0116]),
    
    # PG model
    'pg_bin1': np.array([0.0000, 0.0000, 0.0002, 0.0007, 0.0010, 0.0040, 0.0128, 0.0256, 0.0420, 0.0648]),
    'pg_bin3': np.array([0.0000, 0.0001, 0.0001, 0.0002, 0.0002, 0.0012, 0.0050, 0.0115, 0.0206, 0.0332]),
    'pg_bin5': np.array([0.0000, 0.0001, 0.0002, 0.0006, 0.0007, 0.0016, 0.0060, 0.0120, 0.0217, 0.0343])
}

# MPII to XGaze data
mpii_data = {
    # Base model (ours)
    'base_bin1': np.array([0.0000, 0.0004, 0.0005, 0.0008, 0.0015, 0.0071, 0.0243, 0.0510, 0.0884, 0.1366]),
    'base_bin3': np.array([0.0000, 0.0005, 0.0009, 0.0020, 0.0031, 0.0134, 0.0538, 0.1113, 0.1817, 0.2559]),
    'base_bin5': np.array([0.0000, 0.0000, 0.0005, 0.0014, 0.0034, 0.0103, 0.0398, 0.0923, 0.1594, 0.2346]),
    
    # R18 model
    'r18_bin1': np.array([0.0000, 0.0001, 0.0006, 0.0008, 0.0015, 0.0061, 0.0256, 0.0607, 0.1116, 0.1730]),
    'r18_bin3': np.array([0.0001, 0.0006, 0.0019, 0.0033, 0.0046, 0.0150, 0.0621, 0.1232, 0.1931, 0.2751]),
    'r18_bin5': np.array([0.0000, 0.0004, 0.0008, 0.0013, 0.0030, 0.0121, 0.0484, 0.0980, 0.1665, 0.2410]),
    
    # PG model
    'pg_bin1': np.array([0.0001, 0.0006, 0.0008, 0.0010, 0.0013, 0.0070, 0.0275, 0.0590, 0.1018, 0.1568]),
    'pg_bin3': np.array([0.0002, 0.0008, 0.0017, 0.0027, 0.0035, 0.0151, 0.0529, 0.1176, 0.1976, 0.2822]),
    'pg_bin5': np.array([0.0001, 0.0002, 0.0008, 0.0012, 0.0017, 0.0083, 0.0369, 0.0737, 0.1260, 0.1793])
}

# XGaze to XGaze data
xgaze_data = {
    # Base model (ours)
    'base_bin1': np.array([0.0007, 0.0033, 0.0067, 0.0109, 0.0165, 0.0603, 0.2027, 0.3827, 0.5567, 0.6884]),
    'base_bin3': np.array([0.0015, 0.0058, 0.0122, 0.0204, 0.0297, 0.1055, 0.3243, 0.5356, 0.7081, 0.8206]),
    'base_bin5': np.array([0.0010, 0.0047, 0.0110, 0.0179, 0.0259, 0.1063, 0.3500, 0.5962, 0.7737, 0.8815]),
    
    # R18 model
    'r18_bin1': np.array([0.0130, 0.0451, 0.0990, 0.1644, 0.2381, 0.5897, 0.8508, 0.9219, 0.9505, 0.9669]),
    'r18_bin3': np.array([0.0218, 0.0827, 0.1749, 0.2861, 0.3940, 0.7809, 0.9498, 0.9755, 0.9847, 0.9890]),
    'r18_bin5': np.array([0.0295, 0.1117, 0.2201, 0.3490, 0.4726, 0.8531, 0.9715, 0.9843, 0.9885, 0.9910]),
    
    # PG model
    'pg_bin1': np.array([0.0001, 0.0005, 0.0014, 0.0033, 0.0050, 0.0208, 0.0786, 0.1752, 0.2970, 0.4167]),
    'pg_bin3': np.array([0.0004, 0.0021, 0.0041, 0.0072, 0.0127, 0.0500, 0.1794, 0.3488, 0.5102, 0.6435]),
    'pg_bin5': np.array([0.0004, 0.0016, 0.0044, 0.0082, 0.0129, 0.0521, 0.2125, 0.4301, 0.6234, 0.7575])
}

# Plot all three sets of results
plot_differences(gaze360_data, 0, axes1, axes2, 'Gaze360 to XGaze')
plot_differences(mpii_data, 1, axes1, axes2, 'MPII to XGaze')
plot_differences(xgaze_data, 2, axes1, axes2, 'XGaze to XGaze')

plt.tight_layout()
# Save both figures
fig1.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/eyeheight_bin_fairness_low.pdf')
fig2.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/eyeheight_bin_fairness_high.pdf')

# Print numerical differences for all three datasets
def print_differences(data_dict, title):
    print(f"\n{title} - Numerical differences at specific thresholds:")
    print("Threshold | Model | Bin1-3 | Bin1-5 | Bin3-5")
    print("-" * 50)
    
    for i, t in enumerate(thresholds_low):
        r18_diff_1_3, r18_diff_1_5, r18_diff_3_5 = calculate_differences(
            data_dict['r18_bin1'][i], data_dict['r18_bin3'][i], data_dict['r18_bin5'][i])
        pg_diff_1_3, pg_diff_1_5, pg_diff_3_5 = calculate_differences(
            data_dict['pg_bin1'][i], data_dict['pg_bin3'][i], data_dict['pg_bin5'][i])
        base_diff_1_3, base_diff_1_5, base_diff_3_5 = calculate_differences(
            data_dict['base_bin1'][i], data_dict['base_bin3'][i], data_dict['base_bin5'][i])
            
        print(f"{t:.1f}      | R18   | {r18_diff_1_3:.4f} | {r18_diff_1_5:.4f} | {r18_diff_3_5:.4f}")
        print(f"         | PG    | {pg_diff_1_3:.4f} | {pg_diff_1_5:.4f} | {pg_diff_3_5:.4f}")
        print(f"         | Ours  | {base_diff_1_3:.4f} | {base_diff_1_5:.4f} | {base_diff_3_5:.4f}")
        print("-" * 50)

    for i, t in enumerate(thresholds_high):
        r18_diff_1_3, r18_diff_1_5, r18_diff_3_5 = calculate_differences(
            data_dict['r18_bin1'][i], data_dict['r18_bin3'][i], data_dict['r18_bin5'][i])
        pg_diff_1_3, pg_diff_1_5, pg_diff_3_5 = calculate_differences(
            data_dict['pg_bin1'][i], data_dict['pg_bin3'][i], data_dict['pg_bin5'][i])
        base_diff_1_3, base_diff_1_5, base_diff_3_5 = calculate_differences(
            data_dict['base_bin1'][i], data_dict['base_bin3'][i], data_dict['base_bin5'][i])
            
        print(f"{t:.1f}      | R18   | {r18_diff_1_3:.4f} | {r18_diff_1_5:.4f} | {r18_diff_3_5:.4f}")
        print(f"         | PG    | {pg_diff_1_3:.4f} | {pg_diff_1_5:.4f} | {pg_diff_3_5:.4f}")
        print(f"         | Ours  | {base_diff_1_3:.4f} | {base_diff_1_5:.4f} | {base_diff_3_5:.4f}")
        print("-" * 50)

print_differences(gaze360_data, "Gaze360 to XGaze")
print_differences(mpii_data, "MPII to XGaze")
print_differences(xgaze_data, "XGaze to XGaze")

def find_best_threshold(data_dict):
    # Calculate differences for base model (ours)
    base_diff_1_3, base_diff_1_5, base_diff_3_5 = calculate_differences(
        data_dict['base_bin1'], data_dict['base_bin3'], data_dict['base_bin5'])
    
    # Combine all thresholds and differences
    all_thresholds = np.concatenate([thresholds_low, thresholds_high])
    
    # Find best threshold for each comparison
    best_1_3_idx = np.argmin(base_diff_1_3)
    best_1_5_idx = np.argmin(base_diff_1_5)
    best_3_5_idx = np.argmin(base_diff_3_5)
    
    return {
        'bin1-3': (all_thresholds[best_1_3_idx], base_diff_1_3[best_1_3_idx]),
        'bin1-5': (all_thresholds[best_1_5_idx], base_diff_1_5[best_1_5_idx]),
        'bin3-5': (all_thresholds[best_3_5_idx], base_diff_3_5[best_3_5_idx])
    }

# After plotting, analyze best thresholds
print("\nBest thresholds for 'Ours' model:")
print("=" * 60)

for dataset_name, data in [
    ("Gaze360 to XGaze", gaze360_data),
    ("MPII to XGaze", mpii_data),
    ("XGaze to XGaze", xgaze_data)
]:
    best_results = find_best_threshold(data)
    print(f"\n{dataset_name}:")
    print(f"Bin 1-3: Threshold = {best_results['bin1-3'][0]:.1f}°, Difference = {best_results['bin1-3'][1]:.4f}")
    print(f"Bin 1-5: Threshold = {best_results['bin1-5'][0]:.1f}°, Difference = {best_results['bin1-5'][1]:.4f}")
    print(f"Bin 3-5: Threshold = {best_results['bin3-5'][0]:.1f}°, Difference = {best_results['bin3-5'][1]:.4f}")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('/home/tpei0009/gaze-demo/gaze-demo/attr_file/gender_fairness_results.csv')

# Calculate average differences for each model across all scenarios
avg_diff = df.groupby('Threshold')[['Ours_Diff', 'R18_Diff', 'PG_Diff']].mean()

print("Average Gender Differences Across All Scenarios:")
print(avg_diff)

# Calculate average differences for low thresholds (0.1-0.5)
low_thresh_avg = df[df['Threshold'] <= 0.5].groupby('Threshold')[['Ours_Diff', 'R18_Diff', 'PG_Diff']].mean()

print("\nAverage Gender Differences for Low Thresholds (0.1-0.5):")
print(low_thresh_avg)

# Visualization
plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
sns.lineplot(data=df[df['Threshold'] <= 0.5], x='Threshold', y='Ours_Diff', label='Ours')
sns.lineplot(data=df[df['Threshold'] <= 0.5], x='Threshold', y='R18_Diff', label='ResNet18')
sns.lineplot(data=df[df['Threshold'] <= 0.5], x='Threshold', y='PG_Diff', label='PureGaze')
# plt.title('Gender Differences at Low Thresholds (0.1-0.5)')
plt.xlabel('Threshold')
plt.ylabel('EO Difference')

# plt.subplot(1, 2, 2)
# data_to_plot = df[df['Threshold'] <= 0.5][['Ours_Diff', 'R18_Diff', 'PG_Diff']].melt()
# sns.boxplot(data=data_to_plot, x='variable', y='value')
# plt.title('Distribution of Gender Differences\nat Low Thresholds')
# plt.xlabel('Model')
# plt.ylabel('Absolute Gender Difference')

# plt.tight_layout()
plt.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/gender_fairness_analysis.png')
# fig.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/gender_fairness_analysis.pdf', 
#             format='pdf', 
#             bbox_inches='tight',
#             pad_inches=0.1,
#             dpi=300)
plt.close()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Modify thresholds to only include values up to 1
# thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]

# # Gaze360 to MPII
# gaze360_mpii_male_ours = [0.0, 0.0001, 0.0001, 0.0002, 0.0002, 0.0011, 0.004, 0.0082, 0.0141, 0.023]
# gaze360_mpii_female_ours = [0.0, 0.0, 0.0, 0.0001, 0.0003, 0.0018, 0.0073, 0.016, 0.027, 0.0408]
# gaze360_mpii_male_res = [0.0001, 0.0003, 0.0004, 0.0008, 0.0010, 0.0042, 0.0176, 0.0379, 0.0628, 0.0982]
# gaze360_mpii_female_res = [0.0000, 0.0001, 0.0003, 0.0005, 0.0008, 0.0043, 0.0185, 0.0418, 0.0729, 0.1138]
# gaze360_mpii_male_pg = [0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.0008, 0.0036, 0.0088, 0.0159, 0.0246]
# gaze360_mpii_female_pg = [0.0001, 0.0001, 0.0001, 0.0002, 0.0004, 0.0014, 0.0053, 0.0124, 0.0216, 0.0331]

# # Gaze360 to XGaze
# gaze360_xgaze_male_ours = [0.0, 0.0, 0.0, 0.0, 0.0002, 0.0007, 0.0027, 0.0068, 0.0137, 0.0225]
# gaze360_xgaze_female_ours = [0.0, 0.0, 0.0, 0.0, 0.0001, 0.0005, 0.0019, 0.0039, 0.0068, 0.0111]
# gaze360_xgaze_male_res = [0.0000, 0.0001, 0.0002, 0.0006, 0.0008, 0.0027, 0.0077, 0.0170, 0.0287, 0.0456]
# gaze360_xgaze_female_res = [0.0000, 0.0000, 0.0001, 0.0003, 0.0004, 0.0017, 0.0066, 0.0145, 0.0252, 0.0396]
# gaze360_xgaze_male_pg = [0.0007, 0.0026, 0.0058, 0.0083, 0.0131, 0.0514, 0.1871, 0.3674, 0.5416, 0.6673]
# gaze360_xgaze_female_pg = [0.0002, 0.0011, 0.0029, 0.0058, 0.0093, 0.0387, 0.1518, 0.3072, 0.4614, 0.5950]

# # XGaze to MPII
# xgaze_mpii_male_ours = [0.0001, 0.0005, 0.0009, 0.0014, 0.0021, 0.0113, 0.0487, 0.107, 0.1794, 0.2616]
# xgaze_mpii_female_ours = [0.0001, 0.0004, 0.0014, 0.002, 0.0033, 0.0126, 0.0538, 0.1221, 0.2099, 0.3142]
# xgaze_mpii_male_res = [0.0000, 0.0002, 0.0005, 0.0012, 0.0016, 0.0065, 0.0268, 0.0564, 0.0933, 0.1396]
# xgaze_mpii_female_res = [0.0000, 0.0001, 0.0003, 0.0005, 0.0008, 0.0043, 0.0185, 0.0418, 0.0729, 0.1138]
# xgaze_mpii_male_pg = [0.0000, 0.0002, 0.0005, 0.0012, 0.0016, 0.0065, 0.0268, 0.0564, 0.0933, 0.1396]
# xgaze_mpii_female_pg = [0.0000, 0.0001, 0.0004, 0.0008, 0.0015, 0.0059, 0.0222, 0.0475, 0.0837, 0.1288]

# # XGaze to XGaze
# xgaze_xgaze_male_ours = [0.0009, 0.0038, 0.0079, 0.0134, 0.0212, 0.0827, 0.2812, 0.4912, 0.666, 0.7893]
# xgaze_xgaze_female_ours = [0.0009, 0.0043, 0.0087, 0.0165, 0.0243, 0.0956, 0.3173, 0.5455, 0.7233, 0.843]
# xgaze_xgaze_male_res = [0.0257, 0.0913, 0.1884, 0.2986, 0.4131, 0.7911, 0.9436, 0.9695, 0.9797, 0.9856]
# xgaze_xgaze_female_res = [0.0197, 0.0734, 0.1581, 0.2579, 0.3571, 0.7299, 0.9234, 0.9607, 0.9750, 0.9825]
# xgaze_xgaze_male_pg = [0.0007, 0.0026, 0.0058, 0.0083, 0.0131, 0.0514, 0.1871, 0.3674, 0.5416, 0.6673]
# xgaze_xgaze_female_pg = [0.0002, 0.0011, 0.0029, 0.0058, 0.0093, 0.0387, 0.1518, 0.3072, 0.4614, 0.5950]

# # MPII to MPII
# mpii_mpii_male_res = [0.0063, 0.0272, 0.0626, 0.1074, 0.1589, 0.4626, 0.8284, 0.9402, 0.9714, 0.9815]
# mpii_mpii_female_res = [0.0081, 0.0337, 0.0712, 0.1217, 0.1831, 0.5071, 0.8743, 0.9630, 0.9852, 0.9925]
# mpii_mpii_male_pg = [0.0003, 0.0013, 0.0046, 0.0087, 0.0138, 0.0780, 0.4426, 0.8335, 0.9635, 0.9862]
# mpii_mpii_female_pg = [0.0009, 0.0028, 0.0068, 0.0119, 0.0192, 0.0930, 0.5060, 0.8934, 0.9863, 0.9961]
# mpii_mpii_male_ours = [0.0007, 0.0022, 0.0048, 0.009, 0.0145, 0.0571, 0.2113, 0.3934, 0.5699, 0.7124]
# mpii_mpii_female_ours = [0.0005, 0.0021, 0.0044, 0.0081, 0.0129, 0.0529, 0.1924, 0.3745, 0.5613, 0.7259]

# # MPII to XGaze
# mpii_xgaze_male_res = [0.0001, 0.0003, 0.0007, 0.0012, 0.0023, 0.0105, 0.0406, 0.0870, 0.1530, 0.2363]
# mpii_xgaze_female_res = [0.0000, 0.0003, 0.0012, 0.0020, 0.0032, 0.0129, 0.0498, 0.1032, 0.1679, 0.2386]
# mpii_xgaze_male_pg = [0.0002, 0.0004, 0.0011, 0.0015, 0.0027, 0.0115, 0.0451, 0.0975, 0.1613, 0.2313]
# mpii_xgaze_female_pg = [0.0001, 0.0007, 0.0011, 0.0017, 0.0024, 0.0101, 0.0398, 0.0878, 0.1482, 0.2162]
# mpii_xgaze_male_ours = [0.0002, 0.0005, 0.0015, 0.0019, 0.0028, 0.0113, 0.0409, 0.0882, 0.1472, 0.2153]
# mpii_xgaze_female_ours = [0.0, 0.0001, 0.0007, 0.0013, 0.0019, 0.0088, 0.0376, 0.0826, 0.1374, 0.2]

# # Create a single set of subplots
# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

# # Function to plot data with logarithmic x-axis
# def plot_data(ax, male_ours, female_ours, male_res, female_res, male_pg, female_pg, title):
#     # Take only first 6 values (up to threshold 1) for each array
#     ax.plot(thresholds, np.abs(np.array(male_ours[:7]) - np.array(female_ours[:7])), 'b-o', label='Ours')
#     ax.plot(thresholds, np.abs(np.array(male_res[:7]) - np.array(female_res[:7])), 'g-.^', label='ResNet18')
#     ax.plot(thresholds, np.abs(np.array(male_pg[:7]) - np.array(female_pg[:7])), 'r--s', label='PureGaze')
#     ax.set_title(title)
#     ax.set_xlabel('Threshold (degrees)')
#     ax.set_ylabel('EO')
#     ax.legend()
#     ax.grid(True)
#     ax.set_xscale('log')  # Set x-axis to logarithmic scale

# # Plot all data
# plot_pairs = [
#     (ax1, gaze360_mpii_male_ours, gaze360_mpii_female_ours, gaze360_mpii_male_res, gaze360_mpii_female_res, gaze360_mpii_male_pg, gaze360_mpii_female_pg, 'Gaze360 → MPII'),
#     (ax2, gaze360_xgaze_male_ours, gaze360_xgaze_female_ours, gaze360_xgaze_male_res, gaze360_xgaze_female_res, gaze360_xgaze_male_pg, gaze360_xgaze_female_pg, 'Gaze360 → XGaze'),
#     (ax3, xgaze_mpii_male_ours, xgaze_mpii_female_ours, xgaze_mpii_male_res, xgaze_mpii_female_res, xgaze_mpii_male_pg, xgaze_mpii_female_pg, 'XGaze → MPII'),
#     (ax4, xgaze_xgaze_male_ours, xgaze_xgaze_female_ours, xgaze_xgaze_male_res, xgaze_xgaze_female_res, xgaze_xgaze_male_pg, xgaze_xgaze_female_pg, 'XGaze → XGaze'),
#     (ax5, mpii_mpii_male_ours, mpii_mpii_female_ours, mpii_mpii_male_res, mpii_mpii_female_res, mpii_mpii_male_pg, mpii_mpii_female_pg, 'MPII → MPII'),
#     (ax6, mpii_xgaze_male_ours, mpii_xgaze_female_ours, mpii_xgaze_male_res, mpii_xgaze_female_res, mpii_xgaze_male_pg, mpii_xgaze_female_pg, 'MPII → XGaze')
# ]

# # Plot all data
# for ax, male_ours, female_ours, male_res, female_res, male_pg, female_pg, title in plot_pairs:
#     plot_data(ax, male_ours, female_ours, male_res, female_res, male_pg, female_pg, title)

# plt.tight_layout()

# # Save the figure
# fig.savefig('/home/tpei0009/gaze-demo/gaze-demo/attr_file/new_gender_fairness.pdf')

# # After creating all plots, create a dictionary to store results
# results_data = []

# # Function to calculate difference and store results
# def store_results(source, target, male_ours, female_ours, male_res, female_res, male_pg, female_pg):
#     for i, threshold in enumerate(thresholds):  # This will now only loop through values up to 1
#         row = {
#             'Source': source,
#             'Target': target,
#             'Threshold': threshold,
#             'Ours_Diff': abs(male_ours[i] - female_ours[i]),
#             'R18_Diff': abs(male_res[i] - female_res[i]),
#             'PG_Diff': abs(male_pg[i] - female_pg[i])
#         }
#         results_data.append(row)

# # Store results for each pair
# store_results('Gaze360', 'MPII', gaze360_mpii_male_ours, gaze360_mpii_female_ours, 
#              gaze360_mpii_male_res, gaze360_mpii_female_res, 
#              gaze360_mpii_male_pg, gaze360_mpii_female_pg)

# store_results('Gaze360', 'XGaze', gaze360_xgaze_male_ours, gaze360_xgaze_female_ours,
#              gaze360_xgaze_male_res, gaze360_xgaze_female_res,
#              gaze360_xgaze_male_pg, gaze360_xgaze_female_pg)

# store_results('XGaze', 'MPII', xgaze_mpii_male_ours, xgaze_mpii_female_ours,
#              xgaze_mpii_male_res, xgaze_mpii_female_res,
#              xgaze_mpii_male_pg, xgaze_mpii_female_pg)

# store_results('XGaze', 'XGaze', xgaze_xgaze_male_ours, xgaze_xgaze_female_ours,
#              xgaze_xgaze_male_res, xgaze_xgaze_female_res,
#              xgaze_xgaze_male_pg, xgaze_xgaze_female_pg)

# store_results('MPII', 'MPII', mpii_mpii_male_ours, mpii_mpii_female_ours,
#              mpii_mpii_male_res, mpii_mpii_female_res,
#              mpii_mpii_male_pg, mpii_mpii_female_pg)

# store_results('MPII', 'XGaze', mpii_xgaze_male_ours, mpii_xgaze_female_ours,
#              mpii_xgaze_male_res, mpii_xgaze_female_res,
#              mpii_xgaze_male_pg, mpii_xgaze_female_pg)

# # Convert to pandas DataFrame and save to CSV
# df = pd.DataFrame(results_data)
# df.to_csv('/home/tpei0009/gaze-demo/gaze-demo/attr_file/gender_fairness_results_low1.csv', index=False)
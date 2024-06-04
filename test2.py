import matplotlib.pyplot as plt
import numpy as np

# Data for the combined method with different epochs
combined_data = {
    
    '50 Epochs': {
        905: 33.831074512004854,
        634: 5.985699605941772,
        398: 10.15323394536972,
        176: 19.29336587190628,
        145: 9.506738531589507,
    },
    '100 Epochs': {
        905: 22.16139508485794,
        634: 10.241388416290283,
        398: 10.601877319812774,
        176: 10.975213849544526,
        145: 11.033583879470825,
    },
    '200 Epochs': {
        905: 21.92353789806366,
        634: 5.453200685977936,
        398: 10.403748273849487,
        176: 30.723695945739745,
        145: 10.95587978363037,
    }
}

# Environment IDs
environments = [905, 634, 398, 176, 145]
# Epoch configurations
epochs = ['50 Epochs', '100 Epochs', '200 Epochs']

# # Setting up the plot
# fig, ax = plt.subplots(figsize=(10, 6))

# # Width of the bars and index for the environments
# bar_width = 0.25
# index = np.arange(len(environments))

# # Plotting each epoch configuration
# for i, epoch in enumerate(epochs):
#     times = [combined_data[epoch].get(env, 0) for env in environments]
#     plt.bar(index + i * bar_width, times, bar_width, label=epoch)

# # Finalizing the plot
# plt.xlabel('Environment ID', fontweight='bold', fontsize=12)
# plt.ylabel('Time Taken to Reach the Goal', fontweight='bold', fontsize=12)
# plt.title('Performance Across Different Epochs in Combined Method', fontweight='bold', fontsize=14)
# plt.xticks(index + bar_width, environments, fontweight='bold', fontsize=10)
# plt.legend()

# plt.tight_layout()
# plt.show()

# Calculating the average time to reach the goal for each epoch setting
avg_times_epochs = {
    epoch: np.mean(list(times.values())) for epoch, times in combined_data.items()
}

# Extracting the epoch settings and their corresponding average times
epochs_list = list(avg_times_epochs.keys())
avg_times_epochs_list = list(avg_times_epochs.values())

# Setting up the plot with the specified x-axis and y-axis
plt.figure(figsize=(8, 5))
plt.bar(epochs_list, avg_times_epochs_list, color='salmon')

plt.xlabel('Figure 2: Hybrid Method', fontweight='bold', fontsize=12)
plt.ylabel('Average Time to Reach the Goal', fontweight='bold', fontsize=12)
plt.title('Average Time to Reach the Goal by Epochs Across Environments', fontweight='bold', fontsize=14)
plt.xticks(fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()

# fig, ax1 = plt.subplots(figsize=(12, 7))

# # Plot for average time across epochs
# ax1.bar(epochs_list, avg_times_epochs_list, color='salmon', label='Average Across Environments', width=0.3)
# ax1.set_xlabel('Epochs', fontweight='bold', fontsize=12)
# ax1.set_ylabel('Average Time to Reach the Goal', fontweight='bold', fontsize=12)
# ax1.tick_params(axis='y', labelcolor='black')
# colors = [
#    'Blue',
#     'Green',
#  'Red',
#     'Purple',
#     'Orange'
# ]
# # Creating a second y-axis to plot the individual environment curves
# ax2 = ax1.twinx()
# for i, env in enumerate(environments):
#     times = [combined_data[epoch][env] for epoch in epochs]
#     ax2.plot(epochs, times, color=colors[i], marker='o', linestyle='-', label=f' {environments[i]}')
# ax2.set_ylabel('Time Taken to Reach the Goal', fontweight='bold', fontsize=12)
# ax2.tick_params(axis='y', labelcolor='black')

# # Adding a legend that combines both plots
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper center',fontsize=9)

# plt.title('Combined Performance Across Epochs and Individual Environments', fontweight='bold', fontsize=14)
# plt.xticks(fontweight='bold', fontsize=10)
# plt.tight_layout()
# plt.show()
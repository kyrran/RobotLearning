

import matplotlib.pyplot as plt
import numpy as np



# data = {
#     176: {'Closed Loop': 17.93807301521301, 'Open Loop': 13.559377861022949, 'Hybrid': 10.975213849544526},
#     145: {'Closed Loop': 16.273885834217072, 'Open Loop': 10.125012564659118, 'Hybrid': 11.033583879470825},
#     398: {'Closed Loop': 9.953436064720153, 'Open Loop': 5.56718381643295, 'Hybrid': 10.601877319812774},
#     634: {'Closed Loop': 7.763943076133728, 'Open Loop': 5.862216258049011, 'Hybrid': 10.241388416290283},
#     905: {'Closed Loop': 31.298144721984862, 'Open Loop': 33.87153378725052, 'Hybrid': 22.16139508485794},
# }

data = {
    176: {'Closed Loop': 17.93807301521301, 'Open Loop': 13.559377861022949, 'Hybrid': 30.723695945739745},
    145: {'Closed Loop': 16.273885834217072, 'Open Loop': 10.125012564659118, 'Hybrid': 10.95587978363037},
    398: {'Closed Loop': 9.953436064720153, 'Open Loop': 5.56718381643295, 'Hybrid': 10.403748273849487},
    634: {'Closed Loop': 7.763943076133728, 'Open Loop': 5.862216258049011, 'Hybrid': 5.453200685977936},
    905: {'Closed Loop': 31.298144721984862, 'Open Loop': 33.87153378725052, 'Hybrid': 21.92353789806366},
}

# Environment IDs
environments = list(data.keys())
# Method labels
methods = ['Closed Loop', 'Open Loop', 'Hybrid']


avg_times = {method: np.mean([data[env][method] for env in environments]) for method in methods}

# Extracting the method names and their corresponding average times
methods_list = list(avg_times.keys())
avg_times_list = list(avg_times.values())



colors = ['skyblue', 'lightgreen', 'salmon']

# Adjusting font settings for better readability
plt.figure(figsize=(8, 5))
plt.bar(methods_list, avg_times_list, color=colors)

plt.xlabel('Figure 1: 200 Epoch Setting', fontweight='bold', fontsize=12)
plt.ylabel('Average Time to Reach the Goal', fontweight='bold', fontsize=12)
plt.title('Average Time to Reach the Goal by Method Across All Environments', fontweight='bold', fontsize=14)
plt.xticks(fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()

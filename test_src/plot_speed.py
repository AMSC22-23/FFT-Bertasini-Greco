import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

data_folder = sys.argv[1]

# Read in the data
df1 = pd.read_csv(data_folder + '/scalability_speedup.txt', sep=',')
df2 = pd.read_csv(data_folder + '/scalability_time.txt', sep=',')

# add headers
df1.columns = ['n', 'speedup']
df2.columns = ['n', 'time']

# Calculate the regression line
x = np.array(df1['n'])
y = np.array(df1['speedup'])
m, b = np.polyfit(x, y, 1)
df1['regression'] = m*x + b

# Plot the data and regression line 
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(x, df1['regression'], '-', color='crimson')
ax[0].plot(df1['n'], df1['speedup'], 'o-', color='tan')
ax[0].set_xlabel('Number of cores')
ax[0].set_ylabel('Speedup')
ax[0].set_title('Speedup')
ax[0].grid(True)

ax[1].plot(df2['n'], df2['time'] / 1000, 'o-', color='tan')
ax[1].set_xlabel('Number of samples')
ax[1].set_ylabel('Time (ms)')
# set log base 2 scale
ax[1].set_xscale('log', base=2)
ax[1].set_yscale('log', base=10)
ax[1].set_title('Time')
ax[1].grid(True)

plt.tight_layout()
plt.savefig(data_folder + '/scalability.png')

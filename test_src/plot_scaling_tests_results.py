import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

data_folder = sys.argv[1]

#FFT plots

#1. strong scalability test
file_path = data_folder + '/strong_scalability_test_fft.txt'
data = np.loadtxt(file_path, delimiter=',')
n = data[:, 0]
time = data[:, 1]
speedup = time[0] / time

m, b = np.polyfit(n[:8], speedup[:8], 1)
regression_line = m * n[:8] + b

plt.plot(n, speedup, 'o-', color='tan', label='Data')
plt.plot(n[:8], regression_line, 'k--', label='Regression (first 8 points)')

plt.xlabel('Number of cores')
plt.ylabel('Speedup')
plt.title('Strong Scalability Test for FFT')
plt.grid(True)
plt.xticks(range(int(min(n)), int(max(n)) + 1))
plt.savefig(data_folder + '/strong_scalability_test_fft.png')
plt.show()

#2. Weak scalability test
file_path = data_folder + '/weak_scalability_test_fft.txt'
data = np.loadtxt(file_path, delimiter=',')
n = data[:, 0]
time = data[:, 2]
speedup = time[0]*n/time


plt.plot(n, speedup, 'o-', color='tan', label='Data')

plt.xlabel('Number of cores')
plt.ylabel('Speedup')
plt.title('Weak Scalability Test for FFT')
plt.grid(True)
plt.xticks(range(int(min(n)), int(max(n)) + 1))
plt.savefig(data_folder + '/weak_scalability_test_fft.png')
plt.show()


#DWT plots

#1. strong scalability test
file_path = data_folder + '/strong_scalability_test_dwt.txt'
data = np.loadtxt(file_path, delimiter=',')
n = data[:, 0]
time = data[:, 1]
speedup = time[0] / time

m, b = np.polyfit(n[:8], speedup[:8], 1)
regression_line = m * n[:8] + b

plt.plot(n, speedup, 'o-', color='tan', label='Data')
plt.plot(n[:8], regression_line, 'k--', label='Regression (first 8 points)')

plt.xlabel('Number of cores')
plt.ylabel('Speedup')
plt.title('Strong Scalability Test for DWT')
plt.grid(True)
plt.xticks(range(int(min(n)), int(max(n)) + 1))
plt.savefig(data_folder + '/strong_scalability_test_dwt.png')
plt.show()

#2. Weak scalability test
file_path = data_folder + '/weak_scalability_test_dwt.txt'
data = np.loadtxt(file_path, delimiter=',')
n = data[:, 0]
time = data[:, 2]
speedup = time[0]*n/time


plt.plot(n, speedup, 'o-', color='tan', label='Data')

plt.xlabel('Number of cores')
plt.ylabel('Speedup')
plt.title('Weak Scalability Test for DWT')
plt.grid(True)
plt.xticks(range(int(min(n)), int(max(n)) + 1))
plt.savefig(data_folder + '/weak_scalability_test_dwt.png')
plt.show()




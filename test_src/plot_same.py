import numpy as np
import sys
import os
import matplotlib.pyplot as plt

data_numpy = np.loadtxt(sys.argv[1], delimiter=',', dtype=np.complex_)
data_user = np.loadtxt(sys.argv[2], delimiter=',', dtype=np.complex_)



# use a side by side plot to compare first half of the abs data in blue
width = 1400;
height = 700;
dpi = 100;
fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
ax1 = fig.add_subplot(121)
ax1.plot(np.abs(data_numpy[:len(data_numpy)//2]))
ax1.set_title('numpy')
ax2 = fig.add_subplot(122)
ax2.plot(np.abs(data_user[:len(data_user)//2]))
ax2.set_title('user')
# get path of output file
output_file = os.path.join(os.path.dirname(sys.argv[1]), 'compare_fft.png')
plt.savefig(output_file)

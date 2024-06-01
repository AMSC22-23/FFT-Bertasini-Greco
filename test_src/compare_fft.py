# load two files and compare the data
#
# Usage: python compare_fft.py input.csv output.csv

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def compare_fft(file, reference_file):
    # load the vector of complex numbers from input file
    data = np.loadtxt(file, delimiter=',', dtype=np.complex_)
    # load the vector of complex numbers from output file
    reference_data = np.loadtxt(reference_file, delimiter=',', dtype=np.complex_)

    difference = data - reference_data
    norm = np.linalg.norm(difference)
    print('The relative error is: ', norm/np.linalg.norm(reference_data))

def visual_comparison (file, reference_file):

    data_numpy = np.loadtxt(reference_file, delimiter=',', dtype=np.complex_)
    data_user = np.loadtxt(file, delimiter=',', dtype=np.complex_)

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


if __name__ == '__main__':
    compare_fft(sys.argv[1], sys.argv[2])
    visual_comparison(sys.argv[1], sys.argv[2])
    

# load two files and compare the data
#
# Usage: python compare_fft.py input.csv output.csv

import numpy as np
import sys

def compare_fft(file, reference_file):
    # load the vector of complex numbers from input file
    data = np.loadtxt(file, delimiter=',', dtype=np.complex_)
    # load the vector of complex numbers from output file
    reference_data = np.loadtxt(reference_file, delimiter=',', dtype=np.complex_)

    difference = data - reference_data
    norm = np.linalg.norm(difference)
    print('The relative error is: ', norm/np.linalg.norm(reference_data))

if __name__ == '__main__':
    compare_fft(sys.argv[1], sys.argv[2])
    
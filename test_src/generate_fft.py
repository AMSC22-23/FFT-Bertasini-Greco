import numpy as np
import sys
import os

# load a list of data from input file and calculate the fft and save it to output file
def generate_fft(input_file, output_file):
    # load data from input file
    data = np.loadtxt(input_file, delimiter=',')
    # calculate fft
    fft = np.fft.fft(data)
    # save fft to output file
    np.savetxt(output_file, fft, delimiter=',')
    return fft

if __name__ == '__main__':
    generate_fft(sys.argv[1], sys.argv[2])
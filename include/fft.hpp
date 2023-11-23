#ifndef FFT_DFT_HPP
#define FFT_DFT_HPP

#include <vector>
#include <complex>

std::vector<std::complex<double>>  fft(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> X);

#endif
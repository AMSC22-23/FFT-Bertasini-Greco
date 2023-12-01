#ifndef FFT_DFT_HPP
#define FFT_DFT_HPP

#include <vector>
#include <complex>

auto fft(std::vector<std::complex<double>> x)  -> std::vector<std::complex<double>>;
auto ifft(std::vector<std::complex<double>> X) -> std::vector<std::complex<double>>;

#endif
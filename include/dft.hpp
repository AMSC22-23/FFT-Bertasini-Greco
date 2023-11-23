#ifndef DFT_DFT_HPP
#define DFT_DFT_HPP

#include <vector>
#include <complex>

std::vector<std::complex<double>>  dft(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> idft(std::vector<std::complex<double>> X);

#endif
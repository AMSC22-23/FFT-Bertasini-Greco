#ifndef DFT_DFT_HPP
#define DFT_DFT_HPP

#include <vector>
#include <complex>

auto dft (std::vector<std::complex<double>> x) -> std::vector<std::complex<double>>;
auto idft(std::vector<std::complex<double>> X) -> std::vector<std::complex<double>>;

#endif
#ifndef FFT_HPP
#define FFT_HPP

#include <vector>
#include <complex>

namespace sequential {
    auto fft (std::vector<std::complex<double>> x) -> std::vector<std::complex<double>>;
    auto ifft(std::vector<std::complex<double>> X) -> std::vector<std::complex<double>>;
}

#endif
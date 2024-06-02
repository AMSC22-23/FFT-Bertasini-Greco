#ifndef ITERATIVE_FAST_FOURIER_TRANSFORM_HPP
#define ITERATIVE_FAST_FOURIER_TRANSFORM_HPP

#include "FourierTransform.hpp"

class IterativeFastFourierTransform : public FourierTransform {
    private:
    auto fft(Typedefs::vcpx&, const bool) const -> void;
    int n_cores;
    public:
    IterativeFastFourierTransform(const int n_cores = -1) : FourierTransform(), n_cores(n_cores) {};
    auto set_n_cores(const int n_cores) -> void { this->n_cores = n_cores; };
    auto operator()(Typedefs::vcpx&, const bool) const -> void override;
};

#endif
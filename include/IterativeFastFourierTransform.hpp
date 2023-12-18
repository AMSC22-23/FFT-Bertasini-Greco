#ifndef ITERATIVE_FAST_FOURIER_TRANSFORM_HPP
#define ITERATIVE_FAST_FOURIER_TRANSFORM_HPP

#include <FourierTransform.hpp>

class IterativeFastFourierTransform : public FourierTransform {
    private:
    auto fft(vcpx&, const bool) -> void;
    public:
    IterativeFastFourierTransform(const unsigned int n_cores) : FourierTransform(n_cores) {};
    auto operator()(vcpx&, const bool) -> void override;
};

#endif
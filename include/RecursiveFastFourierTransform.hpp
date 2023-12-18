#ifndef RECURSIVE_FAST_FOURIER_TRANSFORM_HPP
#define RECURSIVE_FAST_FOURIER_TRANSFORM_HPP

#include <FourierTransform.hpp>

class RecursiveFastFourierTransform : public FourierTransform {
    private:
    auto fft(vcpx&, const bool) -> void;
    public:
    RecursiveFastFourierTransform() : FourierTransform(-1) {};
    auto operator()(vcpx&, const bool) -> void override;
};

#endif
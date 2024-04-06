#ifndef RECURSIVE_FAST_FOURIER_TRANSFORM_HPP
#define RECURSIVE_FAST_FOURIER_TRANSFORM_HPP

#include <FourierTransform.hpp>

class RecursiveFastFourierTransform : public FourierTransform {
    private:
    auto fft(Typedefs::vcpx&, const bool) const -> void;
    public:
    RecursiveFastFourierTransform() : FourierTransform() {};
    auto operator()(Typedefs::vcpx&, const bool) const -> void override;
};

#endif
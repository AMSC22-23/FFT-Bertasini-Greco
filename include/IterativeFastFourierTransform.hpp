#ifndef ITERATIVE_FAST_FOURIER_TRANSFORM_HPP
#define ITERATIVE_FAST_FOURIER_TRANSFORM_HPP

#include <FourierTransform.hpp>

class IterativeFastFourierTransform : public FourierTransform {
    private:
    auto fft(Typedefs::vcpx&, const bool) const -> void;
    public:
    IterativeFastFourierTransform(const int n_cores) : FourierTransform(n_cores) {};
    auto operator()(Typedefs::vcpx&, const bool) const -> void override;
};

#endif
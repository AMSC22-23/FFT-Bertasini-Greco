#ifndef DISCRETE_FOURIER_TRANSFORM_HPP
#define DISCRETE_FOURIER_TRANSFORM_HPP

#include <FourierTransform.hpp>

class DiscreteFourierTransform : public FourierTransform {
    private:
    auto dft(vcpx&, const bool) const -> void;
    public:
    DiscreteFourierTransform() : FourierTransform(-1) {};
    auto operator()(vcpx&, const bool) const -> void override;
};

#endif
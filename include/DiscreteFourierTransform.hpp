#ifndef DISCRETE_FOURIER_TRANSFORM_HPP
#define DISCRETE_FOURIER_TRANSFORM_HPP

#include <FourierTransform.hpp>

class DiscreteFourierTransform : public FourierTransform {
    private:
    auto dft(Typedefs::vcpx&, const bool) const -> void;
    public:
    DiscreteFourierTransform() : FourierTransform() {};
    auto operator()(Typedefs::vcpx&, const bool) const -> void override;
};

#endif
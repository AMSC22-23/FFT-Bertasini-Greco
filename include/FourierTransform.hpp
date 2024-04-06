#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <typedefs.hpp>

class FourierTransform {
    protected:
    public:
    FourierTransform() {};
    virtual auto operator()(Typedefs::vcpx&, const bool) const -> void = 0;
    virtual ~FourierTransform() = default;
};

#endif
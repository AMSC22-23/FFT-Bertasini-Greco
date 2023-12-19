#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <typedefs.hpp>

class FourierTransform {
    protected:
    int n_cores;
    public:
    FourierTransform(const int n_cores) : n_cores(n_cores) {};
    virtual auto operator()(Typedefs::vcpx&, const bool) const -> void = 0;
    virtual ~FourierTransform() = default;
    auto set_n_cores(const int n_cores) -> void { this->n_cores = n_cores; };
};

#endif
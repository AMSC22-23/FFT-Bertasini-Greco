#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <complex>
#include <vector>
#include <functional>

namespace Typedefs {
    using DType  = double;
    using vec    = std::vector<DType>;
    using cpx    = std::complex<DType>;
    using vcpx   = std::vector<cpx>;
    using vcpx2D = std::vector<std::vector<cpx>>;
    using vcpx3D = std::vector<std::vector<std::vector<cpx>>>;
    using ft     = std::function<auto (vcpx&, const bool, const int) -> void>;
}

#endif